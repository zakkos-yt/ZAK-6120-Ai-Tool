"""Da un link YouTube a un articolo Markdown, interamente in locale.

Pipeline:
    yt-dlp -> Faster Whisper large-v3 -> Ollama qwen36-35b-a3b-q4km:latest -> file .md

Installazione nel venv:
    pip install -U flask yt-dlp faster-whisper ollama \
        nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"

Avvio:
    python3 youtube_articolo_md.py

Interfaccia:
    http://127.0.0.1:5000

Non usa WordPress, non contiene credenziali e cancella l'audio temporaneo.
"""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from urllib.parse import parse_qs, urlparse


def bootstrap_nvidia_libraries() -> None:
    """Riavvia Python con cuBLAS/cuDNN del venv nel loader path.

    CTranslate2 cerca le librerie all'avvio del processo. Questo evita di dover
    esportare LD_LIBRARY_PATH a mano a ogni attivazione del venv.
    """
    if os.environ.get("ZAK_WHISPER_LD_READY") == "1":
        return
    try:
        import nvidia.cublas
        import nvidia.cudnn

        paths = [
            str(Path(next(iter(nvidia.cublas.__path__))) / "lib"),
            str(Path(next(iter(nvidia.cudnn.__path__))) / "lib"),
        ]
    except (ImportError, StopIteration):
        return

    current = os.environ.get("LD_LIBRARY_PATH", "")
    existing = [item for item in current.split(":") if item]
    merged = paths + [item for item in existing if item not in paths]
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = ":".join(merged)
    env["ZAK_WHISPER_LD_READY"] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


bootstrap_nvidia_libraries()

import ollama
import yt_dlp
from faster_whisper import WhisperModel
from flask import Flask, jsonify, render_template_string, request, send_file


OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen36-35b-a3b-q4km:latest")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "16384"))
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "large-v3")
WHISPER_GPU = int(os.getenv("WHISPER_GPU", "1"))
MAX_TRANSCRIPT_CHARS = int(os.getenv("MAX_TRANSCRIPT_CHARS", "50000"))
OUTPUT_ROOT = Path(os.getenv("OUTPUT_DIR", "articoli_generati")).resolve()

app = Flask(__name__)
ollama_client = ollama.Client(host=OLLAMA_HOST)

job_lock = threading.Lock()
job = {
    "id": None,
    "running": False,
    "phase": "idle",
    "message": "Pronto.",
    "progress": 0,
    "error": None,
    "article_path": None,
    "result_dir": None,
    "stats": None,
}
whisper_instance = None


HTML_PAGE = """
<!doctype html>
<html lang="it">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>YouTube → articolo Markdown</title>
  <style>
    :root { color-scheme:dark; --bg:#11111b; --panel:#1e1e2e; --box:#181825;
      --text:#cdd6f4; --muted:#a6adc8; --blue:#89b4fa; --green:#a6e3a1;
      --red:#f38ba8; --track:#45475a; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font:16px system-ui,sans-serif; }
    main { max-width:900px; margin:40px auto; padding:0 20px; }
    .panel { background:var(--panel); padding:28px; border-radius:14px;
      box-shadow:0 10px 35px #0006; }
    h1 { margin-top:0; color:var(--blue); }
    .subtitle { color:var(--muted); }
    .row { display:flex; gap:10px; margin-top:22px; }
    input { flex:1; min-width:0; padding:13px; border:1px solid #585b70;
      border-radius:7px; background:var(--box); color:var(--text); font-size:1rem; }
    button, .download { border:0; border-radius:7px; padding:13px 18px;
      background:var(--blue); color:#11111b; font-weight:750; cursor:pointer;
      text-decoration:none; text-align:center; }
    button:disabled { opacity:.45; cursor:wait; }
    .progress { height:14px; margin-top:22px; border-radius:99px;
      background:var(--track); overflow:hidden; }
    .bar { height:100%; width:0; background:linear-gradient(90deg,#89b4fa,#a6e3a1);
      transition:width .35s ease; }
    #status { margin:12px 0 0; min-height:24px; color:var(--muted); }
    #details { margin-top:18px; padding:14px; border-radius:8px; background:var(--box);
      white-space:pre-wrap; line-height:1.55; }
    #result { margin-top:20px; padding:18px; border-left:4px solid var(--green);
      border-radius:8px; background:var(--box); }
    .ok { color:var(--green) !important; } .err { color:var(--red) !important; }
    .download { display:inline-block; margin-top:10px; background:var(--green); }
    code { color:#fab387; }
    @media (max-width:700px) { .row { flex-direction:column; } }
  </style>
</head>
<body><main><section class="panel">
  <h1>YouTube → articolo Markdown</h1>
  <p class="subtitle">Whisper <b>{{ whisper }}</b> sulla GPU {{ gpu }} →
     <b>{{ ollama }}</b> senza thinking. Audio eliminato automaticamente.</p>
  <div class="row">
    <input id="url" placeholder="https://youtu.be/…" autocomplete="off">
    <button id="start">Genera articolo</button>
  </div>
  <div class="progress"><div class="bar" id="bar"></div></div>
  <div id="status">Pronto.</div>
  <div id="details" hidden></div>
  <div id="result" hidden>
    <strong>Articolo completato.</strong><br>
    <a class="download" id="download" href="#">Scarica il file Markdown</a>
  </div>
</section></main>
<script>
const start = document.getElementById('start');
const statusBox = document.getElementById('status');
const bar = document.getElementById('bar');
const details = document.getElementById('details');
const result = document.getElementById('result');
let timer = null;

async function poll(jobId) {
  try {
    const response = await fetch('/api/status?id=' + encodeURIComponent(jobId));
    const data = await response.json();
    bar.style.width = data.progress + '%';
    statusBox.textContent = data.message;
    if (data.error) {
      statusBox.className = 'err';
      details.hidden = false; details.textContent = data.error;
      start.disabled = false; clearInterval(timer); return;
    }
    if (!data.running && data.phase === 'done') {
      statusBox.className = 'ok';
      details.hidden = false; details.textContent = data.stats;
      document.getElementById('download').href = '/api/download?id=' + encodeURIComponent(jobId);
      result.hidden = false; start.disabled = false; clearInterval(timer);
    }
  } catch (error) {
    statusBox.className = 'err'; statusBox.textContent = 'Errore nel controllo: ' + error.message;
    start.disabled = false; clearInterval(timer);
  }
}

start.onclick = async () => {
  const url = document.getElementById('url').value.trim();
  if (!url) { alert('Incolla un link YouTube.'); return; }
  start.disabled = true; result.hidden = true; details.hidden = true;
  statusBox.className = ''; statusBox.textContent = 'Avvio…'; bar.style.width = '2%';
  try {
    const response = await fetch('/api/start', {method:'POST',
      headers:{'Content-Type':'application/json'}, body:JSON.stringify({url})});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Impossibile avviare');
    timer = setInterval(() => poll(data.id), 1000); poll(data.id);
  } catch (error) {
    statusBox.className = 'err'; statusBox.textContent = 'Errore: ' + error.message;
    start.disabled = false;
  }
};
</script></body></html>
"""


def update(**values) -> None:
    with job_lock:
        job.update(values)


def response_value(response, key: str, default=None):
    value = getattr(response, key, None)
    if value is None and isinstance(response, dict):
        value = response.get(key)
    return default if value is None else value


def extract_video_id(url_or_id: str) -> str:
    value = url_or_id.strip()
    parsed = urlparse(value)
    if parsed.hostname in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.strip("/").split("/")[0]
    elif parsed.hostname and "youtube.com" in parsed.hostname:
        video_id = parse_qs(parsed.query).get("v", [""])[0]
        if not video_id and parsed.path.startswith("/shorts/"):
            video_id = parsed.path.split("/")[2]
    else:
        video_id = value
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
        raise ValueError("URL o ID YouTube non valido")
    return video_id


def safe_name(value: str) -> str:
    value = re.sub(r"[^\w.-]+", "_", value, flags=re.UNICODE).strip("_.")
    return value[:90] or "articolo"


def ts(seconds: float) -> str:
    total = round(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def get_whisper() -> WhisperModel:
    global whisper_instance
    if whisper_instance is None:
        update(message=f"Caricamento Whisper {WHISPER_MODEL} sulla GPU {WHISPER_GPU}…", progress=24)
        whisper_instance = WhisperModel(
            WHISPER_MODEL,
            device="cuda",
            device_index=WHISPER_GPU,
            compute_type="float16",
        )
    return whisper_instance


def download_audio(url: str, temp_dir: Path) -> tuple[Path, dict]:
    def hook(data: dict) -> None:
        if data.get("status") != "downloading":
            return
        downloaded = data.get("downloaded_bytes") or 0
        total = data.get("total_bytes") or data.get("total_bytes_estimate") or 0
        if total:
            pct = min(100, downloaded * 100 / total)
            update(message=f"Download audio: {pct:.0f}%", progress=5 + int(pct * 0.15))

    options = {
        "format": "bestaudio/best",
        "outtmpl": str(temp_dir / "audio.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "progress_hooks": [hook],
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        requested = info.get("requested_downloads") or []
        candidate = requested[0].get("filepath") if requested else None
        audio_path = Path(candidate) if candidate else Path(ydl.prepare_filename(info))
    if not audio_path.is_file():
        matches = [path for path in temp_dir.glob("audio.*") if path.is_file()]
        if not matches:
            raise RuntimeError("yt-dlp non ha prodotto il file audio")
        audio_path = matches[0]
    return audio_path, info


def transcribe(audio_path: Path) -> tuple[str, str, float, float]:
    model = get_whisper()
    update(phase="transcribe", message="Analisi audio e avvio trascrizione…", progress=27)
    started = time.monotonic()
    segments_iter, info = model.transcribe(
        str(audio_path), language="it", task="transcribe", beam_size=5,
        vad_filter=True, vad_parameters={"min_silence_duration_ms": 500},
        condition_on_previous_text=True,
    )
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    plain_parts = []
    timed_parts = []
    for segment in segments_iter:
        text = segment.text.strip()
        if not text:
            continue
        plain_parts.append(text)
        timed_parts.append(f"[{ts(segment.start)} → {ts(segment.end)}] {text}")
        pct = min(1.0, segment.end / duration) if duration else 0.0
        update(
            message=f"Trascrizione Whisper: {pct * 100:.0f}% — {len(plain_parts)} segmenti",
            progress=28 + int(pct * 37),
        )
    if not plain_parts:
        raise RuntimeError("Whisper non ha prodotto alcun testo")
    elapsed = time.monotonic() - started
    return " ".join(plain_parts), "\n".join(timed_parts), elapsed, duration


def article_prompt(video_title: str, transcript: str) -> str:
    return f"""
Trasforma la trascrizione Whisper di un video YouTube in un articolo tecnico in
italiano, completo, leggibile e fedele all'autore.

Titolo originale del video (usalo solo per comprendere il contesto e i nomi,
non come fonte autonoma di fatti): {video_title}

REGOLE TASSATIVE
- Usa esclusivamente informazioni sostenute dalla trascrizione.
- Correggi errori fonetici soltanto quando la correzione è praticamente certa.
- Se un dettaglio resta ambiguo, omettilo: non inventare capacità, frequenze,
  generazioni, dimensioni, strumenti, comandi o combinazioni di tasti.
- Non mostrare parole trascritte male né inserire note editoriali o dubbi.
- Elimina sigla, musica, saluti, inviti a iscriversi, volgarità, ripetizioni e
  frammenti incomprensibili privi di contenuto tecnico.
- Conserva procedura, componenti, risultati misurati, opinioni e conclusioni.
- Distingui le opinioni dell'autore dai fatti e non aggiungere consigli esterni.

FORMATO
- Prima riga: titolo Markdown informativo e non clickbait, preceduto da #.
- Usa sezioni ## e ### soltanto quando migliorano la struttura.
- Usa elenchi con moderazione.
- Restituisci esclusivamente l'articolo Markdown.

TRASCRIZIONE WHISPER:
---
{transcript}
---
"""


def generate_article(video_title: str, transcript: str) -> tuple[str, dict, float]:
    update(phase="article", message=f"Generazione articolo con {OLLAMA_MODEL}…", progress=68)
    started = time.monotonic()
    chunks = []
    final_metrics = {}
    stream = ollama_client.generate(
        model=OLLAMA_MODEL,
        prompt=article_prompt(video_title, transcript),
        think=False,
        stream=True,
        options={"temperature": 0.2, "num_ctx": OLLAMA_NUM_CTX},
    )
    for part in stream:
        piece = response_value(part, "response", "")
        if piece:
            chunks.append(piece)
            chars = sum(len(item) for item in chunks)
            update(message=f"Qwen sta scrivendo… {chars:,} caratteri", progress=min(97, 70 + chars // 220))
        if response_value(part, "done", False):
            final_metrics = {
                "eval_count": response_value(part, "eval_count"),
                "eval_duration": response_value(part, "eval_duration"),
            }
    article = "".join(chunks).strip()
    if article.startswith("```markdown") and article.endswith("```"):
        article = article[len("```markdown"): -3].strip()
    if not article.startswith("# "):
        raise RuntimeError("Qwen non ha restituito un articolo Markdown valido")
    return article, final_metrics, time.monotonic() - started


def worker(job_id: str, url: str) -> None:
    total_started = time.monotonic()
    try:
        video_id = extract_video_id(url)
        update(phase="download", message="Recupero informazioni e download dell'audio…", progress=4)
        with tempfile.TemporaryDirectory(prefix="zak_audio_") as temporary:
            temp_dir = Path(temporary)
            download_started = time.monotonic()
            audio_path, info = download_audio(url, temp_dir)
            download_time = time.monotonic() - download_started
            video_title = str(info.get("title") or video_id)
            update(message="Audio scaricato. Avvio Whisper…", progress=22)
            transcript, timed, whisper_time, audio_duration = transcribe(audio_path)

            if len(transcript) > MAX_TRANSCRIPT_CHARS:
                raise RuntimeError(
                    f"Trascrizione troppo lunga ({len(transcript):,} caratteri): "
                    f"limite attuale {MAX_TRANSCRIPT_CHARS:,}. Nessun testo è stato tagliato."
                )
            article, metrics, article_time = generate_article(video_title, transcript)

        # Uscendo dal blocco TemporaryDirectory l'audio viene cancellato.
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = safe_name(video_title)
        result_dir = OUTPUT_ROOT / f"{base}_{video_id}_{stamp}"
        result_dir.mkdir(parents=True, exist_ok=False)
        article_path = result_dir / f"{base}.md"
        transcript_path = result_dir / f"{base}.trascrizione.txt"
        timed_path = result_dir / f"{base}.timestamp.txt"
        stats_path = result_dir / f"{base}.risultati.json"
        article_path.write_text(article + "\n", encoding="utf-8")
        transcript_path.write_text(transcript + "\n", encoding="utf-8")
        timed_path.write_text(timed + "\n", encoding="utf-8")

        eval_count = metrics.get("eval_count")
        eval_duration = metrics.get("eval_duration")
        speed = (
            round(eval_count / (eval_duration / 1_000_000_000), 2)
            if eval_count and eval_duration else None
        )
        total_time = time.monotonic() - total_started
        data = {
            "video_id": video_id,
            "video_title": video_title,
            "source_url": url,
            "whisper_model": WHISPER_MODEL,
            "whisper_gpu": WHISPER_GPU,
            "ollama_model": OLLAMA_MODEL,
            "thinking": False,
            "audio_duration_seconds": round(audio_duration, 1),
            "transcript_words": len(transcript.split()),
            "article_words": len(article.split()),
            "download_seconds": round(download_time, 1),
            "transcription_seconds": round(whisper_time, 1),
            "article_seconds": round(article_time, 1),
            "total_seconds": round(total_time, 1),
            "generated_tokens": eval_count,
            "generation_tokens_per_second": speed,
            "temporary_audio_deleted": True,
        }
        stats_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        stats_text = (
            f"Video: {video_title}\n"
            f"Durata audio: {audio_duration / 60:.1f} minuti\n"
            f"Trascrizione: {data['transcript_words']:,} parole in {whisper_time:.1f} s\n"
            f"Articolo: {data['article_words']:,} parole in {article_time:.1f} s\n"
            f"Velocità Qwen: {speed if speed else 'n/d'} token/s\n"
            f"Tempo totale: {total_time:.1f} secondi\n"
            f"Cartella: {result_dir}"
        )
        update(
            running=False, phase="done", message="Completato: audio temporaneo eliminato.",
            progress=100, article_path=str(article_path), result_dir=str(result_dir),
            stats=stats_text,
        )
    except Exception as exc:
        traceback.print_exc()
        update(
            running=False, phase="error", message="Elaborazione interrotta.",
            error=f"{type(exc).__name__}: {exc}", progress=0,
        )


@app.get("/")
def index():
    return render_template_string(
        HTML_PAGE, whisper=WHISPER_MODEL, gpu=WHISPER_GPU, ollama=OLLAMA_MODEL
    )


@app.post("/api/start")
def start_job():
    payload = request.get_json(silent=True) or {}
    url = str(payload.get("url") or "").strip()
    try:
        extract_video_id(url)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    with job_lock:
        if job["running"]:
            return jsonify({"error": "C'è già un'elaborazione in corso."}), 409
        job_id = uuid.uuid4().hex
        job.update({
            "id": job_id, "running": True, "phase": "starting",
            "message": "Avvio elaborazione…", "progress": 2, "error": None,
            "article_path": None, "result_dir": None, "stats": None,
        })
    threading.Thread(target=worker, args=(job_id, url), daemon=True).start()
    return jsonify({"id": job_id})


@app.get("/api/status")
def status():
    requested_id = request.args.get("id")
    with job_lock:
        if not requested_id or requested_id != job["id"]:
            return jsonify({"error": "Elaborazione non trovata."}), 404
        return jsonify({
            "running": job["running"], "phase": job["phase"],
            "message": job["message"], "progress": job["progress"],
            "error": job["error"], "stats": job["stats"],
        })


@app.get("/api/download")
def download():
    requested_id = request.args.get("id")
    with job_lock:
        path = job["article_path"] if requested_id == job["id"] else None
    if not path or not Path(path).is_file():
        return jsonify({"error": "Articolo non disponibile."}), 404
    return send_file(path, as_attachment=True, download_name=Path(path).name)


if __name__ == "__main__":
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"[*] Interfaccia: http://127.0.0.1:5000")
    print(f"[*] Output: {OUTPUT_ROOT}")
    print(f"[*] Whisper: {WHISPER_MODEL} su GPU {WHISPER_GPU}")
    print(f"[*] Ollama: {OLLAMA_MODEL}, thinking disattivato")
    app.run(host="127.0.0.1", port=5000, debug=False, threaded=True)
