import streamlit as st
import subprocess
import requests
import os
import cpuinfo
import platform
import yt_dlp
from datetime import datetime
from youtube_comment_downloader import YoutubeCommentDownloader
from urllib.parse import urlparse
import time
from itertools import islice

# --- CONFIGURAZIONE ---
MODEL_NAME = "qwen36-35b-a3b-q4km:latest"
OLLAMA_BASE_URL = "http://localhost:11434"
REQUEST_TIMEOUT = 600
MAX_COMMENTS_PER_VIDEO = 5000  # Limite per non superare context window

st.set_page_config(page_title="Zakkos Analyze v3+", page_icon="🚀", layout="wide")


# --- FUNZIONI DI UTILITÀ ---

def validate_url(url: str) -> bool:
    """Valida se l'URL è un URL YouTube valido."""
    try:
        parsed = urlparse(url)
        return any(x in parsed.netloc for x in ['youtube.com', 'www.youtube.com'])
    except Exception:
        return False


def safe_get(data, key, default=None):
    """Sicuro accesso a dizionari con fallback."""
    if isinstance(data, dict):
        return data.get(key, default)
    return default


# --- FUNZIONI CORE CON CACHE ---

@st.cache_data(ttl=3600)  # Cache per 1 ora
def get_best_videos_cached(url: str, scan_limit: int, top_count: int) -> list:
    """Recupera i migliori video con caching."""
    channel_url = url.split('/videos')[0].split('/featured')[0].rstrip('/')
    target_url = f"{channel_url}/videos"

    ydl_opts_list = {
        'quiet': True,
        'extract_flat': True,
        'playlistend': scan_limit,
        'no_warnings': True,
    }

    videos_raw = []
    try:
        with yt_dlp.YoutubeDL(ydl_opts_list) as ydl:
            result = ydl.extract_info(target_url, download=False)
            if 'entries' in result:
                videos_raw = [e for e in result['entries'] 
                              if e and (e.get('duration') is None or e.get('duration') >= 60)]
    except Exception as e:
        raise RuntimeError(f"Errore nel recupero video: {str(e)}")

    if not videos_raw:
        return []

    refined_videos = []
    
    for i, entry in enumerate(videos_raw):
        v_id = safe_get(entry, 'id')
        if not v_id:
            continue
            
        try:
            with yt_dlp.YoutubeDL({'quiet': True, 'no_warnings': True, 'format': 'best'}) as ydl_single:
                info = ydl_single.extract_info(f"https://www.youtube.com/watch?v={v_id}", 
                                               download=False, process=False)
                
                views = int(info.get('view_count') or 0)
                refined_videos.append({
                    'id': v_id,
                    'title': info.get('title') or "Senza titolo",
                    'url': f"https://www.youtube.com/watch?v={v_id}",
                    'views': views,
                    'thumbnail': f"https://i.ytimg.com/vi/{v_id}/hqdefault.jpg"
                })
            
            # Rate limiting per evitare ban
            time.sleep(0.1)
        except Exception:
            continue

    refined_videos.sort(key=lambda x: x['views'], reverse=True)
    return refined_videos[:top_count]


@st.cache_data(ttl=3600)  # Cache per 1 ora
def get_video_details_cached(video_id: str) -> dict:
    """Recupera dettagli del video con caching."""
    cmd = [
        "yt-dlp", 
        "--get-filename", 
        "-o", "%(upload_date)s", 
        f"https://www.youtube.com/watch?v={video_id}"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8', timeout=30)
        raw_date = result.stdout.strip()
        if raw_date and len(raw_date) == 8:
            date_obj = datetime.strptime(raw_date, "%Y%m%d")
            return {'date': date_obj.strftime("%d/%m/%Y")}
        return {'date': "Data N/D"}
    except Exception:
        return {'date': "Data N/D"}


@st.cache_data(ttl=3600)  # Cache per 1 ora
def get_all_comments_cached(url: str) -> tuple:
    """Recupera commenti con caching e limite."""
    downloader = YoutubeCommentDownloader()
    try:
        comments = downloader.get_comments_from_url(url, sort_by=1)
        # Limita i commenti per non superare context window
        limited_comments = islice(comments, MAX_COMMENTS_PER_VIDEO)
        extracted = [f"- {c.get('text', '')}" for c in limited_comments if c.get('text')]
        return "\n".join(extracted), len(extracted)
    except Exception:
        return "", 0


def query_ollama(prompt: str, base_url: str = OLLAMA_BASE_URL) -> str:
    """Query all'API di Ollama con gestione errori."""
    try:
        res = requests.post(
            f"{base_url}/api/generate", 
            json={
                "model": MODEL_NAME, 
                "prompt": prompt, 
                "stream": False
            },
            timeout=REQUEST_TIMEOUT
        )
        res.raise_for_status()
        return res.json()['response']
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Impossibile connettersi a Ollama. Verifica che sia in esecuzione.")
    except requests.exceptions.Timeout:
        raise RuntimeError("Timeout nella richiesta AI. Riprova più tardi.")
    except Exception as e:
        return f"Errore nella query AI: {str(e)}"


# --- FUNZIONI SISTEMA ---

def get_hardware_info():
    """Recupera info hardware con gestione errori."""
    cpu_name = "CPU Generica"
    gpu_name = "Grafica Integrata"
    
    try:
        raw_cpu = cpuinfo.get_cpu_info()['brand_raw']
        cpu_name = raw_cpu.replace("(R)", "").replace("(TM)", "")                          .replace("Intel", "").replace("CPU", "")                          .split("@")[0].strip()
    except Exception:
        cpu_name = platform.processor() or "CPU Generica"
    
    if platform.system() == "Windows":
        try:
            cmd = "wmic path win32_VideoController get name"
            gpu_name = subprocess.check_output(cmd, shell=True).decode().splitlines()[1].strip()
        except Exception:
            pass
    else:
        try:
            res = subprocess.check_output("lspci | grep -E 'VGA|3D'", shell=True).decode()
            if "[" in res:
                gpu_name = res.split("[")[1].split("]")[0]
            else:
                gpu_name = res.split(":")[-1].strip()
            gpu_name = gpu_name.replace("NVIDIA Corporation", "").replace("Lite Hash Rate", "")                               .strip()
        except Exception:
            pass
            
    return cpu_name, gpu_name


# --- INIZIALIZZAZIONE SESSION STATE ---
if 'report_generated' not in st.session_state:
    st.session_state.report_generated = False

# --- SIDEBAR ---
with st.sidebar:
    st.header("🎯 Target ...")
    channel_url = st.text_input("Scrivi URL Canale:", "https://www.youtube.com/@zakkos")
    
    # Validazione URL in tempo reale
    if channel_url and not validate_url(channel_url):
        st.error("URL YouTube non valido!")
    
    st.divider()
    
    ollama_url = st.text_input("Ollama API URL:", value=OLLAMA_BASE_URL)
    st.info(f"🤖 **Modello AI:** {MODEL_NAME}")
    
    scan_depth = st.radio(
        "Estensione ricerca:", 
        ["Ultimi 10 video", "Ultimi 50 video", "Ultimi 100 video", "Tutto il canale"], 
        index=0
    )
    depth_map = {
        "Ultimi 10 video": 10,
        "Ultimi 50 video": 50, 
        "Ultimi 100 video": 100, 
        "Tutto il canale": 5000
    }
    
    top_n = st.number_input("Quanti video TOP analizzare?", 1, 20, 5)


# --- MAIN UI ---
st.title("📈 Zakkos Analyze v3+")

cpu_sys, gpu_sys = get_hardware_info()
st.markdown(f"**Sistema:** `{cpu_sys}` | **GPU:** `{gpu_sys}`")

# --- ESECUZIONE ---

def run_analysis():
    if not validate_url(channel_url):
        st.error("Inserisci un URL YouTube valido!")
        return
    
    with st.spinner(f"Ricerca video migliori..."):
        try:
            best_list = get_best_videos_cached(channel_url, depth_map[scan_depth], top_n)
        except Exception as e:
            st.error(f"Errore nel recupero video: {str(e)}")
            return
    
    if not best_list:
        st.warning("Nessun video trovato.")
        return

    st.success(f"Trovati {len(best_list)} video di successo.")
    
    all_comments_bundle = ""
    cols = st.columns(len(best_list))
    
    for i, vid in enumerate(best_list):
        with cols[i]:
            st.image(vid['thumbnail'], use_container_width=True)
            st.metric("Views", f"{vid['views']:,}")
            
            details = get_video_details_cached(vid['id'])
            st.markdown(f"🗓️ **{details['date']}**")
            
            st.markdown(f"**[{vid['title'][:40]}...]({vid['url']})**")
            
            with st.spinner("⏳ Commenti..."):
                try:
                    text, count = get_all_comments_cached(vid['url'])
                    all_comments_bundle += (
                        f"\n--- VIDEO: {vid['title']} "
                        f"(Visualizzazioni: {vid['views']}) ---\n"
                        f"{text}\n"
                    )
                    st.success(f"{count} estratti")
                except Exception as e:
                    st.error(f"Errore recupero commenti: {str(e)}")

    # Generazione report AI
    st.divider()
    
    with st.spinner(f"Sto elaborando con {MODEL_NAME}..."):
        prompt_str = f"""
Analizza questi dati del canale YouTube:

DATASET VIDEO E COMMENTI:
{all_comments_bundle[:10000]}  # Limita lunghezza per sicurezza

ISTRUZIONI PER IL REPORT:
1. Identifica il sentimento prevalente degli utenti.
2. Analizzando il rapporto tra i temi trattati e il successo (views), spiega perché questi video hanno funzionato meglio di altri.
3. Estrai i problemi o i dubbi più frequenti dei follower.
4. Suggerisci 3 titoli per video futuri che potrebbero replicare queste performance.

Analisi rigorosa e professionale in lingua Italiana.
"""
        
        try:
            report = query_ollama(prompt_str, ollama_url)
            st.subheader("🤖 Strategic Intelligence Report")
            st.markdown(report)
            st.download_button("Scarica Report", report, file_name="report_zakkos.md")
            st.session_state.report_generated = True
        except Exception as e:
            st.error(f"Errore nella generazione del report: {str(e)}")


if st.button("🚀 AVVIA ANALISI"):
    run_analysis()

# Reset button per pulire cache
st.sidebar.markdown("<br>", unsafe_allow_html=True)
if st.sidebar.button("🗑️ Pulisci Cache"):
    get_best_videos_cached.clear()
    get_video_details_cached.clear()
    get_all_comments_cached.clear()
    st.sidebar.success("Cache pulita!")
