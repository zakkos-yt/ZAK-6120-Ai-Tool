# Zak-6120 AI Workstation

In questo repository trovate alcuni dei file che uso quotidianamente con
la mia workstation per AI locale: workflow per ComfyUI, script Python e
due piccole pagine web interattive generate con AI.

## La mia configurazione

-   **Scheda madre:** ASUS X99 Deluxe
-   **CPU:** Intel Xeon E5-1650 v4
-   **RAM:** 64 GB DDR4 ECC
-   **GPU:** 2 × NVIDIA GeForce RTX 3060 12 GB
-   **Storage:** NVMe 1 TB
-   **Sistema operativo:** Linux Mint Debian Edition 7 (LMDE 7)

I workflow e gli script descritti qui sono quelli che utilizzo realmente
sulla mia macchina. Potrebbero richiedere modifiche per funzionare su
configurazioni hardware, sistemi operativi o installazioni differenti.

## Contenuto del repository

Sono presenti:

-   3 workflow JSON per **ComfyUI** configurati per l'utilizzo di due
    GPU;
-   2 script Python realizzati con l'ausilio dell'AI e successivamente
    provati e adattati al mio utilizzo;
-   una pagina web locale per creare una **Tier List di sistemi
    operativi**;
-   una pagina web locale, realizzata come esempio, per creare una
    **Tier List di animali casuali**.

La Tier List degli animali è quella mostrata nel video YouTube del mio
canale dedicato a questi strumenti.

Potete utilizzare e modificare liberamente questi file per adattarli
alle vostre esigenze. Se siete content creator e li mostrate nei vostri
contenuti, è gradito un cenno di riconoscimento.

------------------------------------------------------------------------

# Workflow per ComfyUI

> **Avviso**
>
> I workflow sono stati testati esclusivamente con ComfyUI installato
> **bare metal** su LMDE 7, all'interno di un ambiente virtuale Python.
> Non ho verificato il funzionamento con ComfyUI Portable, Windows,
> Docker o altre configurazioni.

## Requisito MultiGPU: ComfyUI-MultiGPU / DisTorch2

I tre workflow utilizzano i nodi MultiGPU di **ComfyUI-MultiGPU**, che
permettono di distribuire il modello tra più GPU tramite DisTorch2.

L'installazione tramite **ComfyUI-Manager** è il metodo consigliato:
cercate `ComfyUI-MultiGPU` tra i custom node e installatelo.

Per l'installazione manuale:

``` bash
cd ~/ComfyUI/custom_nodes
git clone https://github.com/pollockjj/ComfyUI-MultiGPU.git
```

Dopo l'installazione riavviate ComfyUI.

> Se ComfyUI si trova in un percorso differente da `~/ComfyUI`,
> modificate i percorsi indicati nei comandi di questa guida.

## FLUX.1 Dev Q4_K_M MultiGPU

**File workflow:** `flux1-dev_q4_k_m_multigpu_zak6120.json`

Questo workflow è dedicato alla generazione di immagini con **FLUX.1 Dev
Q4_K_M in formato GGUF**.

Nel workflow originale il modello veniva caricato tramite
`UnetLoaderGGUF`. L'ho sostituito con:

``` text
UnetLoaderGGUFDisTorch2MultiGPU
```

ripristinando i collegamenti `MODEL` verso il guider e lo scheduler.

Nella versione pubblicata nel repository la configurazione è:

``` text
Compute: cuda:1
Donor:   cuda:0
```

La seconda RTX 3060 viene usata come GPU di compute perché, nella
disposizione fisica della mia workstation, riceve un flusso d'aria
migliore e lavora a temperature inferiori. Questa scelta non è
obbligatoria: `cuda:0` e `cuda:1` possono essere invertite in base alla
vostra configurazione.

Il workflow utilizza:

``` text
flux1-dev-q4_k_m.gguf
clip_l.safetensors
t5xxl_fp8_e4m3fn.safetensors
ae.safetensors
```
``` bash
cd ~/ComfyUI/models
wget -O unet/flux1-dev-q4_k_m.gguf \
"https://huggingface.co/unsloth/FLUX.1-dev-GGUF/resolve/main/flux1-dev-Q4_K_M.gguf?download=true"

wget -O text_encoders/clip_l.safetensors \
"https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/clip_l.safetensors?download=true

wget -O text_encoders/t5xxl_fp8_e4m3fn.safetensors \
"https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp8_e4m3fn.safetensors?download=true"

wget -O vae/ae.safetensors \
"https://huggingface.co/flux-safetensors/flux-safetensors/resolve/main/ae.safetensors?download=true"
```

Posizione dei file:

``` text
ComfyUI/
└── models/
    ├── diffusion_models/
    │   └── flux1-dev-q4_k_m.gguf
    ├── text_encoders/
    │   ├── clip_l.safetensors
    │   └── t5xxl_fp8_e4m3fn.safetensors
    └── vae/
        └── ae.safetensors
```

> Il loader GGUF richiede inoltre che nella vostra installazione siano
> disponibili i nodi GGUF compatibili con ComfyUI.

## Wan 2.2 TI2V 5B MultiGPU

**File workflow:** `wan22_5B_ti2v_multigpu_zak6120.json`

Questo workflow utilizza **Wan 2.2 TI2V 5B FP16** e può essere usato per
la generazione video text-to-video/image-to-video prevista dal modello.

Anche in questo caso il loader originale è stato affiancato/sostituito
nella catena attiva dal nodo:

``` text
UNETLoaderDisTorch2MultiGPU
```

configurato come:

``` text
Compute: cuda:1
Donor:   cuda:0
```

Il workflow utilizza:

``` text
wan2.2_ti2v_5B_fp16.safetensors
umt5_xxl_fp8_e4m3fn_scaled.safetensors
wan2.2_vae.safetensors
```

I file vanno posizionati rispettivamente in:

``` text
ComfyUI/models/diffusion_models/
ComfyUI/models/text_encoders/
ComfyUI/models/vae/
```

Per scaricarli direttamente nelle cartelle corrette:

``` bash
cd ~/ComfyUI/models/diffusion_models
wget https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_ti2v_5B_fp16.safetensors

cd ~/ComfyUI/models/text_encoders
wget https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors

cd ~/ComfyUI/models/vae
wget https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors
```

## Wan 2.2 A14B MultiGPU

**File workflow:** `wan22_A14B_multigpu_zak6120.json`

Questo workflow utilizza **Wan 2.2 A14B** nella configurazione HIGH/LOW
NOISE.

A differenza dei workflow precedenti, qui vengono caricati **due modelli
distinti**, uno per la fase HIGH NOISE e uno per la fase LOW NOISE.
Entrambi i loader sono stati sostituiti con:

``` text
UNETLoaderDisTorch2MultiGPU
```

e configurati come:

``` text
Compute: cuda:1
Donor:   cuda:0
```

I modelli utilizzati sono:

``` text
Wan2_2-T2V-A14B_HIGH_fp8_e4m3fn_scaled_KJ.safetensors
Wan2_2-T2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors
umt5_xxl_fp8_e4m3fn_scaled.safetensors
wan_2.1_vae.safetensors
```

Nel workflow il modello HIGH NOISE viene utilizzato nella prima parte
del sampling e il LOW NOISE nella fase finale.

I due modelli HIGH e LOW A14B possono essere scaricati dal repository Hugging Face
di Kijai:

``` bash
cd ~/ComfyUI/models/diffusion_models

wget https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/T2V/Wan2_2-T2V-A14B_HIGH_fp8_e4m3fn_scaled_KJ.safetensors

wget https://huggingface.co/Kijai/WanVideo_comfy_fp8_scaled/resolve/main/T2V/Wan2_2-T2V-A14B-LOW_fp8_e4m3fn_scaled_KJ.safetensors
```

Text encoder e VAE:

``` bash
cd ~/ComfyUI/models/text_encoders
wget https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors

cd ~/ComfyUI/models/vae
wget https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors
```

> I modelli A14B sono molto grandi. Verificate di avere spazio libero
> sufficiente prima di avviare i download.

------------------------------------------------------------------------

# Script Python: Zakkos Analyze v3+

**File:** `zakkos_analyze_v3_qwen.py`

Questo script analizza i commenti dei video YouTube di un canale a
vostra scelta. È pensato principalmente per i content creator che
vogliono analizzare il proprio canale, ma può essere utilizzato con
qualsiasi canale YouTube pubblico compatibile con gli strumenti
impiegati.

L'interfaccia viene eseguita localmente tramite **Streamlit**.

## Come funziona

Il processo si divide in tre fasi:

1.  Si inserisce l'URL del canale, si sceglie se esaminare gli ultimi
    10, 50, 100 video oppure l'intero canale e si decide quanti video
    TOP analizzare.
2.  `yt-dlp` recupera i video e le relative informazioni, mentre
    `youtube-comment-downloader` estrae i commenti. Lo script limita
    l'analisi a un massimo di 5000 commenti per video.
3.  I commenti raccolti vengono inviati a **Ollama**, che genera un
    report con sentiment prevalente, possibili motivi del successo dei
    video, dubbi ricorrenti del pubblico e suggerimenti per nuovi
    contenuti.

La versione pubblicata nel repository è configurata per:

``` text
qwen36-35b-a3b-q4km:latest
```

Il nome del modello può essere modificato direttamente nello script
tramite la variabile `MODEL_NAME`.

Ollama deve essere in esecuzione e raggiungibile, per impostazione
predefinita, all'indirizzo locale `http://localhost:11434`.

## Installazione

Create e attivate un ambiente virtuale Python:

``` bash
python3 -m venv venv
source venv/bin/activate
```

Installate le dipendenze:

``` bash
pip install streamlit requests py-cpuinfo yt-dlp youtube-comment-downloader
```

Avviate lo script:

``` bash
streamlit run zakkos_analyze_v3_qwen.py
```

L'interfaccia si aprirà nel browser tramite il server locale di
Streamlit.

------------------------------------------------------------------------

# Script Python: YouTube → articolo Markdown

**File:** `articolo_da_youtube_a_md.py`

Questo script genera un articolo in formato Markdown partendo dal link
di un video YouTube. Lo utilizzo principalmente per creare una base di
partenza per gli articoli del mio blog.

Il risultato non va considerato infallibile o pronto per la
pubblicazione senza controllo: è una base che conviene sempre rileggere
e correggere prima di importarla in WordPress.

L'intera elaborazione avviene in locale.

## Pipeline

``` text
YouTube
   ↓
yt-dlp
   ↓
Faster-Whisper large-v3
   ↓
Ollama + qwen36-35b-a3b-q4km:latest
   ↓
Articolo Markdown
```

Il funzionamento si sviluppa in tre fasi principali:

1.  Si incolla il link del video nell'interfaccia web locale.
2.  `yt-dlp` scarica temporaneamente l'audio e **Faster-Whisper
    large-v3** lo trascrive. Nella configurazione predefinita Whisper
    utilizza `cuda:1`.
3.  La trascrizione viene inviata a **Ollama con `qwen36-35b-a3b-q4km:latest`**, che la
    trasforma in un articolo Markdown.

Lo script salva anche la trascrizione, una versione con timestamp e un
file JSON con le statistiche dell'elaborazione. L'audio temporaneo viene
eliminato automaticamente al termine.

L'interfaccia web non utilizza Streamlit: viene servita localmente
tramite **Flask** all'indirizzo:

``` text
http://127.0.0.1:5000
```

## Requisiti

-   Python con supporto alle dipendenze richieste;
-   Ollama in esecuzione;
-   modello `qwen36-35b-a3b-q4km:latest` disponibile in Ollama;
-   GPU NVIDIA/CUDA per la configurazione Faster-Whisper utilizzata
    dallo script.

## Installazione

Create e attivate un ambiente virtuale Python:

``` bash
python3 -m venv venv
source venv/bin/activate
```

Installate le dipendenze:

``` bash
pip install -U flask yt-dlp faster-whisper ollama \
    nvidia-cublas-cu12 "nvidia-cudnn-cu12==9.*"
```

Avviate lo script:

``` bash
python3 articolo_da_youtube_a_md.py
```

Aprite quindi nel browser:

``` text
http://127.0.0.1:5000
```

Lo script prova automaticamente ad aggiungere al loader path le librerie
cuBLAS e cuDNN installate nel virtual environment, evitando nella
configurazione prevista di dover esportare manualmente
`LD_LIBRARY_PATH`.

I parametri principali possono essere modificati tramite variabili
d'ambiente o direttamente nello script, tra cui modello Ollama, GPU
utilizzata da Whisper, context size e directory di output.

------------------------------------------------------------------------

# Pagine web Tier List

## Tier List animali

La Tier List degli animali è un piccolo esempio mostrato nel mio video.
È stata generata con **Unsloth Studio** in circa 50 secondi con la
modalità di generazione codice attivata.

Gira interamente in locale all'interno del browser e non richiede
dipendenze o server. Non ha uno scopo particolare oltre a mostrare un
esempio di generazione di codice HTML tramite un modello AI locale.

## Tier List sistemi operativi

La Tier List dei sistemi operativi nasce vedendo molti video di questo
tipo su YouTube e volendo riprodurre la stessa idea con una pagina
completamente locale.

Anche questa non richiede server o dipendenze.

È sufficiente:

1.  estrarre il contenuto del file ZIP;
2.  mantenere insieme i file e le cartelle contenuti nell'archivio;
3.  aprire `index.html` con un browser.

------------------------------------------------------------------------

# Note finali

Questi file sono stati creati per il mio utilizzo personale e
successivamente condivisi perché possano essere utili anche ad altri.

Non sono progetti commerciali e non posso garantire il funzionamento su
ogni configurazione. In particolare, i workflow MultiGPU sono stati
realizzati e testati sulla mia workstation con **due RTX 3060 da 12
GB**.

## Licenza

Questo progetto è distribuito sotto licenza GNU General Public License v3.0 (GPLv3).
Potete utilizzare, studiare, modificare e redistribuire il codice nel rispetto dei termini della licenza.
Consultate il file `LICENSE` per il testo completo.

Sentitevi liberi di studiarli e modificarli per adattarli al vostro
hardware e al vostro modo di lavorare.

## Link

Il mio canale Youtube: https://www.youtube.com/@zakkos
Se questi strumenti vi sono stati utili e volete supportare il mio lavoro:
https://www.paypal.me/zakkos
