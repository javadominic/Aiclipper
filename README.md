# Autoclipper 🎬⚡
### Open-Source AI Video Clipper & Automated Shorts Maker
**Turn YouTube, Twitch & Kick live streams and long videos into viral, captioned vertical clips in minutes.**

[![License: MIT](https://img.shields.io/badge/License-MIT-emerald.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FFmpeg Full](https://img.shields.io/badge/FFmpeg-libass%20enabled-green.svg)](https://ffmpeg.org/)
[![Self-Hosted](https://img.shields.io/badge/Privacy-100%25%20Local%20%26%20Self--Hosted-darkgreen.svg)](#privacy--security)
[![Repo](https://img.shields.io/badge/GitHub-javadominic%2FAiclipper-brightgreen.svg)](https://github.com/javadominic/Aiclipper)

---

## 📌 Overview: What is Autoclipper?

**Autoclipper** is a 100% free, privacy-first, open-source alternative to paid clipping software like OpusClip, Klap, and Submagic. It runs entirely on your own computer or server using your own AI keys (BYOK).

Simply paste any **YouTube**, **Twitch**, or **Kick** video link (or upload an MP4/MOV file). Autoclipper:
1. 🎙️ **Transcribes speech on-device** (audio never leaves your machine).
2. 🧠 **Scouts viral hooks & punchy moments** using an LLM of your choice (Google Gemini, OpenAI, Claude, Groq, or local Ollama).
3. ✂️ **Reframes footage to 9:16 vertical video** and burns high-retention, animated word-by-word karaoke captions.
4. 🚀 **Auto-schedules & posts directly** to **YouTube Shorts**, **TikTok**, and **Instagram Reels**.

---

## ⚡ Key Features

- **Live Stream Highlight Clipper:** Connect a live stream from Twitch, Kick, or YouTube. Autoclipper records into a rolling buffer and extracts viral clips *while the creator is still streaming*.
- **Multi-Creator Workspaces:** Manage multiple streamers and channels with independent settings, custom caption presets, and social post queues.
- **On-Device Speech-to-Text:** Uses `sherpa-onnx` and FastConformer/Parakeet models running locally. Fast, private, and zero third-party transcription fees.
- **Customizable AI Brain:** Plug in free Google Gemini Flash, OpenAI GPT-4o-mini, Groq Llama-3, Claude 3.5 Sonnet, or run 100% offline with Ollama.
- **Zero-Downtime AI Auto-Fallback:** Set a backup provider (e.g. Groq) alongside your primary AI (e.g. Gemini). When rate limits or quota caps hit, Autoclipper switches instantly without interrupting your clips.
- **Dynamic Animated Subtitles:** Word-level highlighted captions in popular styles (Bilingual, MrBeast/Hormozi style, clean minimal) powered by `ffmpeg-full` and `libass`.
- **Direct Multi-Platform Publishing:** Built-in OAuth integration to upload directly to YouTube Shorts, Instagram Reels, and TikTok.
- **Zero Lock-In & Self-Hosted:** No subscription fees, no watermarks, no limits on video duration or clip quantities.

---

## 📊 Autoclipper vs. Paid Alternatives

| Feature | Autoclipper | OpusClip / Klap | Submagic |
|---|:---:|:---:|:---:|
| **Price** | **100% Free & Open Source** | $19–$99 / month | $20–$100 / month |
| **Watermark** | ❌ None | ✔️ Free tier has watermarks | ✔️ Watermarked |
| **Video Length Limits** | ❌ None (process hours of footage) | ⚠️ Strict monthly credit caps | ⚠️ Credit caps |
| **Live Stream Clipping** | ✔️ Real-time live clipping | ❌ VOD only | ❌ VOD only |
| **Privacy** | 🔒 Audio transcribed locally | ☁️ Uploaded to cloud | ☁️ Uploaded to cloud |
| **Supported AI Models** | 🤖 Gemini, OpenAI, Claude, Ollama | 🔒 Proprietary / Fixed | 🔒 Proprietary / Fixed |

---

## 🚀 Quick Start (What, Where & How)

### 1. Prerequisites (What you need)
- **Python 3.10+**
- **FFmpeg with `libass` support** (required for burning animated karaoke captions):
  - **macOS (Homebrew):**
    ```bash
    brew install ffmpeg-full && brew link --overwrite ffmpeg-full
    ```
  - **Windows (Winget):**
    ```bash
    winget install Gyan.FFmpeg
    ```
  - **Linux (Ubuntu/Debian):**
    ```bash
    sudo apt update && sudo apt install -y ffmpeg libass-dev
    ```

---

### 2. Installation (How to set it up)

```bash
# Clone the repository
git clone https://github.com/javadominic/Aiclipper.git
cd Aiclipper

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# (Optional) Pre-download the on-device speech model (~480 MB)
python -m studio.fetch_model
```

---

### 3. Launch Autoclipper (Where to use it)

Start the local studio server:
```bash
python -m studio.server
```

Open your browser at **[http://127.0.0.1:8765](http://127.0.0.1:8765)**.

> **Note:** Sign-in is disabled by default for single-user local setups. The dashboard opens immediately.

---

## 🤖 Choose Your AI Provider (BYOK)

Autoclipper works with any JSON-capable LLM to scout the best moments and generate viral hooks and titles.

### Recommended: Google Gemini (Free & High Speed)
1. Get a free API key at [Google AI Studio](https://aistudio.google.com/apikey).
2. Open Autoclipper at `http://127.0.0.1:8765` → Click **AI Engine** in the sidebar.
3. Select **Google Gemini**, set the model to `gemini-3-flash-preview` (or `gemini-2.5-flash`), paste your API key, and click **Save and test**.

Alternatively, configure your keys directly in your `.env` file:
```ini
AUTOCLIPPER_AI_PROVIDER=gemini
AUTOCLIPPER_AI_MODELS=gemini-3-flash-preview
AUTOCLIPPER_AI_KEY=your_gemini_api_key_here
```

### Supported Providers:
| Provider | Setting | Key URL |
|---|---|---|
| **Google Gemini** | `AUTOCLIPPER_AI_PROVIDER=gemini` | [Google AI Studio](https://aistudio.google.com/apikey) |
| **OpenAI** | `AUTOCLIPPER_AI_PROVIDER=openai` | [OpenAI Platform](https://platform.openai.com/api-keys) |
| **Groq (Ultra-Fast)** | `AUTOCLIPPER_AI_PROVIDER=groq` | [Groq Console](https://console.groq.com/keys) |
| **Anthropic Claude** | `AUTOCLIPPER_AI_PROVIDER=anthropic` | [Anthropic Console](https://console.anthropic.com/) |
| **OpenRouter** | `AUTOCLIPPER_AI_PROVIDER=openrouter` | [OpenRouter Keys](https://openrouter.ai/keys) |
| **Ollama (100% Offline)**| `AUTOCLIPPER_AI_PROVIDER=ollama` | [Ollama](https://ollama.com/) |

### 🔄 Zero-Downtime Automatic AI Fallback

Never let free-tier rate limits or quota exhaustion stop your video clipping pipeline. Autoclipper supports **dual-provider auto-failover**:

1. In **Settings → AI**, configure your **Primary AI** (e.g., Google Gemini Flash).
2. Right below it, configure your **Backup AI** (e.g., Groq `llama-3.3-70b-versatile`).
3. If your primary provider hits an HTTP 429, rate limit, or credit exhaustion, Autoclipper **switches instantly to the backup model** without dropping clips or falling back to loudness only.
4. An informative notice will display in the studio topbar confirming that backup AI seamlessly took over.

You can also define your fallback directly in your `.env` file:
```ini
# Primary AI
AUTOCLIPPER_AI_PROVIDER=gemini
AUTOCLIPPER_AI_MODELS=gemini-3-flash-preview,gemini-2.5-flash
AUTOCLIPPER_AI_KEY=your_gemini_key

# Backup AI (Automatic Fallback)
AUTOCLIPPER_AI_FALLBACK_PROVIDER=groq
AUTOCLIPPER_AI_FALLBACK_MODELS=llama-3.3-70b-versatile
AUTOCLIPPER_AI_FALLBACK_KEY=your_groq_key
```

---

## 🎬 How to Clip Videos & Streams

### Method 1: Clip an Existing Video or VOD
1. Paste a link from YouTube, Twitch, or Kick (or click **Browse** to upload an MP4/MOV).
2. Select your target clip count and caption style.
3. Click **Make clips**. Autoclipper extracts the audio, transcribes speech, asks the AI for viral moments, and renders 9:16 vertical videos.

### Method 2: Clip a Live Stream in Real-Time
1. Add a creator under **Creators → New creator** and link their Twitch, Kick, or YouTube channel.
2. Click **Record live stream**.
3. Autoclipper monitors the broadcast in a rolling buffer. When funny or high-energy moments occur, highlights are generated immediately.

---

## 📱 Connecting YouTube Shorts & Auto-Posting

To connect YouTube Shorts:
1. Create a project in [Google Cloud Console](https://console.cloud.google.com/) and enable the **YouTube Data API v3**.
2. Create an **OAuth 2.0 Client ID** of type **Web application**:
   - **Authorized JavaScript origins:** `http://127.0.0.1:8765`
   - **Authorized redirect URIs:** `http://127.0.0.1:8765/oauth/youtube/callback`
3. Add your Gmail under **OAuth consent screen → Test users**.
4. In Autoclipper, navigate to **Settings → Connected Apps → YouTube → Set up**, paste your Client ID & Secret, and click **Connect**.

---

## 🛠️ Common Troubleshooting

- **`No option name near 'clip01.ass:fontsdir=f'`:** Your system FFmpeg is missing `libass`. On macOS, run `brew install ffmpeg-full && brew link --overwrite ffmpeg-full`.
- **`Error 400: redirect_uri_mismatch`:** Verify in Google Cloud Console that `http://127.0.0.1:8765/oauth/youtube/callback` is listed under **Authorized redirect URIs** (NOT Authorized JavaScript origins).
- **`Error 403: access_denied` during YouTube OAuth:** Your Google Cloud OAuth app is in testing mode. Add your email address to the **Test users** list in the Google Cloud Console.

---

## 🔒 Privacy & Security

- All video downloads, cached chunks, and transcripts stay on your local disk in `./data`.
- Audio files are **never** uploaded to cloud transcription APIs.
- API keys are stored locally and are never transmitted to external telemetry servers.

---

## 🎖️ Credits & Acknowledgements

- **Autoclipper** is a modified and enhanced fork of the open-source **[PeakCut](https://github.com/haziqahmedqasmi/PeakCut)** project by Haziq Ahmed Qasmi.
- Distributed under the terms of the [MIT License](LICENSE). Sincere gratitude to the original creator and open-source contributors.
- Enhancements in this edition include redesigned Electric Emerald UI/UX, dual-provider automatic AI rate-limit fallback (e.g. Gemini → Groq), streamlined sidebar settings navigation, built-in interactive OAuth/FFmpeg troubleshooting guide, in-progress clip cancellation, and refreshed vector branding assets.

---

## 📄 License & Attribution

- Released under the [MIT License](LICENSE).
- Subtitle display typography (Anton, Archivo Black, Poppins) licensed under SIL Open Font License.
- Repository: [https://github.com/javadominic/Aiclipper](https://github.com/javadominic/Aiclipper)
