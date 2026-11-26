# 🎨 AI Local Image Studio (Open Source)

**100% local AI image generator with built-in LoRA training.** Generate images with Stable Diffusion 1.5, SDXL, Z-Image-Turbo, and FLUX.2 [klein] on your own PC — no cloud, no API keys. A local LLM (Ollama) translates prompts from your own language into English. The interface is available in **English, Türkçe, 한국어, and Deutsch**.

> 🇹🇷 Ayrıntılı Türkçe kılavuz ve sürüm notları: **[BENIOKU.md](BENIOKU.md)**

<img width="2560" height="1392" alt="image" src="https://github.com/user-attachments/assets/3ee340d2-f65d-4758-85c0-7675acd7e250" />


---

## ✨ Features

### 🖼️ Image Generation & Low-VRAM Support
- **Multiple engines:** SD 1.5, SDXL 1.0, Z-Image-Turbo, and FLUX.2 [klein] 4B — with quality presets, samplers, hires fix, aspect ratios, seed control, and 2× upscaling (Real-ESRGAN).
- **Runs on modest hardware:** Built-in memory guard with GPU, low-VRAM, and CPU profiles. When VRAM runs out, it offloads and slows down instead of crashing (tested on a **4 GB VRAM laptop GPU + 32 GB RAM**).
- **Face & hand fixer:** Detects faces and hands (YuNet, MediaPipe palm, anime faces) and redraws them sharper.
- **Local prompt writer (Ollama):** Uses a local Ollama model (Qwen3, etc. — locally or over LAN) to write English prompts and settings, then unloads itself so the image model gets the VRAM. Translates faithfully without adding details you didn't ask for, validates nationality/details, and trims overly long prompts.

<img width="1310" height="1161" alt="image" src="https://github.com/user-attachments/assets/58ffe616-83bd-46d1-814f-73f236f5c3f3" />

### 🧠 Built-in LoRA Training & Dataset Tools
- **Dataset creation:** Build datasets from folders, images, and **videos** (automatically picks sharp, varied frames), or optionally from openly licensed images (Openverse). Files are cleanly renamed after the dataset (`istanbul_street_0001.jpg`).
- **Auto-captioning:** Images without descriptions are captioned automatically via **Florence-2**.
- **Collage / split-screen splitter:** Detects multi-panel collages or split-screen video frames, previews cut lines, and splits them into separate training images in one click (trims separator bands, handles nested layouts). Originals are moved to the dataset's `excluded` folder, never deleted.
- **Dedicated teach modes:** Choose what to train — 🎨 **style**, 🏙️ **place / city** (blurs faces so surroundings come first), 🧸 **subject / object / character**, 🧍 **human anatomy / pose**, or 👤 **face** (learns only the face).
- **Light & Deep training modes:**
  - **Light mode** saves a word card.
  - **Deep mode** trains a LoRA on **SD 1.5** (4 GB VRAM or CPU), **SDXL** (≥ 10 GB), **FLUX.2 [klein] 4B** (≥ 12 GB; trained on Apache-2.0 *klein base 4B*), or **Z-Image** (≥ 16 GB; trained on Apache-2.0 *Z-Image base*).
  - **RAM-assisted training (block swapping):** If your GPU is smaller than the full-speed requirement (e.g., FLUX.2 on 4 GB VRAM), weights stay in system RAM and swap to the GPU piece by piece. Slower, but works with **≥ 3.5 GB VRAM** and enough RAM (**SDXL:** 16 GB, **FLUX.2:** 24 GB, **Z-Image:** 28 GB). Out-of-memory errors automatically retry at a lower resolution.
  - **Large datasets & self-learning:** Supports thousands of images (caches prepared data to disk when RAM is full and cleans up afterward) and includes **Diffusion-DPO** self-learning (SD 1.5) from 👍 / 👎 ratings.

<img width="875" height="620" alt="image" src="https://github.com/user-attachments/assets/404d34dd-d07d-48af-a742-abc56c9fea1b" />

### 📦 Custom Models & LoRA Management
- **Bring your own models:** Drop single-file SD 1.5 or SDXL `.safetensors` models (e.g., from Civitai or Hugging Face) into `addons\models\` to see them in the model list (📦). The app never downloads third-party checkpoints and refuses `.ckpt` / `.pt` files for safety. First-time use needs either the matching built-in model or a ~4 MB config pack (no weights, asked first). Model licenses are your responsibility.
- **LoRA picker:** Lists LoRAs from `addons\` alongside your trained LoRAs with search, weight sliders, and compatibility checks. Deleted LoRAs move safely to `loras\.trash` without overwriting existing files.

<img width="641" height="474" alt="image" src="https://github.com/user-attachments/assets/40dcac7c-1a13-4589-b048-68dab5124f40" />

### 🌐 Local Network Image Server (LAN API)
Use one PC's GPU and models from other devices on your network (like Ollama, but for images). Enable via **Section 4 → 🌐 Network server…** (off by default):
- **Access modes:** *This computer only* (`127.0.0.1`, no key, browser requests blocked) or *Home / office network* (LAN only, access key required — also suitable for Docker apps).
- **Multi-PC Studio sync:** Another copy of AI Local Image Studio can connect via **Image engine → ➕ Add a server on the network…** (`192.168.x.x:7860` + key). The client needs no local image weights; prompts are still prepared locally, and `.safetensors` LoRAs are uploaded once to `addons\network\` (can be toggled off).
- **Compatible APIs:** Supports **Automatic1111** (`/sdapi/v1/txt2img`, `<lora:name:0.8>`) for Open WebUI, SillyTavern, and Krita plugins, plus **OpenAI Images** (`/v1/images/generations`). The access key works as a Bearer token or HTTP password.
- **Safe queueing:** Requests run one at a time through the memory guard and wait in line while training is active. Outputs are also saved in `outputs\network\`.
- **Firewall note:** Allow **private networks only** when Windows prompts. Never port-forward this server to the public internet.

---

## 🆕 Recent Updates

- **Fixed LoRA deletion:** You can now delete user-added LoRAs as well as trained ones. Loaded files are released before moving to `loras\.trash` so active LoRAs no longer block deletion.
- **Live training preview:** The training panel now displays the latest sample image with a **"next sample in ~X"** countdown.
- **Character-LoRA captions:** Trigger words are no longer duplicated when a caption already starts with the trigger word.
- **Stability improvements:** Auxiliary windows (training, network, etc.) open reliably, and unexpected errors display a dialog and log to `logs\gui_errors.log` instead of crashing silently.

---

## 📦 Requirements

- **OS:** Windows 10 / 11 (64-bit).
- **Hardware:** An NVIDIA GPU is recommended (4 GB+ VRAM with 24–32 GB system RAM for FLUX.2); CPU mode is also supported (slower).
- **Python (only if running from source):** Any Python 3.10+ to launch. The standalone Windows Setup build does **not** require Python.
- **Self-contained environment:** On first start, the app asks for a **data folder** and creates its own isolated environment there (optionally installing its own ~45 MB Python 3.12 via `uv`). Nothing is installed into your system Python, and no large files are downloaded without confirmation.

---

## 🚀 Install & Run

### Option 1: Windows Standalone (No Python needed)
Download the latest **FREE** release from the **[Releases](https://github.com/etemcankvrc/AI-Local-Image-Studio/releases)** page:
- **Setup (`AI_Local_Image_Studio_…_Setup.exe`):** Run once to install, then launch **`AI_Local_Image_Studio.exe`**.
- **Portable ZIP (if provided):** Extract anywhere and run **`AI_Local_Image_Studio.exe`** inside the folder (ignore the supporting files next to it).

> **Windows SmartScreen note:** Because the Windows installer is unsigned, SmartScreen may show a warning on first run. If you prefer not to run prebuilt executables, you can inspect the code and run directly from source below.

### Option 2: Run from Source (Python 3.10+)
```bash
python ai_local_image_studio.py
```

### First-Time Setup
1. Choose a **data folder** (the drive with the most free space is suggested).
2. On your first generation, the app lists required downloads (engine, model, optional LLM) with sizes and asks for confirmation once.
3. Open **🧠 TRAINING / IMAGE ANALYSIS** to build datasets and train your own LoRAs.

All downloaded files stay inside your chosen data folder:
```text
<data folder>\
  runtime\        isolated environment (PyTorch, diffusers…) and optional Python
  models\         image models, upscaler, face/hand detectors, Florence-2
  datasets\       your datasets (images, video frames, captions)
  addons\         user-added LoRAs (never downloaded automatically)
  addons\models\  user-added full models (single-file SD 1.5 / SDXL .safetensors)
  loras\          trained LoRAs & word cards (deleted items: loras\.trash)
  outputs\        generated images
  logs\           ails.log, setup.log, gui_errors.log
```

---

## 💛 Support & Early-Access Builds (Optional)

Everything in this repository — **image generation, prompt writer, RAM-assisted LoRA training (all teach modes), LoRA tools, and the LAN API server** — is free and open source.

If you want to support development, supporters on Patreon get **early access** to separate experimental modules currently in active development (work in progress):
- 🎬 **Image → Video** & **video LoRA training**
- ✏️ Standalone **image editor / retouch** tool
- 🔍 Standalone **batch upscaler**
- 🧰 **Dataset Studio** (automated homogeneous character-LoRA dataset builder)

👉 **[patreon.com/GTEtemcan](https://www.patreon.com/GTEtemcan)**

---

## 📜 Licenses

The source code in this repository is licensed under the **[Apache License 2.0](./LICENSE)**.

Models and third-party components are downloaded separately and keep their own licenses:

| Component | License |
|---|---|
| Stable Diffusion 1.5 | CreativeML OpenRAIL-M |
| Stable Diffusion XL 1.0 | CreativeML OpenRAIL++-M |
| Z-Image-Turbo | Apache-2.0 |
| FLUX.2 [klein] 4B | Apache-2.0 |
| FLUX.2 [klein] base 4B (training only) | Apache-2.0 |
| Z-Image base (training only) | Apache-2.0 |
| Qwen3 / Qwen2.5 (via Ollama) | Apache-2.0 |
| Real-ESRGAN x2plus | BSD-3-Clause |
| OpenCV (incl. its LGPL FFmpeg build) | Apache-2.0 (+ LGPL-2.1) |
| YuNet face detector | MIT |
| MediaPipe palm detector | Apache-2.0 |
| lbpcascade_animeface | MIT |
| Florence-2 base | MIT |
| PEFT | Apache-2.0 |
| uv / CPython 3.12 (optional) | MIT or Apache-2.0 / PSF-2.0 |

⚠️ The **use restrictions** of the OpenRAIL licenses (Attachment A) apply to everyone who generates images with those models.

On first launch, the app displays model licenses and use restrictions for acceptance. Full license texts live in `LICENSES/` (`LICENSES/THIRD_PARTY_NOTICES.md`), and a copy is placed next to each downloaded model (`LICENSE_<name>.txt`).

---

## 🤝 Responsible Use

The *face* and *subject* training modes can learn a person's likeness. Use them only for yourself, people who have given consent, or fictional characters — never to imitate real people without permission. Only use training images you have the right to use; third-party LoRAs placed in `addons\` are your responsibility.
