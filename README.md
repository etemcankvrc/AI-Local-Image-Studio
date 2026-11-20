# 🎨 AI Local Image Studio (Open Source)

**100% local AI image generator with built-in LoRA training.** Generate images with Stable Diffusion 1.5 / SDXL,
Z-Image-Turbo and FLUX.2 [klein] on your own PC — no cloud, no API keys. A local LLM (Ollama) turns what you type in
your own language into English prompts. The interface is available in **English, Türkçe, 한국어 and Deutsch**.

> 🇹🇷 Ayrıntılı Türkçe kılavuz ve sürüm notları: **[BENIOKU.md](BENIOKU.md)**

## ✨ Features

- **Image generation:** SD 1.5, SDXL 1.0, Z-Image-Turbo, FLUX.2 [klein] 4B — quality presets, samplers, hires fix,
  aspect ratios, seed control, 2× upscaling (Real-ESRGAN).
- **Runs on modest hardware:** memory guard with GPU / low-VRAM / CPU profiles; when VRAM runs out it slows down instead
  of crashing (designed to run on a 4 GB GPU with 32 GB RAM).
- **Face & hand fixer:** finds faces and hands (YuNet, MediaPipe palm, anime faces) and redraws them sharper.
- **Prompt writer:** a local Ollama model (Qwen3 etc.) writes the English prompt and settings, then unloads itself so
  the image model gets the VRAM. Ollama on another PC in your LAN can be used too.
- **Training window:**
  - Build datasets from folders, files and **videos** (sharp, varied frames are picked automatically) or, if you allow
    it, from openly licensed images (Openverse). Files are named after the dataset (`istanbul_street_0001.jpg`).
  - Images without a description are **captioned automatically** (Florence-2).
  - **Collage / split-screen check:** images that contain several pictures (collages, split-screen video frames)
    are detected and you are asked whether to leave them out — otherwise the model learns to draw split images.
    Removed files are moved to the dataset's `excluded` folder, never deleted.
  - Choose **what to teach**: 🎨 style, 🏙️ place / city, 🧸 subject / object / character, 🧍 human anatomy / pose,
    👤 face. Each type trains differently. In *place* mode faces are blurred and not learned, so the surroundings
    come first; in *face* mode only the face is learned.
  - **Light** mode saves a word card; **deep** mode trains a LoRA on the model you choose: **SD 1.5** (4 GB VRAM or
    CPU), **SDXL** (≥ 10 GB), **FLUX.2 [klein] 4B** (≥ 12 GB; trained on the Apache-2.0 *klein base 4B*, used with
    the fast model) or **Z-Image** (≥ 16 GB; trained on the Apache-2.0 *Z-Image base*, used with Z-Image-Turbo).
    If the card is too small for full speed (e.g. FLUX.2 on 4 GB), training runs **RAM-assisted** (block swapping:
    weights stay in RAM and are moved to the GPU piece by piece) — slower, but it works with ≥ 3.5 GB VRAM and enough
    RAM (SDXL 16 GB, FLUX.2 24 GB, Z-Image 28 GB). Otherwise the choice is marked and never started;
    out-of-memory retries at a lower resolution.
    **Self-learning** learns from your 👍 / 👎 ratings and an automatic judge (Diffusion-DPO, SD 1.5).
- **Your own models:** drop a single-file SD 1.5 or SDXL model (`.safetensors`, e.g. from Civitai or Hugging Face)
  into `addons\models\` and it appears in the model list as 📦. The program never downloads them; `.ckpt` / `.pt`
  files are never opened (they can contain code). The first time, it needs either the matching built-in model or a
  ~4 MB settings pack (no weights, asked first). Their licenses are your responsibility.
- **LoRA picker:** LoRAs you drop into `addons\` plus the ones the program trains, with search, weights and
  compatibility checks. Deleted LoRAs go to a trash folder, existing LoRAs are never overwritten.

## 📦 Requirements

- **Windows 10/11 (64-bit).** An NVIDIA GPU is recommended; it also works on CPU (slowly).
- **Any Python 3.10+** just to start the program (python.org or Microsoft Store).
- On first start the program asks for a **data folder** and sets up its own environment there. It can also install
  its **own Python 3.12** there (recommended, ~45 MB via `uv`), so the Python installed on your computer doesn't
  matter. Nothing is installed into your system Python, and nothing large is downloaded without your confirmation.

## 🚀 Run

```bash
python ai_local_image_studio.py
```

1. Choose the data folder (the drive with the most free space is suggested).
2. When you generate your first image, the program lists what it needs to download (image engine, model, optional
   LLM) with sizes and destination folder, and asks once.
3. Open **🧠 TRAINING / IMAGE ANALYSIS** to build datasets and train your own LoRAs.

Everything the program downloads goes into the data folder:

```
<data folder>\
  runtime\    program environment (PyTorch, diffusers…) and, if chosen, its own Python
  models\     image models, upscaler, face/hand detectors, Florence-2
  datasets\   your datasets (images, video frames, captions)
  addons\     LoRAs you add yourself (the program never downloads add-ons)
  addons\models\  full models you add yourself (single-file SD 1.5 / SDXL .safetensors)
  loras\      LoRAs and word cards the program trained (deleted ones: loras\.trash)
  outputs\    generated images
  logs\       ails.log, setup.log
```

## 📜 Licenses

The source code in this folder is licensed under the **[Apache License 2.0](../LICENSE)**.

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

⚠️ The **use restrictions** of the OpenRAIL licenses (Attachment A) also apply to everyone who generates images with
those models.

## 🤝 Responsible use

The *face* and *subject* training modes can learn what a person looks like. Use them only for yourself, people who
agreed to it, or fictional characters — never to imitate real people without their consent. Only add images you have
the right to use; add-on LoRAs you drop into `addons\` are your responsibility.
