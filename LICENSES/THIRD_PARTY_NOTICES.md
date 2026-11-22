# AI Local Image Studio — Third-party notices

AI Local Image Studio does **not** bundle any AI model or third-party library in its download.
Every model and package listed here is fetched **only after you choose it** in the program, from its
official source, into the data folder you picked. A copy of the license text is placed next to each
downloaded model (`LICENSE_<name>.txt`), and all texts are kept in this `LICENSES` folder.

You accepted the terms below (including the CreativeML OpenRAIL-M / OpenRAIL++-M use restrictions)
in the dialog shown at first start. The same text can be read again under *Licenses* in the program.

## 1. AI models

| Component | Used for | License | Source | License text |
|---|---|---|---|---|
| Stable Diffusion 1.5 | image generation, LoRA training | CreativeML OpenRAIL-M | huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5 | `CreativeML-OpenRAIL-M.txt` |
| Stable Diffusion XL 1.0 (base) | image generation, LoRA training | CreativeML OpenRAIL++-M | huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 | `CreativeML-OpenRAIL++-M.txt` |
| Z-Image-Turbo / Z-Image (base) | image generation / LoRA training | Apache-2.0 | huggingface.co/Tongyi-MAI/Z-Image-Turbo, huggingface.co/Tongyi-MAI/Z-Image | `Apache-2.0.txt` |
| FLUX.2 [klein] 4B / [klein] base 4B | image generation / LoRA training | Apache-2.0 | huggingface.co/black-forest-labs/FLUX.2-klein-4B, huggingface.co/black-forest-labs/FLUX.2-klein-base-4B | `Apache-2.0.txt` |
| Qwen3 1.7B / 4B / 8B, Qwen2.5 7B (through Ollama) | prompt writing | Apache-2.0 | ollama.com/library/qwen3, ollama.com/library/qwen2.5 | `Apache-2.0.txt` |
| Florence-2 base (Microsoft; florence-community/Florence-2-base-ft) | image descriptions | MIT | huggingface.co/florence-community/Florence-2-base-ft | `MIT_Florence-2.txt` |
| Real-ESRGAN x2plus | 2× upscale | BSD-3-Clause | github.com/xinntao/Real-ESRGAN | `BSD-3-Clause_Real-ESRGAN.txt` |
| YuNet face detector | face / hand fixer, data set tools | MIT | github.com/opencv/opencv_zoo | `MIT_YuNet.txt` |
| MediaPipe palm detector (Google, via opencv_zoo) | hand fixer | Apache-2.0 | github.com/opencv/opencv_zoo | `Apache-2.0.txt` |
| lbpcascade_animeface (nagadomi) | anime face detection | MIT (header inside the XML file) | github.com/nagadomi/lbpcascade_animeface | `MIT_lbpcascade_animeface.txt` |

Models you add yourself (files in the add-on folder, LoRAs from other sites) keep their own licenses;
the program does not change or check them. Stable Diffusion 1.5 / SDXL derivatives (for example
community checkpoints and LoRAs) carry the same OpenRAIL use restrictions.

## 2. Software installed into the program's own environment (with your approval)

| Component | License |
|---|---|
| Python, CPython 3.12 (python-build-standalone) | PSF-2.0 (the standalone build includes further components under their own licenses) |
| uv | MIT / Apache-2.0 |
| Tcl/Tk (tkinter) | Tcl/Tk license (BSD-style) |
| PyTorch | BSD-3-Clause (`BSD-3-Clause_PyTorch.txt`) |
| NVIDIA CUDA / cuDNN runtime libraries (pulled in by the CUDA builds of PyTorch) | NVIDIA's own license (redistributable runtime) |
| diffusers, transformers, accelerate, safetensors, peft, huggingface_hub | Apache-2.0 |
| sentencepiece | Apache-2.0 |
| protobuf | BSD-3-Clause |
| Pillow | MIT-CMU (HPND) |
| NumPy | BSD-3-Clause |
| psutil | BSD-3-Clause |
| requests | Apache-2.0 |
| OpenCV (opencv-python-headless) | Apache-2.0; the bundled FFmpeg is LGPL-2.1 (dynamically linked, unmodified) |
| Ollama | MIT (`MIT_Ollama.txt`) |

OpenCV's FFmpeg is only used to read video files (frames for data sets). No GPL encoder (such as x264)
is bundled or downloaded.

The Real-ESRGAN network definition (RRDBNet) is re-implemented in the program's own code so the published
weights can be loaded without extra packages; the architecture comes from BasicSR (Apache-2.0) and
Real-ESRGAN (BSD-3-Clause) by Xintao Wang et al.

## 3. Use restrictions (summary)

CreativeML OpenRAIL-M and OpenRAIL++-M (paragraph 5 and Attachment A) forbid using the models, their
derivatives or their outputs for: breaking any law; exploiting or harming minors; spreading verifiably
false information to harm others; generating or spreading personal identifiable information to harm a
person; defaming, disparaging or harassing others; fully automated decisions that harm a person's legal
rights; discrimination or harm based on social behaviour or personal characteristics; exploiting the
vulnerabilities of a group; discrimination based on legally protected characteristics; providing
medical advice or interpreting medical results; and generating information for law enforcement,
immigration or asylum decisions (such as predicting that a person will commit fraud or a crime).
The full, binding wording is in `CreativeML-OpenRAIL-M.txt` and `CreativeML-OpenRAIL++-M.txt`.
You are responsible for what you create and how you use it.
