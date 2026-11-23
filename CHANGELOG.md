# Changelog

All notable changes to AI Local Image Studio are listed here. This repository is the free, open-source core
(image generation, prompt writer, LoRA training, network server, LoRA tools). A one-click Windows **PRO** build for
supporters adds separate advanced tools — image→video + video LoRA training, a standalone editor/retouch tool, a
batch upscaler, and **Dataset Studio** — at **[patreon.com/GTEtemcan](https://www.patreon.com/GTEtemcan)**.

## 1.8.7

- **Training preview fixed.** The "next sample" countdown was resetting to when each sample *appeared*, which drifted
  from the real schedule (sample images take time to render, and step times vary), so it jumped around. It is now
  locked to the trainer's actual schedule — every *checkpoint* minutes from the start — so it counts down smoothly.
- The training preview now shows, at a glance: how often a sample is produced, how long until the next one, which
  dataset/trigger is training, and the last sample's step.

## 1.8.5 – 1.8.6

- **Fixed "could not load LoRA" (`requires_grad=True on inference tensor`).** Generating with a freshly trained LoRA
  (or the training preview) could fail because image generation ran under `inference_mode`, which taints tensors so a
  LoRA can't be injected afterwards. Generation now runs under `no_grad`, so trained LoRAs load and apply correctly.
- **Taskbar icon when run from source.** The window now sets an app identity so Windows shows the program's own icon
  in the taskbar instead of Python's (requires `simge.ico` next to the program).

## 1.8.2 – 1.8.4

- **Fixed a crash when loading a model (`OSError [Errno 22]`)** on console-less (windowed) launches: the image/text
  libraries could write their progress bars to an invalid console handle and bring the program down. Standard
  output/error are now hardened at the earliest point for every process, and model-load progress bars are disabled,
  so loading a model to generate or train no longer crashes.
- A few helper sub-processes now decode their output as UTF-8, avoiding rare errors on non-UTF-8 Windows locales.
- Internal: removed long top-of-file description blocks from the source (kept short section comments); the version
  lives only in `APP_VERSION`.

## 1.8.1

- **LoRA delete fixed:** you can now delete LoRAs you added yourself too (not only the ones the program trained). The
  file is released before deleting and the move is retried, so a LoRA that is still loaded no longer blocks deletion.
  Deleted LoRAs still go to `loras\.trash` and are never overwritten.
- **Training window — live preview:** the latest sample image now shows right in the training panel, with a
  **"next sample in ~X"** countdown so you can see when the next preview is coming.
- **Character-LoRA training:** when a caption already starts with the trigger word, the trigger is no longer added a
  second time (no more `trigger, trigger woman …` duplication).
- **More stable builds:** windows (training, network, etc.) open reliably; an unexpected error now shows a message
  instead of closing the program, and is written to `logs\gui_errors.log`.

## 1.4.8

- Dropdown lists no longer clip text — every picker widens to fit its longest line.
- Real training speed is measured and saved to `logs\train_speed.json`; dataset tools use it to estimate training time.
- Datasets can carry a training hint (model, duration, checkpoint interval, hide-faces) that fills the training settings.
- A dataset image's `tags` (e.g. "full body, side view, standing") are added to the training text; duplicates already in the caption are skipped.
- "Hide faces" now works in Anatomy mode too: faces are blurred and not learned while bodies are learned fully.
