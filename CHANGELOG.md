# Changelog

All notable changes to AI Local Image Studio are listed here. This repository is the free, open-source core
(image generation, prompt writer, LoRA training, network server, LoRA tools). A one-click Windows **PRO** build for
supporters adds separate advanced tools — image→video + video LoRA training, a standalone editor/retouch tool, a
batch upscaler, and **Dataset Studio** — at **[patreon.com/GTEtemcan](https://www.patreon.com/GTEtemcan)**.

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
