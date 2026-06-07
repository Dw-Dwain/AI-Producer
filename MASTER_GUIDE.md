# AI Drama Production Studio — Master Guide (The Bible)

Everything needed to deploy, configure, and run this app from scratch on any
GPU instance (RunPod or similar). Written from a real end-to-end deployment,
including every gotcha hit along the way and how to fix it.

> **Repo:** https://github.com/Dw-Dwain/AI-Producer
> **What it is:** A shot-driven AI video production studio (Gradio web app) for
> generating vertical short-form episodic drama. Wan 2.2 is the primary model,
> Hunyuan secondary, LTX-2.3 for native audio+video, LTX-Video 0.9.x for drafts.

---

## Table of Contents

1. [What the app does](#1-what-the-app-does)
2. [Hardware: which pod to use](#2-hardware-which-pod-to-use)
3. [Pod creation](#3-pod-creation)
4. [Clone the repo](#4-clone-the-repo)
5. [Install dependencies](#5-install-dependencies)
6. [Downloading models (READ THIS CAREFULLY)](#6-downloading-models-read-this-carefully)
7. [Model reference table (correct repos + paths)](#7-model-reference-table)
8. [LTX-2.3 special integration](#8-ltx-23-special-integration)
9. [Launching the app](#9-launching-the-app)
10. [Configuring models in the UI](#10-configuring-models-in-the-ui)
11. [Generating video — the workflows](#11-generating-video--the-workflows)
12. [Episode assembly](#12-episode-assembly)
13. [Restarting / recovering a pod](#13-restarting--recovering-a-pod)
14. [Troubleshooting (every issue + fix)](#14-troubleshooting)
15. [Architecture reference](#15-architecture-reference)

---

## 1. What the app does

A full production pipeline for AI episodic drama — script to finished vertical MP4:

- **Pre-production:** projects, episodes, characters (with visual DNA), locations, story bible
- **Shot planning:** scene dialogue → shot breakdown with camera directions
- **Generation:** queue-based background worker drives the video models
- **Voice:** Kokoro TTS per character (unless using LTX-2.3 which does audio natively)
- **Lip sync:** Wav2Lip / MuseTalk / SyncTalk (not needed for LTX-2.3 shots)
- **Consistency engines:** face/wardrobe drift scoring, continuity tracking, performance & cinematography prompt building
- **Assembly:** approved shots + audio + subtitles → finished episode MP4 (ffmpeg)
- **Output:** `studio/output/videos/{Project}/{Character}/`

**Model priority for realistic drama:**
| Model | Role | Strength |
|---|---|---|
| **Wan 2.2 I2V** | Primary — character-consistent shots | realism + same face across shots |
| **Wan 2.2 T2V** | Primary — shots without a reference | realism, motion |
| **Hunyuan Video** | Secondary — wide/establishing | cinematic motion |
| **LTX-2.3** | Native audio+video | synced voice, skips TTS+lipsync |
| **LTX-Video 0.9.x** | Drafts / B-roll | fast iteration |

---

## 2. Hardware: which pod to use

**Recommended: A100 SXM 80 GB** (or RTX Pro 6000 96 GB if available).

### Why 80 GB single GPU
- Wan 2.2 **A14B is a Mixture-of-Experts** model: two ~14B experts (~56 GB bf16) + ~11 GB text encoder. With `model_cpu_offload` (which the code auto-enables) it swaps experts and fits comfortably on 80 GB.
- Hunyuan (~13B) and LTX-2.3 (with fp8) also fit.

### Critical rules
- **Single GPU only.** The app loads each model on one device — it does **not** shard across GPUs. A "2x / 3x / 4x" pod's total VRAM is a lie for this app; only per-GPU VRAM matters.
- **48 GB is the practical minimum** (Wan A14B with offload). 24 GB works only with heavy offload (slow).
- **Driver must support CUDA ≥12.7** if you want LTX-2.3 (it needs torch 2.7). Check with `nvidia-smi` — top-right `CUDA Version`. A100s on RunPod typically show 12.8–13.0. ✅
- **Storage: 500 GB volume minimum.** Models are huge (see §7). The volume has a **hard quota** — `df` shows the whole cluster, NOT your quota. Watch actual usage with `du`.

### VRAM by model (approx, bf16)
| Model | Weights | Fits 80 GB? |
|---|---|---|
| Wan 2.2 T2V A14B | ~56 GB (MoE) | ✅ with offload |
| Wan 2.2 I2V A14B | ~56 GB (MoE) | ✅ with offload |
| Hunyuan Video | ~26 GB | ✅ |
| LTX-2.3 (22B) | ~44 GB / ~22 GB fp8 | ✅ |

---

## 3. Pod creation

1. RunPod → Deploy → **A100 SXM 80 GB** (or RTX Pro 6000)
2. Template: **RunPod PyTorch 2.x** (CUDA 12.x base)
3. **Volume disk: 500 GB** mounted at `/workspace` (this persists across restarts)
4. Container disk: 50 GB
5. Expose **HTTP port 7860** (optional — `--share` gives a public URL anyway)
6. Deploy → connect via **RunPod Web Terminal** (more reliable than Jupyter under heavy I/O)

> **Use the RunPod Web Terminal, not Jupyter, for heavy work.** Jupyter becomes
> unresponsive when the network filesystem is under load (large downloads), and
> you can't open new terminals. The Web Terminal is independent.

---

## 4. Clone the repo

```bash
cd /workspace
git clone https://github.com/Dw-Dwain/AI-Producer.git AI-Studio-Producer
```

---

## 5. Install dependencies

```bash
cd /workspace/AI-Studio-Producer
git pull   # always get latest fixes

# PyTorch (CUDA 12.1 wheel works even on CUDA 13 hosts — it's forward-compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# App dependencies (diffusers, transformers, gradio, ftfy, kokoro, etc.)
pip install -r studio/requirements.txt

# Download accelerators + tools
pip install hf_transfer uv

# ffmpeg (required for video save + episode assembly)
apt-get update && apt-get install -y ffmpeg

# Verify GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expect: `CUDA: True NVIDIA A100-SXM4-80GB`

> **After every pod restart you must reinstall these** — they live on the
> ephemeral container disk, not `/workspace`. See §13.

---

## 6. Downloading models (READ THIS CAREFULLY)

This is where most time was lost. The `/workspace` mount is **MooseFS (a network
filesystem)** and it has three failure modes. Follow these rules and downloads
go smoothly.

### The golden rules

1. **`huggingface-cli` is deprecated — use `hf download`.**
2. **Use the diffusers-format repos** (see §7). Wrong repo = 404 or won't load.
3. **Disable Xet** — the Xet protocol's background writer crashes on MooseFS:
   ```bash
   export HF_HUB_DISABLE_XET=1
   ```
4. **Enable hf_transfer** for speed (after `pip install hf_transfer`):
   ```bash
   export HF_HUB_ENABLE_HF_TRANSFER=1
   ```
5. **Use `--max-workers 1`** — parallel workers fight over lock files on MooseFS and deadlock.
6. **Do NOT wrap large downloads in a short `timeout`.** A single big checkpoint
   (e.g. LTX-2.3's ~44 GB file) takes longer than 5 min to write to MooseFS; a
   short timeout kills it mid-write forever and accumulates junk. Run plain and
   let it finish.
7. **Clean `.cache` after each model** — staging files double your disk usage:
   ```bash
   rm -rf /workspace/models/<MODEL>/.cache
   ```

### The reliable download command (use this for every model)

```bash
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=1

hf download <REPO_ID> --local-dir /workspace/models/<DEST> --max-workers 1
```
Let it run to completion. The two progress bars:
- **`Downloading ... X/Y`** = bytes (the actual transfer)
- **`Fetching N files`** = files finalized (moved into place) — this is the one
  that matters; it climbs slowly on MooseFS (~1 min per large file). 100% on the
  byte bar but a low file count just means it's finalizing — **wait, don't kill**.

### If it genuinely stalls (lock deadlock)

Symptom: `Still waiting to acquire lock on ...` with zero movement for 2+ min,
or the file counter frozen while `du` of the folder isn't growing.

```bash
# From a SECOND terminal:
pkill -9 -f "hf download"
find /workspace/models/<DEST> -name "*.lock" -delete   # clear stale locks (MooseFS doesn't release them on kill)
# then re-run the same download command — it resumes
```
Killed processes can be stuck in D-state (uninterruptible) on MooseFS I/O —
`kill -9` may take 30–60s to land, or need a second terminal.

### Verifying a finished download

```bash
rm -rf /workspace/models/<DEST>/.cache       # reclaim staging space FIRST
du -sh /workspace/models/<DEST>              # real size
find /workspace/models/<DEST> -name "*.incomplete"   # should print NOTHING
ls /workspace/models/<DEST>                  # check model_index.json + subfolders
```

### Watch your quota

`df -h /workspace` shows the **cluster** (e.g. 334T) — useless for your quota.
Use `du` to track your real usage against the 500 GB volume:
```bash
du -sh /workspace/models/* 2>/dev/null
```
`OSError: [Errno 122] Disk quota exceeded` = your 500 GB is full. Delete unused
models and any `.cache` dirs.

---

## 7. Model reference table

**Always use the diffusers-format repos.** These load with the pipeline classes
the app calls.

| Model | HuggingFace repo | Local dir | ~Size |
|---|---|---|---|
| Wan 2.2 T2V (primary) | `Wan-AI/Wan2.2-T2V-A14B-Diffusers` | `/workspace/models/Wan2.2-T2V-A14B` | ~118 GB |
| Wan 2.2 I2V (consistency) | `Wan-AI/Wan2.2-I2V-A14B-Diffusers` | `/workspace/models/Wan2.2-I2V-A14B` | ~118 GB |
| Hunyuan Video (secondary) | `hunyuanvideo-community/HunyuanVideo` | `/workspace/models/HunyuanVideo` | ~25 GB |
| LTX-2.3 (audio+video) | `Lightricks/LTX-2.3` | `/workspace/models/LTX-2.3` | ~108 GB |
| LTX-Video 0.9.x (drafts) | `Lightricks/LTX-Video-0.9.7-distilled` | `/workspace/models/ltx-video-2b-v0.9.7-distilled` | ~15 GB |

> **WRONG repos that waste time / won't work:** `Wan-AI/Wan2.2-T2V-14B` (404 — it's
> `A14B-Diffusers`), `tencent/HunyuanVideo` (not diffusers format), `Lightricks/LTX-Video`
> (the mixed 0.9.x repo, not 2.3).

### Download commands (copy-paste)

```bash
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1

# Wan 2.2 T2V (primary)
hf download Wan-AI/Wan2.2-T2V-A14B-Diffusers --local-dir /workspace/models/Wan2.2-T2V-A14B --max-workers 1
rm -rf /workspace/models/Wan2.2-T2V-A14B/.cache

# Wan 2.2 I2V (character consistency)
hf download Wan-AI/Wan2.2-I2V-A14B-Diffusers --local-dir /workspace/models/Wan2.2-I2V-A14B --max-workers 1
rm -rf /workspace/models/Wan2.2-I2V-A14B/.cache

# Hunyuan (optional, secondary)
hf download hunyuanvideo-community/HunyuanVideo --local-dir /workspace/models/HunyuanVideo --max-workers 1
rm -rf /workspace/models/HunyuanVideo/.cache
```

> You don't need all models to start. **Wan T2V alone** gets you generating.
> Disk budget on a 500 GB volume: Wan T2V + Wan I2V (~236 GB) + one of
> {Hunyuan 25 GB, LTX-2.3 108 GB}. You can't fit everything — pick per project.

---

## 8. LTX-2.3 special integration

LTX-2.3 generates **synchronized audio + video** in one pass (shots made with it
skip TTS + lip-sync). It **cannot share the app's Python environment** — it needs
its own `uv` venv with torch ~2.7. The app runs it as a **subprocess** (handled
by `studio/generation/ltx2_manager.py`).

### Requirements
- Host driver supporting **CUDA ≥12.7** (`nvidia-smi` — A100 shows 12.8–13.0 ✅)
- ~108 GB for the model + ~10 GB for the venv

### Install the LTX-2 package (its own venv)

```bash
cd /workspace
git clone https://github.com/Lightricks/LTX-2.git
cd /workspace/LTX-2
pip install uv

# Build the venv on LOCAL disk, not MooseFS — MooseFS chokes on the thousands of
# small-file ops a venv build needs. Local disk does it in ~35s vs hanging forever.
export UV_CACHE_DIR=/root/.uv-cache
export UV_PROJECT_ENVIRONMENT=/root/ltx2-venv
uv sync --frozen          # if it errors on a CUDA wheel, drop --frozen: `uv sync`
```
Result: venv at **`/root/ltx2-venv`** → the venv python is `/root/ltx2-venv/bin/python`.

> **The LTX-2 venv is on the container disk (`/root`) — it's wiped on restart and
> must be rebuilt** (`uv sync` again, ~1 min from cache). The model on `/workspace` persists.

### Download the LTX-2.3 model + Gemma 3 encoder

```bash
# IMPORTANT: deactivate the LTX-2 venv first so the system `hf` (with hf_transfer) is used
deactivate 2>/dev/null
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1

# The model repo (~147 GB — contains dev + distilled checkpoints, upscalers, LoRAs)
hf download Lightricks/LTX-2.3 --local-dir /workspace/models/LTX-2.3 --max-workers 1
rm -rf /workspace/models/LTX-2.3/.cache

# Gemma 3 text encoder — REQUIRED, NOT bundled in the model repo.
# Must be this exact variant (per LTX-2 quick-start.md). Gated: accept license at
# https://huggingface.co/google/gemma-3-12b-it-qat-q4_0-unquantized first.
hf download google/gemma-3-12b-it-qat-q4_0-unquantized --local-dir /workspace/models/gemma-3-12b-it-qat --max-workers 1
rm -rf /workspace/models/gemma-3-12b-it-qat/.cache
```
The LTX-2.3 repo contains multiple variants; pick the matching `1.1` set:
| Role | File | UI field |
|---|---|---|
| Full checkpoint (stage 1) | `ltx-2.3-22b-dev.safetensors` | Checkpoint |
| Distilled LoRA (stage 2) | `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | Distilled LoRA |
| Spatial upsampler | `ltx-2.3-spatial-upscaler-x2-1.1.safetensors` | Spatial Upsampler |
| Gemma 3 encoder dir | `/workspace/models/gemma-3-12b-it-qat` | Gemma 3 Encoder Dir |
| (fast alt checkpoint) | `ltx-2.3-22b-distilled-1.1.safetensors` | use with `ltx_pipelines.distilled` |

> `--gemma-root` points to the **directory** containing `model*.safetensors`
> (standard HF format) — not a single file.
>
> **Disk note:** LTX-2.3 (~147 GB) + Gemma (~24 GB) + Wan T2V+I2V (~236 GB) = ~407 GB
> of a 500 GB volume. You can't also fit Hunyuan — prune per project.

### CLI reference (how the wrapper drives it)
The pipeline is invoked as a module in the venv:
```bash
/root/ltx2-venv/bin/python -m ltx_pipelines.ti2vid_two_stages \
    --checkpoint-path <ckpt> --gemma-root <gemma> \
    --spatial-upsampler-path <upsampler> --distilled-lora <lora> \
    --prompt "..." --negative-prompt "..." \
    --width W --height H --num-frames N --frame-rate F \
    --num-inference-steps S --seed SEED \
    --quantization fp8-cast --output-path out.mp4
```
Pipeline modules: `ltx_pipelines.distilled` (fastest), `ltx_pipelines.ti2vid_one_stage`, `ltx_pipelines.ti2vid_two_stages` (best), `ltx_pipelines.ti2vid_two_stages_hq` (highest quality, slowest). Set paths in the UI → Model Manager → "LTX-2.3" tab. The "Extra CLI args" field passes through any extra flags without code changes.

### ⚠️ GPU compatibility — LTX-2.3 on A100 (Ampere) is a poor fit

**LTX-2.3's `fp8-cast` quantization requires fp8 tensor cores — Ada / Hopper / Blackwell only (RTX 4090, L40/L40S, RTX 6000 Ada, H100, RTX Pro 6000).** The A100 is **Ampere — no fp8 hardware** — so on A100 you must run **bf16 (Quantization blank) + CPU Offload**, which is slow: ~40 min for a ~5-second clip (121 frames @ 25fps).

Per-pipeline CLI quirks the wrapper handles automatically (learned from each module's `--help`):
- `distilled` uses `--distilled-checkpoint-path`, `--lora`; has **no** `--negative-prompt` or `--num-inference-steps`; needs the distilled checkpoint (`...distilled-1.1.safetensors`)
- `ti2vid_two_stages` uses `--checkpoint-path`, `--distilled-lora`, `--negative-prompt`, `--num-inference-steps`; needs the dev checkpoint (`...22b-dev.safetensors`)
- both require `--spatial-upsampler-path`; width/height must be **multiples of 64**
- `--offload` takes a **value** (`CPU`/`NONE`/`DISK`), not a bare flag

**Recommendation:** On an A100, **skip LTX-2.3** and use **Wan2.2-S2V** for native audio+video dialogue (it's built for it and runs well on 80 GB). Only run LTX-2.3 on an fp8-capable card, where you set Quantization back to `fp8-cast` for ~real-time-ish speed. The integration code is complete and correct for that scenario.

---

## 9. Launching the app

```bash
cd /workspace/AI-Studio-Producer
STUDIO_ENV=production python studio/app.py --port 7860 --share
```
Watch for: `Running on public URL: https://xxxxx.gradio.live` — open that link.

**Run it in the background** (survives terminal disconnects, easy to kill cleanly):
```bash
nohup env STUDIO_ENV=production python studio/app.py --port 7860 --share > /workspace/app.log 2>&1 &
tail -f /workspace/app.log          # watch
pkill -f studio/app.py              # stop
```

`STUDIO_ENV=production` makes it use `/workspace/*` for logs/output/db.

### Exposing a UI that has no `--share` (e.g. ComfyUI on 8188)

The studio app uses Gradio `--share` (a public `gradio.live` URL). For tools without
that (ComfyUI, etc.), use a **cloudflared quick tunnel** — no RunPod port exposure,
no pod restart:

```bash
wget -q https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 -O /usr/local/bin/cloudflared
chmod +x /usr/local/bin/cloudflared
# IMPORTANT: use 127.0.0.1, NOT localhost — localhost resolves to IPv6 ::1 and
# cloudflared gets "connection refused" because the app listens on IPv4 0.0.0.0.
cloudflared tunnel --url http://127.0.0.1:8188
```
It prints a `https://<words>.trycloudflare.com` URL. Run it backgrounded with
`nohup ... > /workspace/cf.log 2>&1 &` then `grep trycloudflare /workspace/cf.log`.

> **Gotcha:** `cloudflared tunnel --url http://localhost:8188` fails with
> `dial tcp [::1]:8188: connect: connection refused` — it tries IPv6. Always use
> `http://127.0.0.1:8188`.

---

## 10. Configuring models in the UI

Open the gradio URL → **🎬 Video Generation** tab → **⚙️ Model Manager** (expand).

### Wan 2.2 (primary)
| Field | Value |
|---|---|
| Wan 2.2 Model Path | `/workspace/models/Wan2.2-T2V-A14B` |
| Wan 2.2 I2V Model Path | `/workspace/models/Wan2.2-I2V-A14B` |

→ **Save Paths** → **Load Wan Pipeline**. First load takes 1–3 min (reads ~118 GB,
sets up CPU offload). Wait for the green "loaded" badge.

### Hunyuan (optional)
| Field | Value |
|---|---|
| Hunyuan Model Path | `/workspace/models/HunyuanVideo` |

### LTX-2.3 (optional, audio+video)
| Field | Value |
|---|---|
| LTX-2 Repo Dir | `/workspace/LTX-2` |
| LTX-2 venv Python | `/root/ltx2-venv/bin/python` |
| Checkpoint | *(from `find ... *.safetensors`)* |
| Spatial Upsampler | *(the upsampler .safetensors)* |
| Distilled LoRA | *(the distilled .safetensors)* |
| Gemma 3 Encoder Dir | *(the gemma folder)* |

→ **Save Paths** → **Detect** (all dots should be 🟢).

---

## 11. Generating video — the workflows

### Text-to-video (no reference)
1. **Shot Builder** tab → Model Family **Wan 2.2** → Pipeline `text2video`
2. Preset: `draft` (fast test) / `production` / `cinema`
3. Write a concrete prompt; the default negative prompt is pre-filled
4. **🚀 Add to Render Queue**
5. **Render Queue** tab → 🔄 Refresh → watch status (`queued`→`running`→`completed`)
6. Select the row → preview → **✅ Approve** → it lands in the Library

### Image-to-video (character consistency — the key to a series)
1. Generate/obtain a clean **character reference portrait** (face, wardrobe, lighting locked) — use FLUX/SDXL separately, or an approved still
2. **Shot Builder** → Model Family **Wan 2.2**
3. **Upload the portrait** to the Reference Image box → it **auto-switches to image2video** and loads the I2V model
4. Prompt the action/shot → queue → Wan animates *from that frame*, keeping the same face every shot

### LTX-2.3 (audio+video in one)
- Model Family **LTX-2.3** → pipeline `two_stage` → prompt → queue.
- Output MP4 has synced audio — **no TTS or lip-sync needed** for these shots.

### Expectations
- **First generation per model is slow** (loads weights + offload). Wan draft ~3–8 min, production/cinema longer (MoE expert swapping is the cost of fitting 80 GB).
- Subsequent jobs reuse the cached pipeline.
- Real generation logs stream in the **terminal**, not the browser.

---

## 12. Episode assembly

Once shots are approved, the assembly engine concatenates them into a finished
episode MP4 (ffmpeg). It can stream-copy (fast) or burn subtitles.

- `studio/intelligence/episode_assembly.py` → `export_episode_video(project_id, episode_id, output_path, burn_subtitles=False)`
- Requires `ffmpeg` on PATH (installed in §5)
- Output: a single MP4 of all approved shots in shot order

---

## 13. Restarting / recovering a pod

A pod **restart** keeps `/workspace` (models, repos, db, outputs) but **wipes the
container disk** (all pip installs + the LTX-2 venv). After any restart:

```bash
# 1. Reinstall deps
cd /workspace/AI-Studio-Producer && git pull
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r studio/requirements.txt
pip install hf_transfer uv
apt-get update && apt-get install -y ffmpeg

# 2. Verify models survived (they're on /workspace)
du -sh /workspace/models/*

# 3. Rebuild the LTX-2 venv (only if using LTX-2.3)
cd /workspace/LTX-2
export UV_CACHE_DIR=/root/.uv-cache UV_PROJECT_ENVIRONMENT=/root/ltx2-venv
uv sync --frozen

# 4. Launch
cd /workspace/AI-Studio-Producer
STUDIO_ENV=production python studio/app.py --port 7860 --share
```

> **Restart, never Terminate.** Terminate wipes the volume too — you lose the models.

---

## 14. Troubleshooting

Every issue hit during deployment and its fix.

| Symptom | Cause | Fix |
|---|---|---|
| `huggingface-cli ... deprecated` | Old CLI | Use `hf download` |
| `Model 'X' not found` (404) | Wrong repo id | Use diffusers repos in §7 |
| `Internal Writer Error: Background writer channel closed` | Xet protocol on MooseFS | `export HF_HUB_DISABLE_XET=1` |
| `hf_transfer ... not available` | Package missing or wrong venv | `pip install hf_transfer`, or `deactivate` if in LTX-2 venv |
| `Still waiting to acquire lock` forever | Stale MooseFS lock from killed process | `find <dir> -name "*.lock" -delete`, re-run |
| Download counter frozen, `du` not growing | Real lock deadlock | kill → delete locks → re-run (resumes) |
| Download counter frozen but `du` growing | Finalizing a big file on slow FS | **Wait** — it's working |
| `kill -9` won't kill process | D-state (uninterruptible I/O on MooseFS) | Wait 30–60s, or use a 2nd terminal; worst case restart pod |
| `OSError: [Errno 122] Disk quota exceeded` | 500 GB volume full | `du -sh /workspace/models/*`, delete unused + `.cache` dirs |
| Model folder huge (e.g. 182 GB for a 46 GB model) | Accumulated `.cache`/`.incomplete` from retries | `rm -rf <dir>` and re-download clean (no short timeout) |
| `uv sync` hangs at "Preparing packages" | venv build on MooseFS | Build on local disk: `export UV_PROJECT_ENVIRONMENT=/root/ltx2-venv UV_CACHE_DIR=/root/.uv-cache` |
| Jupyter unresponsive / can't open terminal | Network FS saturated by download | Use RunPod **Web Terminal** instead |
| `Cannot find empty port in range: 7860-7860` | Old app instance holding the port | `pkill -9 -f studio/app.py`; if D-state, launch on `--port 7861` |
| `NameError: name 'ftfy' is not defined` | Missing diffusers optional dep (Wan I2V) | `pip install ftfy` (now in requirements) |
| OOM loading Wan A14B on 80 GB | MoE too big resident | Already fixed — code auto-enables `model_cpu_offload` below 120 GB VRAM |
| LTX-2.3 `fp8e4nv not supported in this architecture` | fp8 quant on Ampere (A100 has no fp8 tensor cores) | Set LTX-2.3 Quantization = **blank** (bf16) + enable **CPU Offload**. fp8-cast only works on Ada/Hopper/Blackwell (4090, L40, H100). |
| LTX-2.3 `Resolution not divisible by 64` | two-stage needs /64 dims | Already fixed — manager snaps width/height to /64 |
| LTX-2.3 OOM in bf16 on 80 GB | 22B bf16 + Gemma + two-stage | Enable **CPU Offload** (ltx2_offload) in the LTX-2.3 Model Manager |
| LTX-2.3 `invalid OffloadMode value: 'OffloadMode.CPU'` | offload wants bare enum name | Already fixed — wrapper passes `CPU`/`NONE` |
| LTX-2.3 `--offload: expected one argument` | offload is a value, not a flag | Already fixed |
| LTX-2.3 distilled `unrecognized --negative-prompt`/`--num-inference-steps` | distilled pipeline doesn't accept them | Already fixed (pipeline-aware args) |
| LTX-2.3 `ModuleNotFoundError: ltx_pipelines` | wrong venv python (empty `/workspace/LTX-2/.venv`) | Set venv python to `/root/ltx2-venv/bin/python` and **re-Save** in UI |
| cloudflared `dial tcp [::1]:8188 connection refused` | `localhost` resolves to IPv6 | Use `--url http://127.0.0.1:8188` |
| Cinema preset fails on LTX | Old wrong pipeline name | Already fixed (`two_stage_hq`) |
| `Multiple -pix_fmt options` warning | Redundant ffmpeg flag | Harmless; already cleaned in code |

### General download recovery cycle (memorize this)
```
kill -9 the download  →  delete *.lock  →  rm -rf .cache if bloated  →  re-run plain (no short timeout, --max-workers 1, Xet off)
```

---

## 15. Architecture reference

```
studio/
├── app.py                       # Gradio app assembly, tab registration, session restore, launch
├── requirements.txt             # all Python deps
├── deployment/
│   ├── paths.py                 # cross-platform paths (STUDIO_ENV=production → /workspace)
│   ├── logging_setup.py         # centralized logs
│   └── diagnostics.py           # startup health checks
├── database/
│   └── db_manager.py            # SQLite: projects/episodes/scenes/shots, queues, generated_videos, settings...
├── generation/
│   ├── worker.py                # background thread: polls queues, routes by model family, saves output
│   ├── wan_manager.py           # Wan 2.2 (WanPipeline/WanImageToVideoPipeline) — primary
│   ├── hunyuan_manager.py       # Hunyuan (HunyuanVideoPipeline)
│   ├── ltx_manager.py           # LTX-Video 0.9.x (LTXPipeline) — drafts
│   ├── ltx2_manager.py          # LTX-2.3 subprocess wrapper (own venv) — audio+video
│   ├── video_output_manager.py  # MP4 save (imageio/libx264), thumbnail, metadata sidecars
│   ├── tts_manager.py           # Kokoro TTS
│   └── lipsync_manager.py       # Wav2Lip/MuseTalk/SyncTalk subprocess
├── intelligence/
│   └── episode_assembly.py      # ffmpeg concat + optional subtitle burn
└── ui/
    ├── video_tab.py             # the generation UI (model manager, shot builder, queue, library)
    └── ...                      # project/character/location/scene/story_bible/timeline/dashboard/settings tabs
```

### Generation flow
```
UI (Shot Builder) → db.add_to_render_queue()
   → worker.py picks up job
       ├─ Wan/Hunyuan/LTX-0.9.x: load pipeline in-process (diffusers), generate frames,
       │     video_output_manager.save_video() → MP4 + thumb + metadata
       └─ LTX-2.3: ltx2_manager.generate() → subprocess in /root/ltx2-venv → finished MP4 (audio+video)
   → db.add_generated_video() (status: awaiting approval / approved)
   → UI Render Queue → Approve → Video Library → Export / Episode Assembly
```

### Key environment variables
| Var | Purpose |
|---|---|
| `STUDIO_ENV=production` | use `/workspace/*` for data |
| `STUDIO_ROOT` | override root path |
| `HF_HUB_DISABLE_XET=1` | required for downloads on MooseFS |
| `HF_HUB_ENABLE_HF_TRANSFER=1` | faster downloads (needs `hf_transfer`) |
| `UV_PROJECT_ENVIRONMENT=/root/ltx2-venv` | build LTX-2 venv on local disk |
| `UV_CACHE_DIR=/root/.uv-cache` | uv cache on local disk |

---

## Quick-start TL;DR (returning to a fresh pod)

```bash
# 1. Repo + deps
cd /workspace && git clone https://github.com/Dw-Dwain/AI-Producer.git AI-Studio-Producer
cd AI-Studio-Producer
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r studio/requirements.txt hf_transfer uv
apt-get update && apt-get install -y ffmpeg

# 2. Models (Wan T2V minimum to start)
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1
hf download Wan-AI/Wan2.2-T2V-A14B-Diffusers --local-dir /workspace/models/Wan2.2-T2V-A14B --max-workers 1
rm -rf /workspace/models/Wan2.2-T2V-A14B/.cache

# 3. Launch
STUDIO_ENV=production python studio/app.py --port 7860 --share

# 4. In the UI: Model Manager → set Wan path → Save → Load → Shot Builder → generate
```

---

*This guide reflects a real deployment on an A100 SXM 80 GB RunPod instance with a
MooseFS-backed 500 GB volume. The download gotchas (Xet, locks, quota, timeouts)
are specific to network-filesystem volumes; on a local-SSD instance most of §6
simplifies to a plain `hf download`.*
