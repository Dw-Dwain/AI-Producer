# AI Drama Production Studio: Deployment & Operations Manual

This guide provides the instructions for deploying, configuring, and operating the AI Drama Production Studio on **RunPod Linux GPU Instances**, as well as local Windows/Linux development environments.

---

## 📂 1. Directory Structure Layout

In a production environment (RunPod), the application should reside under the `/workspace/studio` directory. The layout is structured as follows:

```text
/workspace/studio/
├── app.py                    # Main app runner entrypoint
├── database/                 # Database schema and SQLite database
│   ├── schema.sql            # SQLite schema configuration
│   └── studio.db             # Active production SQLite database
├── deployment/               # System validation & diagnostics
│   ├── paths.py              # pathlib dynamic path abstractions
│   ├── logging_setup.py      # Logger isolation router
│   ├── diagnostics.py        # System health and model scanner
│   └── backup.py             # Database and assets backup manager
├── logs/                     # Isolated log output files
│   ├── app.log               # Server-side web logs
│   ├── worker.log            # Background render worker logs
│   ├── generation.log        # Model inference generation logs
│   ├── lipsync.log           # Lipsync subprocess execution logs
│   └── models.log            # Startup diagnostics and scanner logs
├── models/                   # Deep Learning Model weights
│   ├── flux/                 # FLUX SFT model checkpoints
│   ├── ltx/                  # LTX Video checkpoints and VAEs
│   ├── wan/                  # WAN 2.2 model files
│   ├── hunyuan/              # Hunyuan video checkpoints
│   ├── loras/                # Custom LoRA models (.safetensors)
│   ├── Wav2Lip/              # Wav2Lip code and weights
│   ├── MuseTalk/             # MuseTalk code and weights
│   └── SyncTalk/             # SyncTalk code and weights
└── output/                   # Saved media outputs
    ├── videos/               # Generated project videos
    └── _lipsync/             # Synced output archives
```

---

## 🛠️ 2. RunPod Deployment & Installation

### Step A. Launching a RunPod GPU Instance
1. Deploy an instance using the **RunPod PyTorch Community Template** (CUDA 12.1+).
2. Choose a GPU with at least **24 GB VRAM** (e.g. RTX 4090, A6000, or A100) to support FLUX, LTX, and Wan models.
3. Expose port `7860` for the Web UI in the container settings.

### Step B. Clone and Initialize Workspace
Open a terminal inside the RunPod instance and run:
```bash
# Clone the repository to workspace
cd /workspace
git clone <repository_url> studio
cd /workspace/studio

# Install standard dependencies
pip install -r studio/requirements.txt
pip install imageio[ffmpeg] soundfile kokoro
```

### Step C. Deploying Deep Learning Models
Place the checkpoints under `/workspace/studio/models/` in their corresponding subdirectories:
- **Flux Dev**: Place `flux1-dev.sft` in `models/flux/`
- **LTX-Video**: Place `ltx-video-2b.sft` in `models/ltx/`
- **Wan 2.2**: Place `wan2.2-i2v.safetensors` in `models/wan/`
- **Hunyuan Video**: Place `hunyuan_video.sft` in `models/hunyuan/`
- **Wav2Lip**:
  - Place Wav2Lip weights `wav2lip_gan.pth` in `models/Wav2Lip/checkpoints/`
- **MuseTalk**:
  - Place MuseTalk checkpoints in `models/MuseTalk/models/musetalk/`
- **SyncTalk**:
  - Place SyncTalk checkpoints in `models/SyncTalk/checkpoints/`

---

## ⚙️ 3. Environment Variables Configuration

The application is completely path-abstracted using `pathlib` and adapts its behavior dynamically based on the following environment variables:

| Variable | Description | Recommended (RunPod Production) | Default (Local Development) |
|---|---|---|---|
| `STUDIO_ENV` | Target execution profile (`production` or `development`) | `production` | `development` |
| `STUDIO_ROOT` | Dynamic project absolute root directory path | `/workspace/studio` | `.` (App directory) |
| `STUDIO_DB_PATH` | SQLite database file location path | `/workspace/studio/database/studio.db` | `studio/database/studio.db` |
| `STUDIO_OUTPUT_DIR` | Directory folder to store generated assets | `/workspace/studio/output` | `studio/output` |
| `STUDIO_LOG_DIR` | Folder target for isolated logs | `/workspace/studio/logs` | `studio/logs` |
| `STUDIO_MODEL_DIR` | Directory to scan for checkpoints | `/workspace/studio/models` | `studio/models` |
| `PORT` | Web UI Gradio container server port | `7860` | `7860` |

To start the server in production mode on RunPod, execute:
```bash
export STUDIO_ENV=production
export STUDIO_ROOT=/workspace/studio
python studio/app.py --port 7860
```

---

## 🏥 4. Health Checks & Diagnostic Console

On application startup, a comprehensive audit validation is performed:
1. **Database Schema check**: Ensures that connection is successful and all 13 core tables exist. If corrupt or missing, a warning banner is shown.
2. **Storage checking**: Calculates database file size and free disk space. If free space falls below 10 GB, warning flags are triggered.
3. **GPU check**: Confirms PyTorch is compiled with CUDA and reports VRAM device visibility.
4. **Model Scanner**: Scans all directories and reports each model check status as `Available`, `Missing`, or `Invalid (File size < 1KB)`.

All system statuses and diagnostic info can be audited directly inside the Gradio UI under the **🏥 System Health & Model Discovery Dashboard** accordion on the **Settings & Status** tab.

---

## ♻️ 5. Backup & Database Maintenance

To secure pre-production work:
- **ZIP Backup creation**: Click **"Create ZIP Backup"** on the Settings tab. It exports `studio.db` and the asset directories (`characters`, `locations`, `projects`, `audio_assets`, `output` references) into a timestamped zip under `studio/backups/`.
- **ZIP Recovery**: Select a backup from the dropdown menu and click **"Restore Selected Backup"**. The system performs atomic extraction and verification to prevent data loss.

---

## 🚨 6. Troubleshooting Guide

### Issue: Startup Warning Alert Banner "🚨 Critical Startup Validation Failures Detected!"
- **Cause**: The SQLite database file has been deleted or schema is modified.
- **Solution**: The system auto-generates a clean database structure from `schema.sql` on startup. If table errors persist, clean out the corrupted database file and restart the application.

### Issue: Logs Selector shows "Log file not found yet"
- **Cause**: No operations (rendering, audio TTS, lip sync) have been queued yet, so the respective handler has not written events.
- **Solution**: Queue a dummy job or start the worker thread to initialize logs.

### Issue: Disk Space Warnings
- **Cause**: High-resolution video generation outputs have filled the storage.
- **Solution**: Delete rejected video files or download backups to local storage and clear space in `/workspace/studio/output`.
