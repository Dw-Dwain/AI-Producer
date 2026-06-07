# Character Consistency Guide — FLUX + ComfyUI + LoRA + Wan-S2V

How to lock a character's identity across an entire episode (the "Stardust TV"
approach), and add native audio-driven dialogue. This is the **front half** of the
pipeline that feeds your studio app's Wan I2V. See `MASTER_GUIDE.md` for the app
itself and the download playbook.

> **Status:** ComfyUI + FLUX installed and running. LoRA training + Wan-S2V planned.
> **Pod:** A100 SXM 80 GB (ideal — Wan-S2V *requires* 80 GB; A100 has no fp8 but
> that doesn't matter for FLUX/Wan).

---

## The consistency principle

Consistency is won **upstream**, not in the video model:

```
LAYER 1: FLUX character LoRA   → the identity (same face/body across the series)
LAYER 2: Reference stills      → per-scene wardrobe/lighting anchors
LAYER 3: Wan I2V (studio app)  → animates the locked still; face can't drift
LAYER 4: Wan-S2V / lipsync     → dialogue with synced mouth
```

| Level | Locked by | Guarantees |
|---|---|---|
| Identity | FLUX LoRA | same face/body across the whole series |
| Scene look | reused FLUX prompt + seed | same wardrobe/lighting within a scene |
| Motion | Wan I2V from locked still | face can't drift mid-shot |
| Voice | XTTS voice clone (or Wan-S2V) | same voice per character |

---

## Part A — ComfyUI + FLUX (DONE)

```bash
cd /workspace
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI
pip install -r requirements.txt
```

### FLUX models (FLUX.1-dev is gated — accept at huggingface.co/black-forest-labs/FLUX.1-dev)
```bash
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=1
cd /workspace/ComfyUI
hf download black-forest-labs/FLUX.1-dev flux1-dev.safetensors --local-dir models/diffusion_models --max-workers 1
hf download black-forest-labs/FLUX.1-dev ae.safetensors          --local-dir models/vae           --max-workers 1
hf download comfyanonymous/flux_text_encoders clip_l.safetensors --local-dir models/clip          --max-workers 1
hf download comfyanonymous/flux_text_encoders t5xxl_fp16.safetensors --local-dir models/clip      --max-workers 1
```
Files land in:
| File | Folder |
|---|---|
| `flux1-dev.safetensors` (~24 GB) | `models/diffusion_models` |
| `ae.safetensors` | `models/vae` |
| `clip_l.safetensors`, `t5xxl_fp16.safetensors` | `models/clip` |

### Launch + access
```bash
cd /workspace/ComfyUI
nohup python main.py --listen 0.0.0.0 --port 8188 > /workspace/comfy.log 2>&1 &
# tunnel (use 127.0.0.1, not localhost — see MASTER_GUIDE troubleshooting)
nohup cloudflared tunnel --url http://127.0.0.1:8188 > /workspace/cf.log 2>&1 &
sleep 6 && grep trycloudflare /workspace/cf.log
```
Open the `trycloudflare.com` URL.

---

## Part B — Generate the character (in ComfyUI)

1. **Workflow → Browse Templates → "Flux Dev"** (auto-wires the nodes).
2. Set the loaders: Load Diffusion Model = `flux1-dev.safetensors`; DualCLIPLoader =
   `t5xxl_fp16` + `clip_l`, type **flux**; Load VAE = `ae.safetensors`.
3. **Positive prompt** — rich + specific, e.g.:
   > *Cinematic portrait of a woman in her late 20s, sharp jawline, dark brown wavy
   > shoulder-length hair, hazel eyes, light freckles, charcoal wool coat, neutral
   > expression, soft window light, shallow depth of field, photorealistic, 85mm,
   > vertical 3:4 framing*
4. Resolution vertical, e.g. **832 × 1216**. Queue (Ctrl+Enter). ~10–20 s/image on A100.
5. **Lock the face:** when you love one, note its **seed**, set `control_after_generate`
   to **fixed**. That seed + description = your character anchor.

### Character sheet (training data)
Generate ~20–30 images of the **same person**, varied pose/expression/lighting/angle.
Two methods:
- **PuLID-FLUX** (best face lock): install the `ComfyUI-PuLID-Flux` custom node + its
  face model, feed your hero image as reference, vary everything else.
- **Same-seed variations** (no extra downloads): reuse the description, vary only
  pose/expression words, hand-pick the frames where the face matches.

Curate the best ~20–30 into `/workspace/training_data/<name>/` with a `.txt` caption
per image (e.g. `maya, a woman with dark wavy hair`).

---

## Part C — Train the character LoRA (ai-toolkit) — PLANNED

```bash
cd /workspace
git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
pip install -r requirements.txt
```
- Use a FLUX LoRA training config pointed at `/workspace/training_data/<name>/`
- ~1500–2500 steps, ~30 min on the A100
- Output: `<name>_character.safetensors` → copy to `ComfyUI/models/loras/`

(Exact training YAML to be added when this stage is set up.)

### Use the LoRA
In ComfyUI: add a **LoraLoader** after the diffusion model, load
`<name>_character.safetensors`, trigger word = your caption token. Now
`FLUX + LoRA + "<name>, <scene>"` → the same character in any scene.

---

## Part D — Feed the studio app (Wan I2V)

1. Generate the per-shot **starting frame** in ComfyUI (FLUX + LoRA + scene prompt).
2. In the studio app → Shot Builder → Model Family **Wan 2.2** → upload that still as
   **Reference Image** (auto-switches to I2V) → prompt the motion → generate.
3. Identity is preserved because the first frame *is* the locked character.

---

## Part E — Dialogue with Wan2.2-S2V (audio-driven) — PLANNED

Wan-S2V generates a talking character from a **reference image + speech audio** with
synced lips and natural motion. Replaces the LTX-2.3 audio+video role on the A100
(and, unlike LTX-2.3, runs great here — no fp8 needed).

- Repo: `Wan-AI/Wan2.2-S2V-14B` (HuggingFace). **Requires ~80 GB VRAM single-GPU** —
  the A100 is exactly right; 48 GB Ada cards cannot run it single-GPU.
- Runs via the Wan2.2 repo's `generate.py`:
  ```bash
  python generate.py --task s2v-14B --size 1024*704 \
      --ckpt_dir ./Wan2.2-S2V-14B/ --offload_model True --convert_model_dtype \
      --prompt "..." --image ref.jpg --audio voice.wav
  ```
  (Video length auto-matches the audio unless `--num_clip` is set.)
- **Integration plan:** wrap as a subprocess model family (same pattern as
  `ltx2_manager.py` / `lipsync_manager.py`): reference image + audio in → muxed
  talking-head MP4 out → register in `generated_videos`, skipping TTS+lipsync.

---

## Disk budget (500 GB volume)

After dropping LTX-2.3 + Gemma (freed ~171 GB):
| Asset | ~Size |
|---|---|
| Wan 2.2 T2V + I2V | 236 GB |
| FLUX.1-dev + encoders | 33 GB |
| Wan2.2-S2V-14B | ~70–80 GB |
| Character LoRAs | <1 GB each |
| **Total** | **~340–350 GB** ✅ fits |

---

## Resume checklist (where we left off)

- [x] ComfyUI installed + running on `:8188`
- [x] FLUX.1-dev + VAE + encoders downloaded
- [x] cloudflared tunnel (use `127.0.0.1`, not `localhost`)
- [ ] Open ComfyUI in browser, generate hero character face
- [ ] Build character sheet (PuLID-Flux or same-seed)
- [ ] Install ai-toolkit, train character LoRA
- [ ] Wire LoRA stills → studio app Wan I2V
- [ ] (Optional) Integrate Wan2.2-S2V for dialogue
