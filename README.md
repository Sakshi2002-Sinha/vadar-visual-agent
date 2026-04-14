# VADAR Visual Agent

A VADAR-inspired Visual Agent for 3D Spatial Reasoning on 2D Images.  
Implements agentic code generation and execution for spatial understanding,
using object detection, monocular depth estimation, and the OpenAI API.

> **Novel** – this version adds 10 enhancements over the original VADAR
> baseline, making it suitable for publication at ECCV / ICCV workshops or
> NAACL / EMNLP (VQA / spatial reasoning tracks).

---

## Quick links

| Resource | File |
|----------|------|
| Full setup instructions | [SETUP_GUIDE.md](SETUP_GUIDE.md) |
| Running on datasets | [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md) |
| Troubleshooting | [TROUBLESHOOTING.md](TROUBLESHOOTING.md) |
| Command cheat-sheet | [QUICK_REFERENCE.md](QUICK_REFERENCE.md) |
| Environment template | [.env.example](.env.example) |

---

## 30-second start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...

# 3. Verify the environment
python quickstart.py --verify-only

# 4. Run a quick demo (no image file needed)
python quickstart.py --demo

# 5. Run on your own image
python quickstart.py --image path/to/image.jpg \
                     --question "Which object is closest to the camera?"
```

---

## Architecture

```
Image
  │
  ▼
VisionModels  (DETR object-detection + DPT depth estimation)
  │  ├─ confidence threshold + IoU NMS filtering  [Feature 5]
  │  └─ depth inversion guard                     [Feature 7]
  │
  ▼
SceneAnalysis  (SpatialObject list + depth map)
  │  ├─ region-median depth per object            [Feature 1]
  │  └─ CLIP color / material / size attributes   [Feature 6]
  │
  ▼
CodeGenerator  (OpenAI GPT-4o + image vision input)
  │  ├─ multimodal vision prompt                  [Feature 2]
  │  └─ self-repair execution loop                [Feature 3]
  │
  ▼
execute_code_with_repair() → answer
```

---

## Novel enhancements (vs. original VADAR paper)

| # | Feature | Description |
|---|---------|-------------|
| 1 | **Region-median depth** | Median of all bbox pixels instead of single center pixel – robust for large / occluded objects |
| 2 | **GPT-4o Vision** | Image base64-encoded and sent alongside structured scene text, giving the LLM direct pixel context |
| 3 | **Self-repair execution** | Failed `exec()` feeds traceback back to the model; up to 2 automatic retries |
| 4 | **Fuzzy answer matching** | Boolean normalisation + ±5 % numeric tolerance + case-fold – eliminates formatting false-negatives |
| 5 | **Confidence threshold + NMS** | Drops ghost detections (default threshold 0.7) and removes IoU duplicates |
| 6 | **CLIP attributes** | Zero-shot color / material / size per bounding box – enables attribute-spatial questions |
| 7 | **Depth inversion guard** | Heuristic flip when bottom-of-frame depth > top-of-frame (catches inverted DPT outputs) |
| 8 | **Extended SpatialReasoner** | Six new helpers: `iou`, `is_occluded`, `is_between`, `count_objects_of_type`, `closest_to`, `depth_rank` |
| 9 | **Richer metrics dashboard** | Per-category accuracy, exec success rate, answer type distribution, confidence scatter plot |
| 10 | **Removed unused segmentation** | Panoptic model removed at startup – saves ~2 GB VRAM and ~8 s load time |

---

## Project structure

```
vadar-visual-agent/
├── vadar_agent.py          # Core agent classes (all novel features)
├── evaluate_benchmark.py   # Batch evaluation script
├── quickstart.py           # Dependency check + single-image demo
├── config.yaml             # Default configuration
├── requirements.txt        # Python dependencies
├── .env.example            # Environment variable template
├── SETUP_GUIDE.md          # Step-by-step installation guide
├── EXECUTION_GUIDE.md      # How to run on datasets
├── TROUBLESHOOTING.md      # Common issues and fixes
└── QUICK_REFERENCE.md      # Command cheat-sheet and benchmarks
```
