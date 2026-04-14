# VADAR Visual Agent

A VADAR-inspired Visual Agent for 3D Spatial Reasoning on 2D Images.  
Implements agentic code generation and execution for spatial understanding,
using object detection, monocular depth estimation, and the OpenAI API.

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
  │
  ▼
SceneAnalysis  (SpatialObject list + depth map)
  │
  ▼
CodeGenerator  (OpenAI → Python program)
  │
  ▼
execute_code() → answer
```

See [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md) for a full pipeline explanation.

---

## Project structure

```
vadar-visual-agent/
├── vadar_agent.py          # Core agent classes
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