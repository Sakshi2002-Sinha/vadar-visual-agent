# VADAR Visual Agent – Quick Reference

---

## Command Cheat-Sheet

### Setup

```bash
# First-time setup
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env && nano .env   # add OPENAI_API_KEY

# Verify everything works
python quickstart.py --verify-only
```

### Single-image inference

```bash
# Via CLI
python vadar_agent.py path/to/image.jpg "Which object is closest to the camera?"

# Via quickstart helper
python quickstart.py --image path/to/image.jpg \
                     --question "Is the red ball to the left of the blue cube?"
```

### Batch evaluation

```bash
# Run evaluation on a JSON test-cases file
python evaluate_benchmark.py \
    --test-cases my_test_cases.json \
    --output-dir outputs/my_run

# Compute accuracy from saved results
python evaluate_benchmark.py \
    --results outputs/my_run/evaluation_results.json \
    --compute-accuracy
```

### Pre-download HuggingFace models

```bash
python -c "
from transformers import pipeline
pipeline('object-detection',   model='facebook/detr-resnet-50')
pipeline('depth-estimation',   model='Intel/dpt-large')
pipeline('image-segmentation', model='facebook/detr-resnet-50-panoptic')
print('All models cached.')
"
```

### Clear HuggingFace model cache

```bash
rm -rf ~/.cache/huggingface/hub/models--facebook--detr*
rm -rf ~/.cache/huggingface/hub/models--Intel--dpt-large
```

---

## Python API Quick Reference

```python
from vadar_agent import VADARAgent
import os

# Initialise (GPU auto-detected via torch.cuda.is_available())
import torch
agent = VADARAgent(
    api_key=os.environ["OPENAI_API_KEY"],
    use_gpu=torch.cuda.is_available(),
    model="gpt-4o",          # OpenAI model for code generation
)

# Analyse an image
scene = agent.analyze_image("path/to/image.jpg")
print(f"{len(scene.objects)} objects detected")

# Answer a question (full pipeline)
result = agent.answer_question(
    question="Is the chair farther than the table?",
    image_path="path/to/image.jpg",
)

print(result["answer"])   # e.g. True / "Yes" / "3"
print(result["status"])   # "Success" or error message
print(result["code"])     # generated Python program
```

---

## Output Interpretation Guide

### `evaluation_results.json`

| Field | Description |
|-------|-------------|
| `sample_id` | Unique identifier for the image |
| `image_path` | Absolute or relative path to the input image |
| `timestamp` | ISO-8601 timestamp when the sample was processed |
| `questions_and_answers[*].question` | The spatial reasoning question |
| `questions_and_answers[*].answer` | Produced value (bool, int, str, …) |
| `questions_and_answers[*].status` | `"Success"` or exception message |
| `questions_and_answers[*].code` | Generated Python source |
| `questions_and_answers[*].code_path` | Path to saved `.py` file |
| `questions_and_answers[*].trace_path` | Path to saved trace `.png` |

### Visual trace panels

```
┌──────────────────┬──────────────────┐
│  Original image  │   Depth map       │
│  (bounding boxes)│  (viridis cmap)   │
├──────────────────┼──────────────────┤
│  Object list     │  Question        │
│  (label, depth)  │  Answer          │
└──────────────────┴──────────────────┘
```

Depth map colour scale:
- **Yellow / bright** → close to camera (low depth value)
- **Purple / dark**   → far from camera (high depth value)

---

## Performance Benchmarks (from the VADAR paper)

The original VADAR system (damianomarsili/VADAR) reported these accuracy
figures on standard benchmarks:

| Benchmark | Task | Accuracy |
|-----------|------|----------|
| Omni3D-Bench | Spatial QA | ~62 % |
| CLEVR | Spatial reasoning | ~91 % |
| GQA | Visual QA | ~54 % |

> These numbers are from the original VADAR paper.  Results for this
> re-implementation may differ depending on the OpenAI model version, prompt
> engineering, and vision model choices.

### Typical runtime (per image)

| Hardware | Detection | Depth | Code gen | Total |
|----------|-----------|-------|----------|-------|
| CPU (no GPU) | ~8 s | ~12 s | ~3 s | ~23 s |
| GPU (RTX 3090) | ~0.5 s | ~1 s | ~3 s | ~4.5 s |

---

## Key Files at a Glance

| File | Purpose |
|------|---------|
| `vadar_agent.py` | Core agent classes |
| `evaluate_benchmark.py` | Batch evaluation script |
| `quickstart.py` | Dependency check + single-image demo |
| `config.yaml` | Default model / pipeline configuration |
| `.env.example` | Environment variable template |
| `SETUP_GUIDE.md` | Full installation instructions |
| `EXECUTION_GUIDE.md` | How to run on datasets |
| `TROUBLESHOOTING.md` | Common issues and fixes |
| `QUICK_REFERENCE.md` | This file |
