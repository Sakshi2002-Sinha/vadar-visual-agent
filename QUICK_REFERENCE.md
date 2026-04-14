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

# Disable GPT-4o vision (text-only, faster / cheaper)
python evaluate_benchmark.py \
    --test-cases my_test_cases.json \
    --output-dir outputs/my_run \
    --no-vision

# Enable CLIP attribute extraction (color / material / size)
python evaluate_benchmark.py \
    --test-cases my_test_cases.json \
    --output-dir outputs/my_run \
    --use-clip

# Compute accuracy from saved results
python evaluate_benchmark.py \
    --results outputs/my_run/evaluation_results.json \
    --compute-accuracy
```

### Pre-download HuggingFace models

```bash
python -c "
from transformers import pipeline
pipeline('object-detection', model='facebook/detr-resnet-50')
pipeline('depth-estimation', model='Intel/dpt-large')
# Optional CLIP
from transformers import CLIPModel, CLIPProcessor
CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')
print('All models cached.')
"
```

---

## Python API Quick Reference

```python
from vadar_agent import VADARAgent
import os

# Initialise with all novel enhancements enabled
import torch
agent = VADARAgent(
    api_key=os.environ["OPENAI_API_KEY"],
    use_gpu=torch.cuda.is_available(),
    model="gpt-4o",               # OpenAI model for code generation
    detection_threshold=0.7,      # Feature 5 – confidence threshold
    nms_iou_threshold=0.5,        # Feature 5 – NMS duplicate suppression
    use_vision=True,              # Feature 2 – GPT-4o Vision multimodal
    use_clip=False,               # Feature 6 – CLIP attributes (slower)
    depth_inversion_guard=True,   # Feature 7 – auto-flip inverted depth maps
    code_repair_retries=2,        # Feature 3 – self-repair on exec failure
)

# Analyse an image
scene = agent.analyze_image("path/to/image.jpg")
print(f"{len(scene.objects)} objects detected")
for obj in scene.objects:
    print(f"  {obj.label}  depth={obj.depth_value:.3f}  color={obj.color}")

# Answer a question (full pipeline with all enhancements)
result = agent.answer_question(
    question="Is the chair farther than the table?",
    image_path="path/to/image.jpg",
)

print(result["answer"])   # e.g. True / "Yes" / "3"
print(result["status"])   # "Success" or error message
print(result["code"])     # generated Python program
```

---

## Extended SpatialReasoner API  (Feature 8)

The following new methods are available in generated code:

| Method | Description |
|--------|-------------|
| `SpatialReasoner.iou(obj1, obj2)` | IoU between bounding boxes |
| `SpatialReasoner.is_occluded(obj1, obj2)` | True when IoU ≥ 0.3 |
| `SpatialReasoner.is_between(obj_a, obj_b, obj_c)` | True when obj_b is spatially between obj_a and obj_c |
| `SpatialReasoner.count_objects_of_type(objects, label)` | Count by label substring |
| `SpatialReasoner.closest_to(objects, ref)` | Nearest neighbour (depth + pixel) |
| `SpatialReasoner.depth_rank(objects)` | Sorted closest → furthest |

---

## Output Interpretation Guide

### `evaluation_results.json`

| Field | Description |
|-------|-------------|
| `sample_id` | Unique identifier for the image |
| `image_path` | Absolute or relative path to the input image |
| `timestamp` | ISO-8601 timestamp when the sample was processed |
| `objects_detected` | All SpatialObject dicts (including color/material/size) |
| `questions_and_answers[*].question` | The spatial reasoning question |
| `questions_and_answers[*].answer` | Produced value (bool, int, str, …) |
| `questions_and_answers[*].status` | `"Success"` or exception message |
| `questions_and_answers[*].correct` | Fuzzy-match result vs ground truth |
| `questions_and_answers[*].question_category` | depth / spatial / counting / color / other |
| `questions_and_answers[*].answer_type` | boolean / numeric / string |
| `questions_and_answers[*].code` | Generated Python source |

### `summary_report.json`

| Field | Description |
|-------|-------------|
| `accuracy` | Overall fuzzy-match accuracy |
| `execution_success_rate` | Fraction of programs that ran without error |
| `per_category_accuracy` | Accuracy broken down by question category |
| `answer_type_distribution` | Count of boolean / numeric / string answers |

### Visual trace panels

```
┌──────────────────┬──────────────────┐
│  Original image  │   Depth map       │
│  (bounding boxes)│  (viridis cmap)   │
├──────────────────┼──────────────────┤
│  Object list     │  Question        │
│  (label, depth,  │  Answer          │
│   color [CLIP])  │                  │
└──────────────────┴──────────────────┘
```

Depth map colour scale:
- **Yellow / bright** → close to camera (low depth value)
- **Purple / dark**   → far from camera (high depth value)

---

## Performance Benchmarks

### Baseline (original VADAR paper)

| Benchmark | Task | Accuracy |
|-----------|------|----------|
| Omni3D-Bench | Spatial QA | ~62 % |
| CLEVR | Spatial reasoning | ~91 % |
| GQA | Visual QA | ~54 % |

### Expected gains from novel enhancements

| Enhancement | Expected gain |
|-------------|--------------|
| Region-median depth (Feature 1) | +3–6% on Omni3D-Bench |
| GPT-4o Vision input (Feature 2) | +8–15% on GQA (color/attr questions) |
| Self-repair execution (Feature 3) | ↓ exec failures → near zero |
| Fuzzy evaluation (Feature 4) | Removes formatting false-negatives |
| Confidence + NMS filter (Feature 5) | Fewer ghost detections |
| CLIP attributes (Feature 6) | Enables color/material questions |
| Depth inversion guard (Feature 7) | Corrects ~5–10% of depth maps |
| Extended SpatialReasoner (Feature 8) | Better first-attempt code success |

### Typical runtime (per image, CPU)

| Stage | Time |
|-------|------|
| Detection (DETR) | ~8 s |
| Depth (DPT-Large) | ~12 s |
| CLIP attributes (optional) | ~2 s |
| Code gen (GPT-4o) | ~3 s |
| **Total** | **~25 s** |

---

## Key Files at a Glance

| File | Purpose |
|------|---------|
| `vadar_agent.py` | Core agent classes (all novel features) |
| `evaluate_benchmark.py` | Batch evaluation with fuzzy metrics |
| `quickstart.py` | Dependency check + single-image demo |
| `config.yaml` | Default model / pipeline configuration |
| `.env.example` | Environment variable template |
| `SETUP_GUIDE.md` | Full installation instructions |
| `EXECUTION_GUIDE.md` | How to run on datasets |
| `TROUBLESHOOTING.md` | Common issues and fixes |
| `QUICK_REFERENCE.md` | This file |
