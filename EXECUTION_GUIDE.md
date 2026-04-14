# VADAR Visual Agent – Execution Guide

This guide explains how to run the VADAR Visual Agent on standard benchmark
datasets and on your own images/datasets, and how to interpret the outputs.

---

## Table of Contents

1. [Running the Quick-Start Demo](#1-running-the-quick-start-demo)
2. [Running on Omni3D-Bench](#2-running-on-omni3d-bench)
3. [Running on CLEVR / GQA](#3-running-on-clevr--gqa)
4. [Using Custom Images or Datasets](#4-using-custom-images-or-datasets)
5. [Output Directory Structure](#5-output-directory-structure)
6. [Interpreting Results](#6-interpreting-results)
7. [Pipeline Components Deep Dive](#7-pipeline-components-deep-dive)

---

## 1. Running the Quick-Start Demo

```bash
python quickstart.py --image sample_images/room_001.jpg \
                     --question "Is the chair farther from the camera than the table?"
```

Expected console output:

```
[Step 1/4] Analysing image …  ✓  (6 objects detected)
[Step 2/4] Generating code …  ✓
[Step 3/4] Executing code …   ✓
[Step 4/4] Answer: Yes, the chair is farther from the camera than the table.

Generated code saved to: outputs/quickstart/code_20240101_120000.py
Visual trace saved to:   outputs/quickstart/trace_20240101_120000.png
```

---

## 2. Running on Omni3D-Bench

### Download the dataset

```bash
# Follow the official Omni3D data download instructions:
# https://github.com/facebookresearch/omni3d#dataset

# In your LOCAL GitHub clone (not Codespaces), download + extract annotations:
./download_omni3d_json.sh ~/datasets/omni3d

# Map extracted annotations into this repo layout:
python setup_omni3d_data.py \
  --source-annotations ~/datasets/omni3d/datasets/Omni3D \
  --annotations-only \
  --mode link \
  --force

# This creates the expected paths:
# data/omni3d/images
# data/omni3d/annotations

# Later, after you separately download Omni3D image assets, run full setup:
python setup_omni3d_data.py \
  --source-images /path/to/omni3d/images \
  --source-annotations ~/datasets/omni3d/datasets/Omni3D \
  --mode link \
  --force
```

### Prepare a test-cases JSON file

```json
[
  {
    "sample_id": "omni3d_000",
    "image_path": "data/omni3d/images/000000.jpg",
    "questions": [
      "Which object is closest to the camera?",
      "How many chairs are visible?"
    ],
    "ground_truth": ["chair", "3"]
  }
]
```

### Run evaluation

```bash
python evaluate_benchmark.py \
    --test-cases data/omni3d/test_cases.json \
    --output-dir outputs/omni3d
```

### Measure accuracy

```bash
python evaluate_benchmark.py \
    --results outputs/omni3d/evaluation_results.json \
    --compute-accuracy
```

---

## 3. Running on CLEVR / GQA

### CLEVR

```bash
# Download CLEVR from https://cs.stanford.edu/people/jcjohns/clevr/
# Place images in data/clevr/images/
# Place questions in data/clevr/questions/CLEVR_val_questions.json

python scripts/convert_clevr.py \
    --questions data/clevr/questions/CLEVR_val_questions.json \
    --output data/clevr/test_cases.json

python evaluate_benchmark.py \
    --test-cases data/clevr/test_cases.json \
    --output-dir outputs/clevr
```

### GQA

```bash
# Download GQA from https://cs.stanford.edu/people/dorarad/gqa/
# Place images in data/gqa/images/
# Place questions in data/gqa/questions/val_balanced_questions.json

python scripts/convert_gqa.py \
    --questions data/gqa/questions/val_balanced_questions.json \
    --output data/gqa/test_cases.json

python evaluate_benchmark.py \
    --test-cases data/gqa/test_cases.json \
    --output-dir outputs/gqa
```

---

## 4. Using Custom Images or Datasets

### Single image via CLI

```bash
python vadar_agent.py path/to/image.jpg "What object is in the foreground?"
```

### Python API

```python
from vadar_agent import VADARAgent
import os

agent = VADARAgent(api_key=os.environ["OPENAI_API_KEY"])

result = agent.answer_question(
    question="Is the red ball to the left of the blue cube?",
    image_path="path/to/image.jpg",
)

print(result["answer"])    # The final answer
print(result["code"])      # Generated Python code
print(result["status"])    # "Success" or error description
```

### Batch evaluation with a custom dataset

Create a JSON file in this format:

```json
[
  {
    "sample_id": "my_scene_001",
    "image_path": "my_data/images/scene_001.png",
    "questions": [
      "What color is the closest object?",
      "Are there more than three objects?"
    ]
  }
]
```

Then run:

```bash
python evaluate_benchmark.py \
    --test-cases my_test_cases.json \
    --output-dir outputs/custom
```

---

## 5. Output Directory Structure

```
outputs/
└── <run_name>/
    ├── evaluation_results.json   # Full results for every question
    ├── summary_report.json       # High-level accuracy / counts
    ├── code/
    │   ├── code_<id>_q0.py       # Generated Python code per question
    │   └── code_<id>_q1.py
    └── traces/
        ├── trace_<id>_q0.png     # Visual trace: image + depth + answer
        └── trace_<id>_q1.png
```

### `evaluation_results.json` schema

```json
[
  {
    "sample_id": "room_001",
    "image_path": "...",
    "timestamp": "2024-01-01T12:00:00",
    "questions_and_answers": [
      {
        "question_id": 0,
        "question": "Is the chair ...",
        "answer": "Yes",
        "status": "Success",
        "code": "...",
        "code_path": "outputs/.../code_room_001_q0.py",
        "trace_path": "outputs/.../traces/trace_room_001_q0.png"
      }
    ]
  }
]
```

---

## 6. Interpreting Results

### Programs (generated code)

Each `.py` file in `outputs/<run>/code/` is the exact Python program generated
by the OpenAI model. Programs use:

- `objects` – a list of `SpatialObject` instances from the scene
- `SpatialReasoner` – helper class for spatial comparisons
- `np` – NumPy for any numerical operations
- `answer` – the variable the program **must** assign its final answer to

Example generated program:

```python
# Question: Is the chair farther from the camera than the table?
chair = SpatialReasoner.get_object_by_label(objects, "chair")
table = SpatialReasoner.get_object_by_label(objects, "dining table")

if chair is None or table is None:
    answer = "Cannot determine – objects not detected"
else:
    answer = SpatialReasoner.is_farther(chair, table)
```

### Visual traces

Each `.png` trace contains four panels:

| Panel | Content |
|-------|---------|
| Top-left | Original image with bounding boxes |
| Top-right | Estimated depth map (warm = close, cool = far) |
| Bottom-left | Detected object list with depth values |
| Bottom-right | Question and final answer |

### Status codes

| Status | Meaning |
|--------|---------|
| `Success` | Code ran without exceptions; `answer` was produced |
| `Execution error: …` | Generated code raised a Python exception |

---

## 7. Pipeline Components Deep Dive

```
Image
  │
  ▼
VisionModels
  ├─ detect_objects()    → List[detection dicts]   (DETR)
  ├─ estimate_depth()    → np.ndarray              (DPT-Large)
  └─ segment_objects()   → List[segment dicts]     (DETR Panoptic)
  │
  ▼
SceneAnalysis  (objects + depth_map)
  │
  ▼
CodeGenerator.generate_code()   → Python source string  (OpenAI)
  │
  ▼
CodeGenerator.execute_code()    → (answer, status)
```

### Signature Agent (CodeGenerator)

- Builds a structured prompt describing all detected objects and their depths
- Calls `gpt-4o` (configurable) with temperature 0.3 for deterministic outputs
- The generated code signature always produces an `answer` variable

### Program Agent (execute_code)

- Runs the generated code in an isolated `exec()` context
- Injects `objects`, `SpatialReasoner`, `np`, and `scene_analysis`
- Captures `answer` from the local namespace after execution

### API Agent (VisionModels)

- Wraps three HuggingFace `pipeline` calls
- Depth values are normalized to `[0, 1]` (0 = closest, 1 = furthest)
- All bounding-box coordinates are normalized to `[0, 1]` before storage
