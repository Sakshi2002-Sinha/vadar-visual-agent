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
| Contribution guide | [CONTRIBUTING.md](CONTRIBUTING.md) |

---

## 30-second start

```bash
# 1. Install dependencies
pip install -r requirements.txt
# or install as a package:
pip install -e .

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

### Docker (no local Python setup required)

```bash
docker build -t vadar-visual-agent .

# Synthetic demo
docker run --rm vadar-visual-agent --demo

# Real image
docker run --rm \
  -e OPENAI_API_KEY=sk-... \
  -v /path/to/images:/data \
  vadar-visual-agent --image /data/photo.jpg \
                     --question "Which object is closest?"
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
  │                 ↑ segmentation depth fallback for overlapping boxes
  ▼
CodeGenerator  (OpenAI → Python program)
  │             supports multi-turn follow-up questions
  ▼
execute_code() → answer
```

See [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md) for a full pipeline explanation.

---

## Multi-turn questions

After the first question, you can ask follow-up questions that reference
earlier results using `answer_followup()`:

```python
from vadar_agent import VADARAgent

agent = VADARAgent(api_key="sk-...")
result1 = agent.answer_question("Which object is closest?", "scene.jpg")
result2 = agent.answer_followup("How far is it from the camera?")
result3 = agent.answer_followup("Is there anything behind it?")
```

---

## Running the test suite

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Unit tests (no GPU / no API key)
pytest tests/test_spatial_reasoner.py tests/test_code_generator.py -v

# Integration tests (mocked vision models + OpenAI)
pytest tests/test_vadar_agent.py -v

# Full suite with coverage
pytest --cov=vadar_agent --cov-report=term-missing
```

CI runs automatically on every push via GitHub Actions (see `.github/workflows/ci.yml`).

---

## Benchmark

### Sample test cases

A sample dataset is provided in `data/sample_test_cases.json` with 5 scenes
and 10 spatial-reasoning questions. To run a benchmark on your own images:

```bash
# 1. Place your images under data/images/
# 2. Edit data/sample_test_cases.json with the correct paths and questions
# 3. Run evaluation
python evaluate_benchmark.py \
    --test-cases data/sample_test_cases.json \
    --output-dir outputs/my_benchmark

# 4. Compute accuracy (if ground_truth fields are filled in)
python evaluate_benchmark.py \
    --results outputs/my_benchmark/evaluation_results.json \
    --compute-accuracy
```

### Baseline results

> **Note:** Baseline results will be populated here after running the
> evaluator on a standard benchmark dataset (e.g. ScanQA or EQA).
> See [EXECUTION_GUIDE.md](EXECUTION_GUIDE.md) for download instructions.

| Dataset | Model | Execution success | Accuracy |
|---------|-------|:-----------------:|:--------:|
| *(pending)* | gpt-4o | – | – |

---

## Project structure

```
vadar-visual-agent/
├── vadar_agent.py          # Core agent classes
├── evaluate_benchmark.py   # Batch evaluation script
├── quickstart.py           # Dependency check + single-image demo
├── config.yaml             # Default configuration
├── requirements.txt        # Python dependencies
├── pyproject.toml          # Installable package definition
├── Dockerfile              # Container image (CPU)
├── .env.example            # Environment variable template
├── data/
│   └── sample_test_cases.json  # Sample benchmark questions
├── tests/
│   ├── test_spatial_reasoner.py   # Unit tests (pure Python)
│   ├── test_code_generator.py     # Unit tests with mocked OpenAI
│   └── test_vadar_agent.py        # Integration tests with mocked models
├── .github/workflows/
│   └── ci.yml              # GitHub Actions CI pipeline
├── SETUP_GUIDE.md          # Step-by-step installation guide
├── EXECUTION_GUIDE.md      # How to run on datasets
├── TROUBLESHOOTING.md      # Common issues and fixes
├── QUICK_REFERENCE.md      # Command cheat-sheet and benchmarks
├── CLONE_AND_SETUP_GUIDE.md # Cloning and bootstrapping guide
└── CONTRIBUTING.md         # How to contribute
```