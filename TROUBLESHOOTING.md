# VADAR Visual Agent – Troubleshooting Guide

---

## Table of Contents

1. [Installation Issues](#1-installation-issues)
2. [CUDA / PyTorch Compatibility](#2-cuda--pytorch-compatibility)
3. [Memory Management for Large Models](#3-memory-management-for-large-models)
4. [OpenAI API Issues](#4-openai-api-issues)
5. [Model Loading Issues](#5-model-loading-issues)
6. [OpenCV Errors](#6-opencv-errors)
7. [Code Execution Errors](#7-code-execution-errors)

---

## 1. Installation Issues

### `ModuleNotFoundError: No module named 'cv2'`

```bash
pip install opencv-python
# If that fails on a headless server:
pip install opencv-python-headless
```

### `ImportError: libGL.so.1: cannot open shared object file`

This happens on headless Linux servers:

```bash
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
# Or switch to the headless OpenCV build:
pip uninstall opencv-python
pip install opencv-python-headless
```

### `ERROR: Could not find a version that satisfies the requirement torch>=2.0.0`

Your pip may be outdated or targeting the wrong Python:

```bash
python -m pip install --upgrade pip
python -m pip install torch torchvision \
    --index-url https://download.pytorch.org/whl/cpu
```

### Dependency conflicts between `transformers` and `tokenizers`

```bash
pip install --upgrade transformers tokenizers
```

---

## 2. CUDA / PyTorch Compatibility

### Check what CUDA version is installed

```bash
nvidia-smi        # Shows driver CUDA version
nvcc --version    # Shows toolkit CUDA version
```

### Install the correct PyTorch build

Visit <https://pytorch.org/get-started/locally/> and pick your CUDA version:

```bash
# CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### Verify CUDA is visible to PyTorch

```python
import torch
print(torch.cuda.is_available())      # True if GPU is usable
print(torch.cuda.get_device_name(0))  # e.g. "NVIDIA GeForce RTX 3090"
```

### `RuntimeError: CUDA error: device-side assert triggered`

This usually means a tensor is on the wrong device. Ensure you pass
`use_gpu=True` to `VADARAgent` only when a GPU is actually present:

```python
import torch
agent = VADARAgent(api_key, use_gpu=torch.cuda.is_available())
```

---

## 3. Memory Management for Large Models

### Symptoms

- `RuntimeError: CUDA out of memory`
- System RAM swapping heavily during model load

### Solutions

**Reduce batch size / image resolution:**

```python
from PIL import Image
image = Image.open("image.jpg")
image = image.resize((640, 480))  # Smaller input → less VRAM
```

**Run models on CPU:**

```python
agent = VADARAgent(api_key, use_gpu=False)
```

**Load models one at a time** (edit `VisionModels.__init__` to lazy-load):

```python
self._object_detector = None  # load on first call
```

**Use half-precision (GPU only):**

```python
import torch
# After creating the pipeline, move it to fp16:
model = pipeline("depth-estimation", model="Intel/dpt-large",
                 torch_dtype=torch.float16, device=0)
```

**Free GPU memory between steps:**

```python
import torch, gc
del depth_map_tensor
gc.collect()
torch.cuda.empty_cache()
```

---

## 4. OpenAI API Issues

### `AuthenticationError: No API key provided`

Make sure the environment variable is exported **before** running Python:

```bash
export OPENAI_API_KEY="sk-..."
python quickstart.py
```

Or load your `.env` file:

```bash
export $(grep -v '^#' .env | xargs)
```

### `RateLimitError: You have exceeded your rate limit`

- Add retries with exponential back-off (the `openai` library ≥ 1.0 does this
  automatically with `max_retries`).
- Reduce concurrent requests.
- Upgrade your OpenAI plan or request a rate-limit increase.

### `openai.APIConnectionError` / timeout

- Check your internet connection.
- If you are behind a corporate proxy, set:

  ```bash
  export HTTPS_PROXY=http://proxy.example.com:8080
  ```

### `InvalidRequestError: model not found`

The default model is `gpt-4o`. If your API key does not have access, use:

```python
agent = VADARAgent(api_key, model="gpt-3.5-turbo")
```

---

## 5. Model Loading Issues

### HuggingFace model download is slow or fails

- Ensure you have at least **10 GB free disk** for the cache.
- Check the cache path: `echo $HF_HOME` (defaults to `~/.cache/huggingface/`).
- Retry: interrupted downloads are resumed automatically.

### `OSError: [Errno 28] No space left on device` during model download

Clear old cached models:

```bash
rm -rf ~/.cache/huggingface/hub/models--facebook--detr-resnet-50
```

Or point the cache to a larger volume:

```bash
export HF_HOME=/mnt/large_disk/.cache/huggingface
```

### `ValueError: Could not load model facebook/detr-resnet-50 with any of the …`

Reinstall the transformers library:

```bash
pip install --force-reinstall transformers
```

---

## 6. OpenCV Errors

### `error: (-215:Assertion failed) !_src.empty()`

The image path is wrong or the file is unreadable:

```python
from PIL import Image
img = Image.open("path/to/image.jpg")
print(img.size)  # Verify the image loads
```

### `cv2.error: OpenCV(4.x) … in function 'resize'`

The depth map or image array is empty. Make sure the vision pipeline completed
successfully before calling `cv2.resize`.

---

## 7. Code Execution Errors

### `answer` variable not assigned

The generated code did not produce an `answer` variable. This can happen when
the LLM generates a partial or malformed response. Retry the question – slight
rephrasing often helps.

### `NameError: name 'SpatialReasoner' is not defined`

This should not occur in normal use because `execute_code` injects
`SpatialReasoner` into the execution namespace. If it does, ensure you are
calling `code_generator.execute_code()` and not `exec(code)` directly.

### Generated code uses a library that is not available

The LLM occasionally imports libraries that are not installed. Inspect the
generated code file and install the missing dependency, or add it to
`requirements.txt`.
