"""
conftest.py – Pytest / nbval configuration for VADAR Visual Agent tests.

Ensures the Jupyter notebook (demo_notebook.ipynb) runs correctly when
tested with pytest-nbval or nbmake:

    pytest --nbval demo_notebook.ipynb
    pytest --nbmake demo_notebook.ipynb

Environment variables defined here are injected into the notebook kernel
before execution, so the notebook does not need hard-coded credentials.
"""

import os
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Make the project root importable in all test modules / notebooks
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Fixtures available to all tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def api_key() -> str:
    """Return the OpenAI API key from the environment, or a placeholder."""
    return os.getenv("OPENAI_API_KEY", "sk-test-placeholder")


@pytest.fixture(scope="session")
def sample_image_path(tmp_path_factory) -> Path:
    """
    Return a path to a small sample image, downloading it if necessary.

    Uses a session-scoped tmp directory so the file is created once per
    pytest run.
    """
    import urllib.request

    SAMPLE_URL = (
        "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/"
        "Cat03.jpg/320px-Cat03.jpg"
    )
    sample_dir = tmp_path_factory.mktemp("sample_images")
    dest = sample_dir / "sample.jpg"

    if not dest.exists():
        try:
            urllib.request.urlretrieve(SAMPLE_URL, dest)
            # Validate that the file is a real image
            from PIL import Image as _Image
            _Image.open(dest).verify()
        except Exception:
            # Create a tiny white JPEG so tests can still run offline
            try:
                from PIL import Image
                Image.new("RGB", (64, 64), color=(200, 200, 200)).save(dest)
            except ImportError:
                dest.write_bytes(b"")  # empty fallback

    return dest


@pytest.fixture(scope="session")
def vadar_agent(api_key):
    """Return an initialised VADARAgent (vision model loading is deferred)."""
    from vadar_agent import VADARAgent
    return VADARAgent(api_key)


# ---------------------------------------------------------------------------
# nbval / nbmake kernel environment
# ---------------------------------------------------------------------------

def pytest_configure(config):
    """
    Inject environment variables into the process so that notebook cells
    that call os.getenv() receive the correct values during testing.
    """
    if not os.environ.get("OPENAI_API_KEY"):
        os.environ.setdefault("OPENAI_API_KEY", "sk-test-placeholder")
