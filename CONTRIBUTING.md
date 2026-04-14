# Contributing to VADAR Visual Agent

Thank you for your interest in contributing! This guide explains how to set
up the development environment, run the tests, and submit a pull request.

---

## Table of contents

1. [Code of conduct](#code-of-conduct)
2. [Development environment](#development-environment)
3. [Running tests](#running-tests)
4. [Code style](#code-style)
5. [Pull request workflow](#pull-request-workflow)
6. [Commit messages](#commit-messages)
7. [Reporting issues](#reporting-issues)

---

## Code of conduct

Be respectful, inclusive, and constructive in all interactions.

---

## Development environment

### Prerequisites

| Tool | Minimum version |
|------|----------------|
| Python | 3.9 |
| Git | 2.30 |
| pip | 23.0 |

### Setup

```bash
git clone https://github.com/Sakshi2002-Sinha/vadar-visual-agent.git
cd vadar-visual-agent

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install the package in editable mode together with dev extras
pip install -e ".[dev]"

# (Optional) install the full set of inference dependencies
pip install -r requirements.txt
```

---

## Running tests

All tests live in the `tests/` directory and use **pytest**.

```bash
# Run all unit tests (no GPU, no API key required)
pytest tests/test_spatial_reasoner.py tests/test_code_generator.py -v

# Run integration tests (no GPU, no API key – uses mocks)
pytest tests/test_vadar_agent.py -v

# Run the full test suite with coverage
pytest --cov=vadar_agent --cov-report=term-missing

# Run the synthetic demo to verify the quickstart script
python quickstart.py --demo
```

Tests must pass before a pull request can be merged.

---

## Code style

- **Formatter**: The project follows PEP 8 conventions.  
  Run `python -m py_compile <file>` to catch syntax errors before committing.
- **Type hints**: All public functions and methods should have type annotations.
- **Docstrings**: Use the existing Google-style docstrings as a template.
- **Imports**: Standard library → third-party → local, each group separated by a blank line.
- **Line length**: 100 characters maximum.
- **No print statements** in library code (`vadar_agent.py`); use the `logging` module.

---

## Pull request workflow

1. **Fork** the repository and create a feature branch from `main`:

   ```bash
   git checkout -b feature/my-feature
   ```

2. Make your changes in **small, focused commits** (see [Commit messages](#commit-messages)).

3. Add or update tests to cover your changes.

4. Ensure all tests pass locally before opening the PR.

5. Push your branch and open a pull request against `main`.

6. Fill in the PR template (title, description, linked issues, test evidence).

7. Address any review comments and push follow-up commits to the same branch.

8. A maintainer will merge once the CI passes and the review is approved.

### Branch naming

| Prefix | Usage |
|--------|-------|
| `feature/` | New features |
| `fix/` | Bug fixes |
| `docs/` | Documentation changes |
| `refactor/` | Code refactoring |
| `test/` | Test-only changes |
| `ci/` | CI / tooling changes |

---

## Commit messages

Follow the [Conventional Commits](https://www.conventionalcommits.org/) specification:

```
<type>(<scope>): <short summary>

[optional body]

[optional footer]
```

Types: `feat`, `fix`, `docs`, `test`, `refactor`, `ci`, `chore`.

**Examples:**

```
feat(agent): add multi-turn follow-up question support
fix(reasoner): correct boundary check in vertical_position
docs: add benchmark results to README
test(code_generator): cover execution error path
```

---

## Reporting issues

Please use the [GitHub Issues](https://github.com/Sakshi2002-Sinha/vadar-visual-agent/issues)
tracker. Include:

- Python version and OS
- Steps to reproduce
- Expected vs. actual behaviour
- Relevant log output or stack trace
