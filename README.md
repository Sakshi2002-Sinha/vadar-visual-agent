---
title: VADAR
emoji: 🧊
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: 4.x
app_file: app.py
pinned: true
---

# VADAR — Visual Agent for 3D Autonomous Reasoning

VADAR is an agentic program synthesis system for natural-language 3D spatial queries.

## Live demo
[Space URL here — fill in after deploy]

## Architecture
![Architecture Diagram](docs/architecture.png)

Query → FastAPI → Agent Loop → LLM (Anthropic) → Program Synthesis → Execution Engine → Result → Database + MLflow logging → Gradio UI

## Example queries

### 1) Query
How many objects are in the scene?

**Synthesised program**
```python
result = len(get_all_objects(scene_id))
```

**Result**
```json
8
```

### 2) Query
Find the object closest to the origin

**Synthesised program**
```python
objects = get_all_objects(scene_id)
ordered = sort_by_distance(objects, (0.0, 0.0, 0.0))
result = ordered[0].name if ordered else None
```

**Result**
```json
"main table"
```

### 3) Query
Are any objects directly above the table?

**Synthesised program**
```python
table = get_object_by_name('table', scene_id)
objects = get_all_objects(scene_id)
result = [o.name for o in objects if o.name != table.name and is_above(o, table)]
```

**Result**
```json
["yellow sphere", "orange lamp"]
```

### 4) Query
What is the largest object by volume?

**Synthesised program**
```python
objects = get_all_objects(scene_id)
result = max(objects, key=lambda o: o.volume).name
```

**Result**
```json
"gray cabinet"
```

### 5) Query
List all red objects sorted by distance to [0,0,0]

**Synthesised program**
```python
objects = filter_by_color('red', scene_id)
ordered = sort_by_distance(objects, (0.0, 0.0, 0.0))
result = [o.name for o in ordered]
```

**Result**
```json
["red chair"]
```

## Evaluation
| # | Query | Category | Pass | Latency(ms) | Failure Reason |
|---|-------|----------|------|-------------|----------------|
| 1 | How many objects are in the scene? | counting | True | 11.73 |  |
| 2 | What is the name of the largest object? | comparison | True | 10.11 |  |
| 3 | Find all objects within 1 metre of the origin | proximity | True | 9.98 |  |
| 4 | Is there any object above height 2.0? | threshold | True | 9.82 |  |
| 5 | What is the closest object to position [1,0,1]? | proximity | True | 10.91 |  |
| 6 | Find all objects that are red | attribute | True | 11.12 |  |
| 7 | What is the average height of all objects? | aggregation | True | 10.74 |  |
| 8 | Are any two objects overlapping? | overlap | True | 10.78 |  |
| 9 | Find the object furthest from the origin | proximity | True | 10.39 |  |
| 10 | List objects sorted by distance to [0,0,0] | sorting | True | 10.60 |  |
| 11 | How many objects are taller than 1.5 units? | comparison | True | 10.98 |  |
| 12 | What colour is the object at position [2,0,2]? | attribute | True | 10.88 |  |
| 13 | Find all objects directly above the floor (y < 0.1) | containment | True | 11.64 |  |
| 14 | What is the total volume of all objects combined? | aggregation | True | 11.46 |  |
| 15 | Is the blue object to the left of the red object? | relative | True | 11.81 |  |
| 16 | Find objects that are not touching any other object | negation | True | 11.70 |  |
| 17 | What is the bounding box of the entire scene? | geometry | True | 12.35 |  |
| 18 | Find all pairs of objects within 0.5 metres of each other | proximity | True | 11.82 |  |
| 19 | Which object is most central in the scene? | comparison | True | 12.44 |  |
| 20 | Describe the spatial layout of the scene in one sentence | description | True | 12.43 |  |

**Overall accuracy: 100.00%  |  Avg latency: 11.18ms  |  Failure rate: 0.00%**

## Failure handling
- MAX_ITERATIONS is enforced (default 5; configurable via `MAX_ITERATIONS`).
- If the iteration limit is reached, VADAR returns:
  - `success=false`
  - `failure_reason="max_iterations_exceeded"`
  - `message="I cannot solve this query"`
  - `partial_program` from the latest attempt.
- LLM synthesis and execution are wrapped in guarded error handling.
- Program execution is restricted and timeout-limited to 10s.

## MLOps
- MLflow experiment name: `vadar-agent-runs`
- Tracking URI from `MLFLOW_TRACKING_URI` (default `./mlruns`)
- Logged per run:
  - params: query (truncated), synthesized program hash, result preview
  - metrics: latency_ms, iterations, success
  - tags: scene_id, failure_reason (when present)

To inspect runs locally:
```bash
mlflow ui
```
