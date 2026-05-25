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