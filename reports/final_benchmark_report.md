# OptiRouteAI Final Benchmark Report

## Overview

This report compares four solver approaches for the Capacitated Vehicle Routing Problem (CVRP):

1. **Nearest Neighbor** — fast heuristic baseline.
2. **Neural MLP** — pure neural imitation solver.
3. **Hybrid MLP alpha=0.5** — combines neural scoring with distance-based heuristic.
4. **OR-Tools** — classical optimization benchmark.

## Solver Comparison

| solver_name          |   num_runs |   avg_distance |   median_distance |   avg_runtime_seconds |   median_runtime_seconds |   feasibility_rate |   avg_gap_vs_ortools_percent |   median_gap_vs_ortools_percent |
|:---------------------|-----------:|---------------:|------------------:|----------------------:|-------------------------:|-------------------:|-----------------------------:|--------------------------------:|
| Nearest Neighbor     |         20 |         7.4165 |            7.5389 |                0.001  |                   0.0008 |                100 |                      28.7648 |                         28.1701 |
| Neural MLP           |         20 |         7.4538 |            7.2713 |                0.0918 |                   0.0857 |                100 |                      29.5222 |                         28.7828 |
| Hybrid MLP alpha=0.5 |         20 |         7.215  |            7.1415 |                0.0927 |                   0.0887 |                100 |                      25.2672 |                         22.4479 |
| OR-Tools             |         20 |         5.7784 |            5.8012 |                1.0039 |                   1.0033 |                100 |                       0      |                          0      |

## Main Result

The strongest non-OR-Tools solver is **Hybrid MLP alpha=0.5**.

- Nearest Neighbor average gap vs OR-Tools: **28.7648%**
- Hybrid MLP alpha=0.5 average gap vs OR-Tools: **25.2672%**
- Absolute gap improvement: **3.4976 percentage points**
- Relative improvement over Nearest Neighbor: **12.16%**

## Runtime Insight

The Hybrid MLP solver is substantially faster than OR-Tools.

- Hybrid MLP alpha=0.5 average runtime: **0.0927 seconds**
- OR-Tools average runtime: **1.0039 seconds**
- Approximate speedup vs OR-Tools: **10.83x**

## Interpretation

The pure Neural MLP solver is feasible, but the best performance comes from combining the neural model with a distance-based heuristic. This indicates that the learned model adds useful signal when combined with a classical routing heuristic.

The current result shows that the hybrid solver improves over the nearest-neighbor baseline while maintaining 100% feasibility across the benchmark instances.

## Current Limitation

The hybrid neural solver is still worse than OR-Tools in route quality. The next improvements should focus on:

- graph or attention-based route representation,
- better route-level objective training,
- reinforcement learning or ranking loss,
- larger benchmark set,
- more realistic routing constraints such as time windows.
