# OptiRouteAI

## Overview
OptiRouteAI is a neural combinatorial optimization project for solving the Capacitated Vehicle Routing Problem (CVRP). It combines heuristic search, OR-Tools, imitation learning, PyTorch MLP models, hybrid neural-distance solvers, FastAPI, Streamlit, and Docker.

## Key Result
The best hybrid neural solver improved over the nearest-neighbor baseline.

- Nearest Neighbor average gap vs OR-Tools: 28.76%
- Best Hybrid MLP alpha=0.5 average gap vs OR-Tools: 25.27%
- Relative improvement over Nearest Neighbor: ~12.16%
- Feasibility rate: 100%

## System Architecture
Synthetic CVRP Generator
→ Distance + Feasibility Engine
→ Nearest Neighbor / OR-Tools Solvers
→ OR-Tools Teacher Solutions
→ Imitation Learning Dataset
→ Logistic Regression + PyTorch MLP
→ Neural MLP Solver + Hybrid Solver
→ Benchmark Reports
→ FastAPI API + Streamlit Dashboard
→ Docker Deployment

## Solvers Implemented
- Capacity-aware Nearest Neighbor
- OR-Tools CVRP Solver
- Logistic Regression Imitation Model
- PyTorch MLP Imitation Model
- Neural MLP Solver
- Hybrid MLP + Distance Solver

## How to Run Locally

### FastAPI
```powershell
$env:PYTHONPATH="src"
uvicorn optirouteai.api.main:app --reload