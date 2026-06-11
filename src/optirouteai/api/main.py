import math
from functools import lru_cache
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from optirouteai.data.generator import generate_cvrp_instance
from optirouteai.evaluation.benchmark import (
    compute_gap_percent,
    solve_with_nearest_neighbor,
)
from optirouteai.optimization.hybrid_solver import HybridNeuralSolver
from optirouteai.optimization.neural_solver import MLPNeuralSolver
from optirouteai.optimization.ortools_solver import (
    SolverResult,
    solve_cvrp_with_ortools,
)


app = FastAPI(
    title="OptiRouteAI API",
    description=(
        "FastAPI service for solving Capacitated Vehicle Routing Problem "
        "instances using heuristic, OR-Tools, neural MLP, and hybrid solvers."
    ),
    version="0.1.0",
)


class GenerateInstanceRequest(BaseModel):
    num_customers: int = Field(default=25, ge=1, le=200)
    num_vehicles: int = Field(default=5, ge=1, le=50)
    vehicle_capacity: int = Field(default=45, ge=1)
    demand_min: int = Field(default=1, ge=1)
    demand_max: int = Field(default=8, ge=1)
    seed: int = Field(default=42)


class SolveRequest(GenerateInstanceRequest):
    solver: Literal[
        "nearest_neighbor",
        "ortools",
        "neural_mlp",
        "hybrid_mlp",
    ] = "hybrid_mlp"

    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Hybrid solver weight. 0 = distance only, 1 = pure MLP.",
    )

    ortools_time_limit_seconds: int = Field(default=1, ge=1, le=30)


class RouteSolutionResponse(BaseModel):
    solver_name: str
    routes: list[list[int]]
    total_distance: float
    runtime_seconds: float
    is_feasible: bool


class SolveResponse(BaseModel):
    request: SolveRequest
    solution: RouteSolutionResponse


class CompareSolversResponse(BaseModel):
    request: GenerateInstanceRequest
    results: list[dict]


def solver_result_to_response(result: SolverResult) -> RouteSolutionResponse:
    return RouteSolutionResponse(
        solver_name=result.solver_name,
        routes=result.routes,
        total_distance=float(result.total_distance),
        runtime_seconds=float(result.runtime_seconds),
        is_feasible=bool(result.is_feasible),
    )


def create_instance(request: GenerateInstanceRequest):
    if request.demand_max < request.demand_min:
        raise HTTPException(
            status_code=400,
            detail="demand_max must be greater than or equal to demand_min.",
        )

    return generate_cvrp_instance(
        num_customers=request.num_customers,
        num_vehicles=request.num_vehicles,
        vehicle_capacity=request.vehicle_capacity,
        demand_min=request.demand_min,
        demand_max=request.demand_max,
        seed=request.seed,
    )


@lru_cache(maxsize=1)
def get_neural_solver() -> MLPNeuralSolver:
    return MLPNeuralSolver(
        checkpoint_path="models/mlp_policy_step9.pkl",
    )


@lru_cache(maxsize=10)
def get_hybrid_solver(alpha: float) -> HybridNeuralSolver:
    return HybridNeuralSolver(
        checkpoint_path="models/mlp_policy_step9.pkl",
        alpha=alpha,
    )


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "OptiRouteAI API",
        "version": "0.1.0",
    }


@app.post("/solve", response_model=SolveResponse)
def solve(request: SolveRequest):
    instance = create_instance(request)

    if request.solver == "nearest_neighbor":
        result = solve_with_nearest_neighbor(instance)

    elif request.solver == "ortools":
        result = solve_cvrp_with_ortools(
            instance=instance,
            time_limit_seconds=request.ortools_time_limit_seconds,
        )

    elif request.solver == "neural_mlp":
        try:
            solver = get_neural_solver()
            result = solver.solve(instance)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail=str(exc),
            ) from exc

    elif request.solver == "hybrid_mlp":
        try:
            solver = get_hybrid_solver(request.alpha)
            result = solver.solve(instance)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail=str(exc),
            ) from exc

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported solver: {request.solver}",
        )

    return SolveResponse(
        request=request,
        solution=solver_result_to_response(result),
    )


@app.post("/compare", response_model=CompareSolversResponse)
def compare_solvers(request: GenerateInstanceRequest):
    instance = create_instance(request)

    nn_result = solve_with_nearest_neighbor(instance)

    ortools_result = solve_cvrp_with_ortools(
        instance=instance,
        time_limit_seconds=1,
    )

    solver_results = [
        nn_result,
        ortools_result,
    ]

    try:
        neural_result = get_neural_solver().solve(instance)
        hybrid_result = get_hybrid_solver(0.5).solve(instance)

        solver_results.extend(
            [
                neural_result,
                hybrid_result,
            ]
        )
    except FileNotFoundError:
        # API still works even if neural checkpoint is missing.
        pass

    reference_distance = None
    if (
        ortools_result.is_feasible
        and math.isfinite(ortools_result.total_distance)
        and ortools_result.total_distance > 0
    ):
        reference_distance = ortools_result.total_distance

    results = []

    for result in solver_results:
        if reference_distance is None:
            gap = None
        else:
            gap = compute_gap_percent(
                solver_distance=result.total_distance,
                reference_distance=reference_distance,
            )

        results.append(
            {
                "solver_name": result.solver_name,
                "routes": result.routes,
                "total_distance": float(result.total_distance),
                "runtime_seconds": float(result.runtime_seconds),
                "is_feasible": bool(result.is_feasible),
                "gap_vs_ortools_percent": None if gap is None else float(gap),
            }
        )

    return CompareSolversResponse(
        request=request,
        results=results,
    )