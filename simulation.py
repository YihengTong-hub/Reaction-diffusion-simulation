"""Visualize the two reaction-diffusion solvers from AMATH 481 Homework 6.

Run ``python simulation.py`` to generate figures, a GIF, and a run summary.
Author: Yiheng Tong. Based on AMATH 481 coursework.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from time import perf_counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import Normalize
import numpy as np
from scipy.fft import fft2, ifft2
from scipy.integrate import solve_ivp


@dataclass
class SimulationResult:
    method: str
    x: np.ndarray
    y: np.ndarray
    times: np.ndarray
    u: np.ndarray
    v: np.ndarray
    elapsed_seconds: float
    rhs_evaluations: int
    max_imaginary_component: float = 0.0


def initial_fields(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The original single-arm spiral initial condition, with winding number 1."""
    xx, yy = np.meshgrid(x, y)
    radius = np.hypot(xx, yy)
    phase = np.angle(xx + 1j * yy) - radius
    return np.tanh(radius) * np.cos(phase), np.tanh(radius) * np.sin(phase)


def reaction_terms(u: np.ndarray, v: np.ndarray, beta: float):
    """Lambda-omega kinetics: lambda=1-|A|^2 and omega=-beta*|A|^2."""
    amplitude_squared = np.abs(u) ** 2 + np.abs(v) ** 2
    growth = 1.0 - amplitude_squared
    rotation = -beta * amplitude_squared
    return growth * u - rotation * v, rotation * u + growth * v


def solve_fourier(
    grid_size: int = 64,
    duration: float = 4.0,
    frames: int = 41,
    diffusion: float = 0.1,
    beta: float = 1.0,
    length: float = 20.0,
    rtol: float = 1e-3,
    atol: float = 1e-6,
) -> SimulationResult:
    """Fourier pseudospectral discretization on a periodic square domain."""
    started = perf_counter()
    n = grid_size
    spacing = length / n
    x = np.linspace(-length / 2, length / 2, n, endpoint=False)
    u0, v0 = initial_fields(x, x)
    wave_numbers = 2 * np.pi * np.fft.fftfreq(n, d=spacing)
    kx, ky = np.meshgrid(wave_numbers, wave_numbers)
    laplacian = -(kx**2 + ky**2)
    size = n * n

    def rhs(_time, state):
        u_hat = state[:size].reshape(n, n)
        v_hat = state[size:].reshape(n, n)
        u, v = ifft2(u_hat), ifft2(v_hat)
        reaction_u, reaction_v = reaction_terms(u, v, beta)
        du_hat = fft2(reaction_u) + diffusion * laplacian * u_hat
        dv_hat = fft2(reaction_v) + diffusion * laplacian * v_hat
        return np.concatenate((du_hat.ravel(), dv_hat.ravel()))

    times = np.linspace(0, duration, frames)
    state0 = np.concatenate((fft2(u0).ravel(), fft2(v0).ravel()))
    solution = solve_ivp(
        rhs, (0, duration), state0, t_eval=times,
        method="RK45", rtol=rtol, atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"Fourier integration failed: {solution.message}")
    u = ifft2(solution.y[:size].T.reshape(-1, n, n), axes=(-2, -1))
    v = ifft2(solution.y[size:].T.reshape(-1, n, n), axes=(-2, -1))
    imaginary = max(float(np.max(np.abs(u.imag))), float(np.max(np.abs(v.imag))))
    if not (np.isfinite(u).all() and np.isfinite(v).all()):
        raise RuntimeError("Fourier integration produced non-finite values.")
    if imaginary > 1e-6:
        raise RuntimeError(f"Unexpected imaginary field component: {imaginary:.3g}")
    return SimulationResult(
        "Fourier", x, x.copy(), solution.t, u.real, v.real,
        perf_counter() - started, solution.nfev, imaginary,
    )


def chebyshev_matrix(degree: int) -> tuple[np.ndarray, np.ndarray]:
    """First derivative matrix on degree+1 Chebyshev-Lobatto nodes."""
    if degree < 1:
        raise ValueError("Chebyshev degree must be at least 1.")
    indices = np.arange(degree + 1)
    nodes = np.cos(np.pi * indices / degree)
    weights = np.ones(degree + 1)
    weights[[0, -1]] = 2.0
    weights *= (-1.0) ** indices
    differences = nodes[:, None] - nodes[None, :]
    derivative = (weights[:, None] / weights[None, :]) / (
        differences + np.eye(degree + 1)
    )
    derivative -= np.diag(derivative.sum(axis=1))
    return derivative, nodes


def solve_chebyshev(
    degree: int = 30,
    duration: float = 4.0,
    frames: int = 41,
    diffusion: float = 0.1,
    beta: float = 1.0,
    length: float = 20.0,
    rtol: float = 1e-3,
    atol: float = 1e-6,
) -> SimulationResult:
    """Preserve the original homework's modified Chebyshev boundary operator.

    The first/last rows of D are zeroed BEFORE squaring, as in the source.
    This discrete treatment is not presented as a verified continuum boundary
    condition; interpreting it requires the original assignment statement.
    """
    started = perf_counter()
    derivative, nodes = chebyshev_matrix(degree)
    derivative[[0, -1], :] = 0.0
    dxx = (derivative @ derivative) / (length / 2) ** 2
    x = nodes * length / 2
    u0, v0 = initial_fields(x, x)
    n = degree + 1
    size = n * n

    def rhs(_time, state):
        u = state[:size].reshape(n, n)
        v = state[size:].reshape(n, n)
        reaction_u, reaction_v = reaction_terms(u, v, beta)
        # Equivalent to the original Kronecker-sum matrix on C-order arrays.
        laplace_u = dxx @ u + u @ dxx.T
        laplace_v = dxx @ v + v @ dxx.T
        return np.concatenate((
            (reaction_u + diffusion * laplace_u).ravel(),
            (reaction_v + diffusion * laplace_v).ravel(),
        ))

    times = np.linspace(0, duration, frames)
    solution = solve_ivp(
        rhs, (0, duration), np.concatenate((u0.ravel(), v0.ravel())),
        t_eval=times, method="RK45", rtol=rtol, atol=atol,
    )
    if not solution.success:
        raise RuntimeError(f"Chebyshev integration failed: {solution.message}")
    if not np.isfinite(solution.y).all():
        raise RuntimeError("Chebyshev integration produced non-finite values.")
    return SimulationResult(
        "Chebyshev", x, x.copy(), solution.t,
        solution.y[:size].T.reshape(-1, n, n),
        solution.y[size:].T.reshape(-1, n, n),
        perf_counter() - started, solution.nfev,
    )


def save_snapshots(result: SimulationResult, path: Path):
    """Plot four U-field snapshots with a shared scale and physical coordinates."""
    indices = np.rint(np.linspace(0, len(result.times) - 1, 4)).astype(int)
    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.9), layout="constrained")
    norm = Normalize(vmin=-1.2, vmax=1.2)
    mesh = None
    for axis, index in zip(axes, indices):
        mesh = axis.pcolormesh(
            result.x, result.y, result.u[index],
            shading="nearest", cmap="RdBu_r", norm=norm, rasterized=True,
        )
        axis.set(title=f"t = {result.times[index]:.2f}", xlabel="x", aspect="equal")
        axis.set_xlim(-10, 10)
        axis.set_ylim(-10, 10)
    axes[0].set_ylabel("y")
    fig.colorbar(mesh, ax=axes, label="U", shrink=0.76, pad=0.02)
    boundary = "periodic domain" if result.method == "Fourier" else "coursework boundary operator"
    fig.suptitle(f"{result.method} reaction-diffusion simulation | {boundary}", fontsize=14)
    fig.savefig(path, dpi=160, facecolor="white")
    plt.close(fig)


def save_animation(result: SimulationResult, path: Path, fps: int = 12):
    """Animate the Fourier U field; color limits stay fixed throughout."""
    spacing = result.x[1] - result.x[0]
    half_length = (result.x[-1] - result.x[0] + spacing) / 2
    fig, axis = plt.subplots(figsize=(5.4, 5.1), layout="constrained")
    image = axis.imshow(
        result.u[0], origin="lower", extent=(-half_length, half_length, -half_length, half_length),
        cmap="RdBu_r", vmin=-1.2, vmax=1.2, interpolation="nearest",
    )
    axis.set(xlabel="x", ylabel="y")
    title = axis.set_title("Fourier reaction-diffusion | t = 0.00")
    fig.colorbar(image, ax=axis, label="U", shrink=0.83)

    def update(index):
        image.set_data(result.u[index])
        title.set_text(f"Fourier reaction-diffusion | t = {result.times[index]:.2f}")
        return image, title

    animation = FuncAnimation(fig, update, frames=len(result.times), interval=1000 / fps)
    animation.save(path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close(fig)


def summarize(result: SimulationResult) -> dict:
    return {
        "method": result.method,
        "grid_shape": [len(result.y), len(result.x)],
        "saved_frames": len(result.times),
        "elapsed_seconds": round(result.elapsed_seconds, 6),
        "rhs_evaluations": int(result.rhs_evaluations),
        "finite_fields": bool(np.isfinite(result.u).all() and np.isfinite(result.v).all()),
        "max_imaginary_component": result.max_imaginary_component,
        "final_u_range": [float(result.u[-1].min()), float(result.u[-1].max())],
        "final_v_range": [float(result.v[-1].min()), float(result.v[-1].max())],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("both", "fourier", "chebyshev"), default="both")
    parser.add_argument("--grid", type=int, default=64, help="Fourier grid points per axis")
    parser.add_argument("--degree", type=int, default=30, help="Chebyshev polynomial degree")
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--frames", type=int, default=41)
    parser.add_argument("--diffusion", type=float, default=0.1, help="Equal diffusion coefficient for U and V")
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--skip-gif", action="store_true")
    parser.add_argument("--save-data", action="store_true", help="Save spatial fields as compressed NPZ files")
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parent / "assets")
    args = parser.parse_args()
    if args.grid < 4 or args.degree < 2 or args.frames < 4:
        parser.error("Use --grid >= 4, --degree >= 2, and --frames >= 4.")
    if args.duration <= 0 or args.diffusion < 0 or args.rtol <= 0 or args.atol <= 0:
        parser.error("Duration and tolerances must be positive; diffusion must be nonnegative.")
    args.output.mkdir(parents=True, exist_ok=True)
    parameters = dict(
        duration=args.duration, frames=args.frames, diffusion=args.diffusion,
        beta=args.beta, rtol=args.rtol, atol=args.atol,
    )
    summaries = []
    methods = ("fourier", "chebyshev") if args.method == "both" else (args.method,)
    for method in methods:
        if method == "fourier":
            result = solve_fourier(grid_size=args.grid, **parameters)
        else:
            result = solve_chebyshev(degree=args.degree, **parameters)
        save_snapshots(result, args.output / f"{method}_snapshots.png")
        if method == "fourier" and not args.skip_gif:
            save_animation(result, args.output / "spiral.gif")
        if args.save_data:
            np.savez_compressed(
                args.output / f"{method}_fields.npz",
                x=result.x, y=result.y, t=result.times, u=result.u, v=result.v,
            )
        summaries.append(summarize(result))
        print(f"{result.method}: {result.u.shape}, solver {result.elapsed_seconds:.2f}s, "
              f"{result.rhs_evaluations} RHS evaluations")
    report = {"parameters": {**parameters, "domain_length": 20.0}, "runs": summaries}
    (args.output / "run_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Figures and summary saved to: {args.output.resolve()}")


if __name__ == "__main__":
    main()
