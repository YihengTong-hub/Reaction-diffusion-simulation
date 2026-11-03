# Reaction-Diffusion Simulation

Two numerical implementations of a two-dimensional lambda-omega reaction-diffusion system, adapted from **AMATH 481 Homework 6** at the University of Washington.

**Author:** Yiheng Tong  
**Tools:** Python, NumPy, SciPy, Matplotlib

![Evolution of the U field using the Fourier method](assets/spiral.gif)

The simulation starts with a single-arm spiral initial condition and follows the evolution of two coupled fields, **U** and **V**. The repository includes a Fourier pseudospectral solver, a Chebyshev collocation solver, and generated visualizations.

## What this project demonstrates

- Building a spatial discretization and integrating the resulting coupled ODE system.
- Computing periodic spatial derivatives through two-dimensional FFTs.
- Constructing Chebyshev differentiation matrices on a nonuniform grid.
- Applying a two-dimensional separable Laplacian without forming a full Kronecker matrix.
- Turning numerical solutions into reproducible figures and an animation.

## Model

The coupled equations are

$$
\frac{\partial U}{\partial t}
= D_1\nabla^2 U + \lambda(A)U - \omega(A)V,
$$

$$
\frac{\partial V}{\partial t}
= D_2\nabla^2 V + \omega(A)U + \lambda(A)V,
$$

where

$$
A^2=U^2+V^2,\qquad \lambda(A)=1-A^2,\qquad \omega(A)=-\beta A^2.
$$

The default configuration uses equal diffusion coefficients, $D_1=D_2=0.1$, and $\beta=1$. The spatial domain is $[-10,10]\times[-10,10]$ and the simulated time interval is $0\le t\le4$.

The initial condition is

$$
U(x,y,0)=\tanh(r)\cos(\theta-r),\qquad
V(x,y,0)=\tanh(r)\sin(\theta-r),
$$

Here, $r=\sqrt{x^2+y^2}$ and the angle theta is computed as `atan2(y, x)`.

## Numerical methods

| Implementation | Spatial grid | Spatial operator | Time integration |
| --- | --- | --- | --- |
| Fourier | 64 by 64 equally spaced samples; repeated periodic endpoint excluded | FFT Laplacian; nonlinear reaction evaluated in physical space | SciPy RK45 |
| Chebyshev | 31 by 31 Chebyshev-Lobatto nodes | Modified differentiation matrix from the original coursework | SciPy RK45 |

The Fourier method treats the domain as periodic. Its wave numbers and Laplacian are computed once and reused throughout integration.

The Chebyshev implementation preserves the original script's boundary construction: the first and last rows of the first derivative matrix are set to zero **before** the matrix is squared. The exact continuum boundary condition has not been verified against the assignment statement. This repository documents the discrete operator rather than assigning it an unverified physical interpretation.

For a field matrix $F$, the two-dimensional Chebyshev Laplacian is applied as $D_{xx}F+FD_{xx}^{T}$. This is algebraically equivalent to the original Kronecker-sum operator while avoiding construction of a 961 by 961 matrix.

The two solvers use different grids and boundary treatments. Their plots are separate demonstrations; they are not a controlled accuracy comparison.

## Results

**Fourier method — U field**

![Fourier snapshots at four simulation times](assets/fourier_snapshots.png)

**Chebyshev method — U field**

![Chebyshev snapshots at four simulation times](assets/chebyshev_snapshots.png)

All plots use the same color limits, from -1.2 to 1.2. Both U and V are computed; the figures display U. The GIF shows 41 saved states over simulation time 0 to 4, played at 12 frames per second.

## Run locally

Requires **Python 3.10 or newer**. Open a terminal in this project folder.

On Windows:

```powershell
py -m pip install -r requirements.txt
py simulation.py
```

On macOS or Linux:

```bash
python3 -m pip install -r requirements.txt
python3 simulation.py
```

The default command runs both solvers and regenerates the figures, GIF, and JSON run summary in `assets/`. No external dataset or FFmpeg installation is needed.

### Options

Use `python` or `python3` instead of `py` if that is your Python command.

```powershell
# Run only the Fourier solver.
py simulation.py --method fourier

# Change the model's rotation parameter.
py simulation.py --method fourier --beta 0.5

# Extend the simulation and save more animation frames.
py simulation.py --method fourier --duration 10 --frames 81

# Generate snapshots without generating the GIF.
py simulation.py --skip-gif

# Save spatial U/V arrays for further analysis.
py simulation.py --save-data

# Show all available parameters.
py simulation.py --help
```

`--grid` controls the number of Fourier grid points per axis; `--degree` controls the Chebyshev degree, giving degree + 1 points per axis. `--rtol` and `--atol` set the time integrator tolerances. Default tolerances, 1e-3 and 1e-6 respectively, match the original script.

## Files

| File | Purpose |
| --- | --- |
| `simulation.py` | Solvers, command-line options, and visualization functions |
| `requirements.txt` | Python dependencies |
| `spiral.gif` | Fourier animation shown at the top of this page |
| `fourier_snapshots.png` | Four Fourier U-field snapshots |
| `chebyshev_snapshots.png` | Four Chebyshev U-field snapshots |
| `assets/run_summary.json` | Parameters, grid sizes, run times, and basic solver diagnostics |

## Validation and scope

The program checks successful time integration and finite solution values. It also checks that imaginary components of the Fourier solution remain small before exporting real-valued fields.

The packaged default run is compared against the original coursework script at its nine output times. Details of that check are recorded in `assets/validation.json`. These checks establish agreement with the source implementation, not independent validation of the model or spatial convergence.

Recorded solver run times exclude plotting and GIF creation and vary with hardware. They should not be interpreted as a fair performance comparison between the two methods.

## Coursework origin

This project is based on an AMATH 481 homework script. The model, initial conditions, and two numerical approaches originate in that coursework. The portfolio version reorganizes the script into reusable functions, precomputes Fourier operators, avoids the full Chebyshev Kronecker matrix, and adds visualizations, run instructions, and numerical diagnostics.

The Fourier implementation follows the original pseudospectral scheme without dealiasing. A spatial convergence study and confirmation of the Chebyshev boundary specification remain possible extensions.
