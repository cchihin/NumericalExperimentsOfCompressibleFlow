# Leblanc Shock Tube — Parameter Discovery Study

## Problem

1D Leblanc shock tube solved with Nektar++ DG (`CompressibleFlowSolver`).

| Parameter | Value |
|-----------|-------|
| Domain | x ∈ [0, 10], shock initially at x = 3 |
| γ | 5/3 |
| ρ_L, u_L, p_L | 1.0,  0.0,  (γ-1)·0.1 |
| ρ_R, u_R, p_R | 1e-3, 0.0,  (γ-1)·1e-10 |
| Method | DG (WeakDG), `GLL_LAGRANGE`, Lax-Friedrichs flux |
| Shock capture | NonSmooth (Persson-Peraire), μ₀=1, S_κ=-1.3, κ=0.2 |
| Final time | t = 3.0 (all wave structures safely inside [0,10]) |

**Wave positions at t = 3.0** (all inside domain — safe final time):

| Feature | Position |
|---------|----------|
| Rarefaction head | x = 2.00 |
| Rarefaction tail | x = 4.49 |
| Contact discontinuity | x = 4.87 |
| Shock | x = 5.49 |

> **Domain note**: the shock exits x=10 at t≈8.44, rarefaction head exits x=0 at t≈9.0.
> Do not run beyond t≈8.0 without extending the domain or using outflow BCs.

---

## Files

| File | Purpose |
|------|---------|
| `Leblanc1DSession.xml` | Nektar++ session template (physics + solver, no geometry) |
| `LeblancSolution.py` | Exact Riemann solver for Leblanc IC (analytical reference) |
| `run.py` | Original run script (Nel=240, P=2) |
| `benchmark.py` | **Phase 1**: time-scheme × dt stability/accuracy sweep |
| `benchmark2.py` | **Phase 2**: Nel × P sweep with CFL-adaptive dt |
| `Benchmark/` | Phase 1 output (run dirs + `results.json`) |
| `Benchmark2/` | Phase 2 output (run dirs + `results2.json`) |

**Solver paths** (compiled Nektar++ build):
```
/home/chihin/Repositories/nektar-cch/build/solvers/CompressibleFlowSolver/CompressibleFlowSolver
/home/chihin/Repositories/nektar-cch/build/utilities/FieldConvert/FieldConvert
```

---

## How to Re-run

```bash
cd /home/chihin/NumericalExperimentsOfCompressibleFlow/LeblancShocktube-claude/Test

# Phase 1 — time scheme × dt sweep (Nel=160, P=2)  ~15 min
python3 benchmark.py

# Phase 2 — Nel × P sweep with CFL-adaptive dt      ~15 min
python3 benchmark2.py
```

Results are written to `Benchmark/results.json` and `Benchmark2/results2.json`.
A human-readable log is also written alongside each JSON.

---

## Findings

### Phase 1 — Time integration scheme (Nel=160, P=2, NonSmooth)

All four schemes are stable at dt=1e-3 through dt=1e-5, with virtually identical L2(ρ)≈2.93e-2.

| Scheme | dt_max tested | L2(ρ) | Wall (t=3) |
|--------|--------------|-------|------------|
| Forward Euler | 1e-3 | 2.924e-2 | 1s |
| **SSP-RK2** | **1e-3** | **2.928e-2** | **3s** |
| SSP-RK3 | 1e-3 | 2.927e-2 | 4s |
| RK4 | 1e-3 | 2.926e-2 | 7s |

**Conclusion**: temporal error is negligible — the spatial discretization dominates.
**Recommended scheme**: **SSP-RK2** (`RungeKutta ORDER=2 VARIANT=SSP`). It is strong-stability-preserving (important for shock problems), 2nd-order accurate, and only 2× the cost of Forward Euler.

**CFL rule** (derived from sweep):
```
dt = 0.04 * h / (2*P + 1)    where h = 10 / Nel
```
This gives dt ≈ 5e-4 for the reference case (Nel=160, P=2).

---

### Phase 2 — Nel × P sweep (SSP-RK2, NonSmooth, CFL-adaptive dt)

**Stability notes:**
- **P=1**: always crashes. The NonSmooth (Persson-Peraire) sensor requires at least 2 polynomial modes — do not use P=1.
- **Nel=80, P=2/3**: unstable or produces NaN fields. Nel=80 is too coarse for this problem.
- **Nel≥160, P∈{2,3,4}**: all stable with the CFL-adaptive dt formula above.

| Nel | P | DoF  | dt      | L2(ρ)      | Wall  |
|-----|---|------|---------|------------|-------|
| 80  | 4 | 400  | 5.6e-4  | 2.993e-02  | 3s    |
| **160** | **2** | **480** | **5.0e-4** | **2.925e-02** | **5s** |
| 160 | 3 | 640  | 3.6e-4  | 2.928e-02  | 8s    |
| 160 | 4 | 800  | 2.8e-4  | 2.944e-02  | 13s   |
| 320 | 2 | 960  | 2.5e-4  | 2.926e-02  | 24s   |
| 320 | 3 | 1280 | 1.8e-4  | 2.926e-02  | 38s   |
| 320 | 4 | 1600 | 1.4e-4  | 2.923e-02  | 49s   |
| 640 | 2 | 1920 | 1.25e-4 | 2.926e-02  | 95s   |
| 640 | 3 | 2560 | 8.9e-5  | 2.926e-02  | 135s  |
| 640 | 4 | 3200 | 6.9e-5  | 2.926e-02  | 184s  |

**Recommended config**: **Nel=160, P=2** — cheapest stable run (480 DoF, 5s), same L2 error as every other configuration.

---

### Key Finding — Error Saturation

**All configurations give L2(ρ) ≈ 2.93e-2 (≈5.8% relative error), regardless of Nel or P.**
Refinement does not reduce the error.

**Root cause**: the simulation initial condition is a *smooth* tanh profile (width=0.5):
```xml
<E VAR="rho" VALUE="rhoL + 0.5*(1+tanh((x-3)/0.5)) * (rhoR-rhoL)" />
```
but `LeblancSolution.py` computes the exact solution from a *perfect step function* at x=3.
This IC mismatch creates an irreducible ~2.93e-2 error floor that no amount of mesh or polynomial refinement can close.

**To observe DG convergence**, reduce the tanh transition width in `Leblanc1DSession.xml`:
```xml
<!-- change 0.5 → 0.01 (or smaller) -->
<E VAR="rho" VALUE="rhoL + 0.5*(1+tanh((x-3)/0.01)) * (rhoR-rhoL)" />
<E VAR="E"   VALUE="EL  + 0.5*(1+tanh((x-3)/0.01)) * (ER-EL)"      />
```
A narrower tanh requires finer elements near x=3 to represent it without oscillations — check that Nel ≥ ~10/tanh_width.

---

## Recommended Next Steps

1. **Fix IC**: narrow the tanh width to ~0.01 and re-run the Nel × P convergence study.
2. **Sweep shock-capture parameters**: vary μ₀ (0.5, 1, 2) and S_κ (-1.0, -1.3, -1.5) to see how they affect accuracy and stability.
3. **Compare ShockCaptureType**: test `NonSmooth` vs other available capture modes.
4. **Extend domain time study**: once the IC is fixed, push to t=6 to see the evolved shock structure more clearly (still inside domain; shock exits at t≈8.44).
