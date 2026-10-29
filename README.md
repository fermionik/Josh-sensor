# Fig. 10 reproduction: single-node cavity-QED RNN trained by BPTT

`fig10_bptt.py` reproduces Fig. 10 of *A single-photon quantum sensor via single-atom cavity-QED* (Burns, Keller, Milburn). It trains the Raman control pulse α(t) of a trapped-atom cavity detector so that the binary readout "atom found in |g⟩" extracts as much Fisher information as possible about the detuning δ of a remote one-sided cavity, and compares this with the quantum Fisher information (QFI) of the single-photon probe.

## Requirements

Python 3.10+ with

```bash
pip install numpy scipy matplotlib jax cma
```

`cma` is optional; without it the gradient-free baseline is skipped.

## Usage

```bash
python fig10_bptt.py
```

Runtime is about 10 s on a laptop CPU. Expected console output:

```text
Fock QFI at sigma_i=1.86, delta_0=0.53: F_Q = 4.622
Gaussian control (energy 1.0): t_a=0.79, sigma_a=1.70, F = 2.952
RBF fit of Gaussian: F = 2.952
BPTT: F = 4.400 (95.2% of QFI) in 80 iters, 94 fwd+adjoint passes
CMA-ES: F = 4.390 after 1703 evaluations
p_g = 0.2925,  p_g + int|xi_out|^2 = 0.9999
```

## Outputs

| File | Contents |
|---|---|
| `fig10_reproduced.pdf` | Vector figure for the manuscript |
| `fig10_reproduced.png` | 200 dpi raster preview |
| `fig10_data.npz` | Training history, initial and optimised weights, F_Q, Gaussian and CMA-ES results, node times |

Left panel: readout Fisher information F_δ against L-BFGS iteration, with the QFI ceiling (dashed) and the CMA-ES result (dotted). Right panel: learned |α(t)| (solid), the 12 Gaussian node values (open circles) and the phase arg α (dotted, right axis).

## Model

All rates are in units of κ = γ = 1. The atom starts in |e⟩ with the atom cavity empty; one photon with Gaussian temporal mode

ξ_i(t) = (πσ_i²)^(−1/4) exp(−t²/2σ_i²)

is reflected from the remote cavity (Eq. 48 of the paper):

```text
dy/dt   = −(γ/2 + iδ_r) y + γ ξ_i(t)
ξ_o(t)  = −ξ_i(t) + y(t)
```

and then drives the atom cavity. In the single-excitation sector, with H = (δ_a/2)σ_z + α a†σ₊ + α* σ₋a, the exact amplitude equations are

```text
dc_e1/dt = −(κ/2) c_e1 − i α(t) c_g0 + √κ ξ_o(t)
dc_g0/dt =  i δ_a c_g0 − i α*(t) c_e1
```

These are equivalent to the Fock-state master-equation hierarchy (Eqs. 28–38) for a one-photon input. The readout probability is p_g = |c_g0(T)|², and the classical Fisher information of the binary outcome is

```text
F_δ = (∂p_g/∂δ_r)² / [p_g(1 − p_g)].
```

The ceiling is the Fock-state QFI (Eqs. 50–53),

```text
F_Q = 4 Var_{|ξ̃(ω)|²}[∂_δ φ],  ∂_δ φ = −4γ / [γ² + 4(ω − δ)²].
```

Probability is checked through p_g + ∫|ξ_out|² dt = 1, with ξ_out = ξ_o − √κ c_e1.

## Control parametrisation

α(t) = Σ_k c_k exp[−½((t − t_k)/w)²], with K = 12 complex coefficients (24 real parameters), nodes t_k evenly spaced on [−3, 9], and w = 1.3/γ. The control energy ∫|α|² dt is fixed to 1 at every evaluation by c → c/√(c†Gc), using the exact Gram matrix G_jk = √π w exp[−(t_j − t_k)²/4w²].

## Optimisation

1. **Baseline.** The best single real Gaussian control of the same energy, found by Nelder–Mead over centre and width.
2. **BPTT.** Initialised from a least-squares fit of the baseline onto the nodes, then optimised by L-BFGS-B for 80 iterations. Gradients come from JAX reverse-mode differentiation through the unrolled RK4 solver, which is the adjoint method (backpropagation through time). ∂p_g/∂δ_r inside F_δ is computed by forward-mode differentiation.
3. **CMA-ES.** Gradient-free comparison from the same start, budget of about 1700 evaluations.

## Parameters

Set at the top of the script.

| Name | Value | Meaning |
|---|---|---|
| `KAPPA` | 1.0 | Atom-cavity decay rate (rate unit) |
| `GAMMA` | 1.0 | Remote-cavity linewidth |
| `DELTA_A` | 0.0 | Atomic (Zeeman) detuning |
| `SIGMA_I` | 1.86 | Input photon width (Table 1, optimised row) |
| `DELTA_0` | 0.53 | Remote-cavity operating detuning (Table 1) |
| `K`, `W` | 12, 1.3 | Number of Gaussian nodes, node width |
| `ENERGY` | 1.0 | Fixed control energy |
| `T0, T1, DT` | −8, 16, 0.01 | Integration window and RK4 step |

With these values F_Q = 4.62, matching Table 1.

## Results

| Readout | F_δ | Fraction of QFI |
|---|---|---|
| Single-Gaussian control | 2.95 | 64% |
| CMA-ES (1703 evaluations) | 4.39 | 95% |
| BPTT, 80 iterations (94 forward + adjoint passes) | 4.40 | 95.2% |
| QFI ceiling | 4.62 | 100% |

The Gaussian baseline independently reproduces the Fock entry of Table 2 (2.96, 64%). BPTT and CMA-ES converge to the same optimum; the advantage of BPTT is that it needs about 18 times fewer evaluations. The remaining ~5% of the QFI leaves in the reflected field (the error channel) and could in principle be recovered by a joint readout of atom and output field.

## Notes and assumptions

- The atom is treated as a two-level Raman system with no spontaneous decay (γ_atom = 0), consistent with the coherence-limited regime of Section 6.
- No preparation or detector errors are included; see Section 4 for the bit-flip models.
- Sign conventions for the input–output coupling do not affect p_g or F_δ.
