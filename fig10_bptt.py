"""
Reproduce Fig. 10 of "A single-photon quantum sensor via single-atom cavity-QED".

Single-node cavity-QED "RNN": a single-photon pulse is reflected from a remote
one-sided cavity (linewidth gamma, detuning delta_r), then absorbed by a
Raman-controlled atom in a cavity (decay kappa). The readout is the binary
outcome "atom found in |g>", with probability p_g. The Raman control alpha(t)
is a sum of K complex Gaussians and is trained by back-propagation through
time (reverse-mode autodiff of the unrolled ODE solver = adjoint method) to
maximise the classical Fisher information F_delta of that binary readout.

Parameters are chosen to match Table 1 (optimised row): sigma_i = 1.86/gamma,
operating point delta_0 = 0.53 gamma, kappa = gamma = 1, for which the Fock
quantum Fisher information is F_Q = 4 Var[d_delta phi] = 4.62.

Dynamics (single-excitation, exact for a one-photon Fock input). Atom starts
in |e>, cavity empty. In a frame rotating with the |e> energy and the photon
carrier, with H = (delta_a/2) sigma_z + alpha a^dag sigma_+ + alpha^* sigma_- a:
    remote cavity :  dy/dt   = -(gamma/2 + i delta_r) y + gamma xi_i(t)
                     xi_o(t) = -xi_i(t) + y(t)                    (Eq. 48)
    atom cavity   :  dc_e1/dt = -(kappa/2) c_e1 - i alpha(t) c_g0 + sqrt(kappa) xi_o(t)
                     dc_g0/dt =  i delta_a c_g0 - i alpha^*(t) c_e1
    p_g = |c_g0(T)|^2,   F = (dp_g/d delta_r)^2 / [p_g (1 - p_g)]
The amplitude equations are equivalent to the Fock-state master-equation
hierarchy (Eqs. 28-38) in the single-excitation sector, and conserve
|c_g0|^2 + int |xi_out|^2 dt = 1 with xi_out = xi_o - sqrt(kappa) c_e1.

Requires: numpy, scipy, matplotlib, jax, cma
"""
import time
import numpy as np
import jax
import jax.numpy as jnp
from scipy.optimize import minimize
import matplotlib.pyplot as plt

jax.config.update("jax_enable_x64", True)

# ----------------------------------------------------------------- parameters
KAPPA = 1.0          # atom-cavity decay (unit of rate)
GAMMA = 1.0          # remote-cavity linewidth
DELTA_A = 0.0        # atomic (Zeeman) detuning
SIGMA_I = 1.86       # input photon width (Table 1, optimised)
DELTA_0 = 0.53       # remote-cavity operating detuning (Table 1, optimised)
T0, T1, DT = -8.0, 16.0, 0.01
K, W = 12, 1.3       # Gaussian radial-basis control: nodes and overlap width
T_NODES = np.linspace(-3.0, 9.0, K)
ENERGY = 1.0         # fixed control energy  int |alpha|^2 dt

ts = jnp.arange(T0, T1, DT)
NSTEP = ts.size


def xi_in(t):
    return (np.pi * SIGMA_I**2) ** -0.25 * jnp.exp(-t**2 / (2 * SIGMA_I**2))


# -------------------------------------------------------- quantum Fisher info
def fock_qfi(sigma=SIGMA_I, d=DELTA_0, g=GAMMA):
    """F_Q = 4 Var_{|xi(w)|^2}[d_delta phi], Eqs. (50)-(53)."""
    w = np.linspace(-40, 40, 400001)
    P = np.exp(-(w * sigma) ** 2)
    P /= np.trapezoid(P, w)
    gd = -4 * g / (g**2 + 4 * (w - d) ** 2)
    m = np.trapezoid(P * gd, w)
    return 4 * (np.trapezoid(P * gd**2, w) - m**2)


F_Q = fock_qfi()


# ------------------------------------------------------------------- dynamics
def rhs(state, t, alpha, delta_r):
    y, c1, c0 = state
    xi = xi_in(t)
    xo = -xi + y
    dy = -(GAMMA / 2 + 1j * delta_r) * y + GAMMA * xi
    dc1 = -(KAPPA / 2) * c1 - 1j * alpha * c0 + jnp.sqrt(KAPPA) * xo
    dc0 = 1j * DELTA_A * c0 - 1j * jnp.conj(alpha) * c1
    return jnp.stack([dy, dc1, dc0])


def evolve(alpha_grid, delta_r):
    """RK4 on a fixed grid; alpha is sampled at t, t+dt/2, t+dt (array of 2N+1)."""
    def step(s, i):
        t = ts[i]
        a0, ah, a1 = alpha_grid[2 * i], alpha_grid[2 * i + 1], alpha_grid[2 * i + 2]
        k1 = rhs(s, t, a0, delta_r)
        k2 = rhs(s + DT / 2 * k1, t + DT / 2, ah, delta_r)
        k3 = rhs(s + DT / 2 * k2, t + DT / 2, ah, delta_r)
        k4 = rhs(s + DT * k3, t + DT, a1, delta_r)
        s = s + DT / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return s, s

    s0 = jnp.zeros(3, dtype=jnp.complex128)
    sT, traj = jax.lax.scan(step, s0, jnp.arange(NSTEP))
    return sT, traj


def p_ground(alpha_grid, delta_r):
    sT, _ = evolve(alpha_grid, delta_r)
    return jnp.abs(sT[2]) ** 2


dp_ddelta = jax.jacfwd(p_ground, argnums=1)


def fisher(alpha_grid, delta_r=DELTA_0):
    p = p_ground(alpha_grid, delta_r)
    dp = dp_ddelta(alpha_grid, delta_r)
    return dp**2 / (p * (1 - p))


# half-step time grid on which alpha is evaluated
t_fine = jnp.linspace(T0, T0 + NSTEP * DT, 2 * NSTEP + 1)


# ------------------------------------------------- Gaussian baseline control
def alpha_gauss(theta):
    A, t_a, s_a, phi = theta
    return A * jnp.exp(-((t_fine - t_a) ** 2) / (2 * s_a**2)) * jnp.exp(1j * phi)


# ------------------------------------------- radial-basis (RNN-weight) control
G = np.sqrt(np.pi) * W * np.exp(-((T_NODES[:, None] - T_NODES[None, :]) ** 2) / (4 * W**2))
BASIS = jnp.exp(-0.5 * ((t_fine[None, :] - T_NODES[:, None]) / W) ** 2)  # (K, 2N+1)


def coeffs(x):
    c = x[:K] + 1j * x[K:]
    norm = jnp.sqrt(jnp.real(jnp.conj(c) @ (G @ c)))
    return c * jnp.sqrt(ENERGY) / norm      # exact energy normalisation c -> c/sqrt(c^dag G c)


def alpha_rbf(x):
    return coeffs(x) @ BASIS


F_rbf = jax.jit(lambda x: fisher(alpha_rbf(x)))
negF_and_grad = jax.jit(jax.value_and_grad(lambda x: -fisher(alpha_rbf(x))))


def main():
    print(f"Fock QFI at sigma_i={SIGMA_I}, delta_0={DELTA_0}: F_Q = {F_Q:.3f}")

    # 1. baseline: best single Gaussian control of the same energy
    def gauss_energy_fixed(th):
        t_a, s_a = th
        A = (ENERGY / (np.sqrt(np.pi) * abs(s_a))) ** 0.5
        return jnp.array([A, t_a, abs(s_a), 0.0])

    fg = jax.jit(lambda th: fisher(alpha_gauss(gauss_energy_fixed(th))))
    best = None
    for t_a in [0.0, 1.0, 2.0]:
        for s_a in [1.5, 2.5, 3.5]:
            r = minimize(lambda th: -float(fg(jnp.array(th))), [t_a, s_a], method="Nelder-Mead",
                         options=dict(xatol=1e-4, fatol=1e-6))
            if best is None or r.fun < best.fun:
                best = r
    t_a, s_a = best.x
    F_gauss = -best.fun
    print(f"Gaussian control (energy {ENERGY}): t_a={t_a:.2f}, sigma_a={abs(s_a):.2f}, F = {F_gauss:.3f}")

    # initial RBF weights: least-squares fit of that Gaussian on the nodes
    a_target = np.asarray(alpha_gauss(gauss_energy_fixed(jnp.array(best.x))))
    c0, *_ = np.linalg.lstsq(np.asarray(BASIS).T, a_target, rcond=None)
    x0 = np.concatenate([c0.real, c0.imag])
    F_init = float(F_rbf(jnp.array(x0)))
    print(f"RBF fit of Gaussian: F = {F_init:.3f}")

    # 2. BPTT / L-BFGS
    hist = [F_init]
    npass = [0]

    def fun(x):
        v, g = negF_and_grad(jnp.array(x))
        npass[0] += 1
        return float(v), np.asarray(g)

    t_start = time.time()
    res = minimize(fun, x0, jac=True, method="L-BFGS-B",
                   callback=lambda xk: hist.append(float(F_rbf(jnp.array(xk)))),
                   options=dict(maxiter=80, gtol=1e-9, ftol=1e-12))
    x_opt = res.x
    hist = np.array(hist)
    print(f"BPTT: F = {hist[-1]:.3f} ({100*hist[-1]/F_Q:.1f}% of QFI) "
          f"in {len(hist)-1} iters, {npass[0]} fwd+adjoint passes, {time.time()-t_start:.0f}s")

    # 3. gradient-free comparison (CMA-ES) from the same start
    try:
        import cma
        es = cma.CMAEvolutionStrategy(x0, 0.1 * np.abs(x0).max(),
                                      dict(maxfevals=1700, verbose=-9, seed=1))
        F_vec = jax.jit(jax.vmap(lambda x: fisher(alpha_rbf(x))))
        while not es.stop():
            X = es.ask()
            es.tell(X, list(-np.asarray(F_vec(jnp.array(X)))))
        F_cma = -es.result.fbest
        n_cma = es.result.evaluations
        print(f"CMA-ES: F = {F_cma:.3f} after {n_cma} evaluations")
    except ImportError:
        F_cma, n_cma = None, 0

    # sanity: probability conservation for the optimum
    a_opt = alpha_rbf(jnp.array(x_opt))
    sT, traj = evolve(a_opt, DELTA_0)
    xi_o = -xi_in(ts + DT) + traj[:, 0]
    xi_out = xi_o - np.sqrt(KAPPA) * traj[:, 1]
    print(f"p_g = {float(abs(sT[2])**2):.4f},  p_g + int|xi_out|^2 = "
          f"{float(abs(sT[2])**2 + np.sum(abs(xi_out)**2)*DT):.4f}")

    np.savez("fig10_data.npz", hist=hist, x_opt=x_opt, x0=x0, F_Q=F_Q, F_gauss=F_gauss,
             F_cma=F_cma if F_cma is not None else np.nan, t_nodes=T_NODES)
    plot(hist, x_opt, F_cma, n_cma, npass[0])


def plot(hist, x_opt, F_cma, n_cma, npass):
    plt.rcParams.update({"font.size": 10, "axes.linewidth": 0.8})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.9))
    it = np.arange(len(hist))
    ax1.plot(it, hist, "-o", color="#1f3b73", ms=3, lw=1.6,
             label="BPTT (adjoint = backprop-through-time)")
    ax1.axhline(F_Q, color="#e07020", ls="--", lw=1.4, label=f"QFI bound / lossless readout ({F_Q:.2f})")
    if F_cma is not None:
        ax1.axhline(F_cma, color="0.35", ls=":", lw=1.4, label=f"gradient-free (CMA-ES), {n_cma} evals")
    ax1.set_xlabel("BPTT / L-BFGS iteration")
    ax1.set_ylabel(r"readout Fisher information $F_\delta$")
    ax1.set_title(f"BPTT training: {100*hist[-1]/F_Q:.0f}% of QFI in {len(hist)-1} iters "
                  f"({npass} fwd+adjoint passes)", fontsize=9.5)
    ax1.legend(loc="lower right", fontsize=8, frameon=False)
    ax1.set_xlim(-1, len(hist))

    t_plot = np.linspace(-4, 10, 600)
    c = np.asarray(coeffs(jnp.array(x_opt)))
    a = (c[None, :] * np.exp(-0.5 * ((t_plot[:, None] - T_NODES[None, :]) / W) ** 2)).sum(1)
    a_nodes = (c[None, :] * np.exp(-0.5 * ((T_NODES[:, None] - T_NODES[None, :]) / W) ** 2)).sum(1)
    ax2.plot(t_plot, np.abs(a), color="#1f3b73", lw=1.8, label=r"gate amplitude $|\alpha(t)|$")
    ax2.plot(T_NODES, np.abs(a_nodes), "o", mfc="white", mec="#1f3b73", ms=5, label="Gaussian weight nodes")
    ax2.set_ylim(0, 1.15 * np.abs(a).max())
    ax2.set_xlabel(r"time $t$ ($1/\gamma$)  (unrolled RNN depth)")
    ax2.set_ylabel(r"recurrent-weight amplitude $|\alpha|$")
    ax3 = ax2.twinx()
    ph = np.unwrap(np.angle(a))
    ph -= ph[np.argmax(np.abs(a))]
    ax3.plot(t_plot, ph, ":", color="#c0392b", lw=1.5, label=r"weight phase $\arg\alpha$")
    ax3.set_ylabel(r"weight phase $\arg\alpha$", color="#c0392b")
    ax3.tick_params(axis="y", colors="#c0392b")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8, frameon=False)
    ax2.set_title(r"Learned recurrent weights $\alpha(t)$ (sum of Gaussians)", fontsize=9.5)
    fig.tight_layout()
    fig.savefig("fig10_reproduced.pdf")
    fig.savefig("fig10_reproduced.png", dpi=200)
    print("saved fig10_reproduced.pdf / .png")


if __name__ == "__main__":
    main()
