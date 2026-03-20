"""
Genetic Algorithm Toolkit (GATO) — GAME v1.0.0
Refactored and extended with GAME methodology by M. Peronaci.

GitHub: https://github.com/matteoperonaci/GAME
"""

import os
import time
import pickle
import random as rnd
import copy
import warnings
from typing import List, Callable, Dict

import numpy as np
import sympy as sym
from scipy.optimize import minimize
from scipy.special import erf
from joblib import Parallel, delayed

# ==========================================
# 1. Grammar & Mathematical Functions
# ==========================================

def poly(x):      return x
def cpl(x):       return x / (1.0 + x)
def cpl2(x):      return x / (1.0 + x**2)
def cpla(x):      return 1.0 - x
def cheb2(x):     return -1.0 + 2.0 * x**2
def cheb3(x):     return -3.0 * x + 4.0 * x**3
def cheb4(x):     return 1.0 - 8.0 * x**2 + 8.0 * x**4
def cheb5(x):     return 5.0 * x - 20.0 * x**3 + 16.0 * x**5
def polyxtox(x):  return x**x

def exp(x):
    return sym.exp(x) if isinstance(x, sym.Expr) else np.exp(x)

def log(x):
    return sym.log(x) if isinstance(x, sym.Expr) else np.log(x)

def log10(x):
    return sym.log(x, 10) if isinstance(x, sym.Expr) else np.log10(x)

def trig1(x):
    return sym.cos(x) if isinstance(x, sym.Expr) else np.cos(x)

def tanh(x):
    return sym.tanh(x) if isinstance(x, sym.Expr) else np.tanh(x)

def inv_poly1(x):
    if isinstance(x, sym.Expr):
        return 1 / (1 + x)
    return 1.0 / (1.0 + x + 1e-9)

def cplexp(x):
    if isinstance(x, sym.Expr):
        ex = sym.exp(x); return ex / (1 + ex)
    ex = np.exp(x); return ex / (1 + ex)

GRAMMAR_MAP = {
    'poly': poly,      'polyxtox': polyxtox, 'exp': exp,       'cpl': cpl,
    'trig1': trig1,    'cpl2': cpl2,         'cpla': cpla,     'cplexp': cplexp,
    'cheb2': cheb2,    'cheb3': cheb3,       'cheb4': cheb4,   'cheb5': cheb5,
    'inv_poly1': inv_poly1, 'tanh': tanh,    'log': log,       'log10': log10,
}

# ==========================================
# 2. Genetic Algorithm Core Logic
# ==========================================

def makeakid(ranges, length):
    return [
        [
            rnd.uniform(ranges[0][0], ranges[0][1]),
            rnd.randrange(ranges[1][0], ranges[1][1]),
            rnd.uniform(ranges[2][0], ranges[2][1]),
            rnd.randrange(ranges[3][0], ranges[3][1]),
        ]
        for _ in range(length)
    ]

def make_function(x, kid, gram, func_prior):
    func0 = 0.0
    for gene in kid:
        c, g_idx, k, p = gene
        basis_func = GRAMMAR_MAP.get(gram[g_idx], poly)
        func0 += c * (basis_func(k * x) ** p)
    return func_prior(x, func0)

def mutation(kid, ranges):
    kid_copy = copy.deepcopy(kid)
    row = rnd.randrange(0, len(kid))
    col = rnd.randrange(0, 4)
    if col == 1:
        kid_copy[row][col] = rnd.randrange(ranges[1][0], ranges[1][1])
    elif col == 3:
        kid_copy[row][col] = rnd.randrange(ranges[3][0], ranges[3][1])
    else:
        r_idx = 0 if col == 0 else 2
        kid_copy[row][col] = rnd.uniform(ranges[r_idx][0], ranges[r_idx][1])
    return kid_copy

def crossover(kid1, kid2):
    pt = rnd.randrange(1, len(kid1))
    return kid1[:pt] + kid2[pt:], kid1[pt:] + kid2[:pt]

def tournament_selection(fitness_pop, k=4, select_winner=True):
    participants = rnd.sample(fitness_pop, k)
    sorted_p = sorted(participants, key=lambda x: x[0])
    return sorted_p[-1][1] if select_winner else sorted_p[0][1]

def _delete_indices(data_list, indices):
    idx_set = set(indices)
    return [ele for i, ele in enumerate(data_list) if i not in idx_set]

# ==========================================
# 3. Chi2 Marginalisation
# ==========================================

def compute_marginalised_chi2(kid, gram, X, Y, sY, func_prior, qchi=1):
    vecy = make_function(X, kid, gram, func_prior)
    if qchi == 0:
        A = np.sum(((Y - vecy) / sY) ** 2)
        B = np.sum((Y - vecy) / sY ** 2)
        C = np.sum(1.0 / sY ** 2)
    else:
        A = np.sum((Y / sY) ** 2)
        B = np.sum((Y * vecy) / sY ** 2)
        C = np.sum((vecy / sY) ** 2)
    return A - B ** 2 / C

def compute_offset(kid, gram, X, Y, sY, func_prior, qchi=1):
    vecy = make_function(X, kid, gram, func_prior)
    if qchi == 0:
        B = np.sum((Y - vecy) / sY ** 2)
        C = np.sum(1.0 / sY ** 2)
    else:
        B = np.sum((Y * vecy) / sY ** 2)
        C = np.sum((vecy / sY) ** 2)
    return B / C

# ==========================================
# 4. Evolution Loop
# ==========================================

DEFAULT_PARAMS = {
    'Nchains': 4,  'Ngens': 100,   'Npops': 100,   'Nseed': 123,
    'ranges': [[-1, 1], [0, 2], [0, 2], [0, 10]],
    'length': 4,   'selectionrate': 0.3, 'toursize': 4,
    'crossoverrate': 0.75, 'mutationrate': 0.3,
    'verbose': True,
    'grammar': ['poly'],
}

def evolution(chi2f, input_params, gram):
    params = DEFAULT_PARAMS.copy()
    params.update(input_params)
    rnd.seed(params['Nseed'])

    n_pops       = params['Npops']
    n_gens       = params['Ngens']
    max_children = int(round((params['selectionrate'] * n_pops) / 2.) * 2)

    kids           = [None] * n_gens
    bestfitperstep = [None] * n_gens

    kids[0] = [makeakid(params['ranges'], params['length']) for _ in range(n_pops)]

    fitness = [[-chi2f(kid, gram), i] for i, kid in enumerate(kids[0])]
    fitness.sort(key=lambda x: x[0])
    bestfitperstep[0] = [-fitness[-1][0], kids[0][fitness[-1][1]]]

    for gen in range(1, n_gens):
        prev_pop = kids[gen - 1]

        loser_indices = set()
        while len(loser_indices) < max_children:
            loser_indices.add(tournament_selection(fitness, params['toursize'], select_winner=False))
        loser_indices = list(loser_indices)

        new_children = [None] * max_children
        for i in range(0, max_children, 2):
            p1 = prev_pop[tournament_selection(fitness, params['toursize'], select_winner=True)]
            p2 = prev_pop[tournament_selection(fitness, params['toursize'], select_winner=True)]
            if rnd.random() < params['crossoverrate']:
                c1, c2 = crossover(p1, p2)
            else:
                c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)

            if rnd.random() < params['mutationrate']:
                c1 = mutation(c1, params['ranges'])
            if rnd.random() < params['mutationrate']:
                c2 = mutation(c2, params['ranges'])
            new_children[i], new_children[i + 1] = c1, c2

        kids[gen] = new_children + _delete_indices(prev_pop, loser_indices)
        fitness   = [[-chi2f(kid, gram), i] for i, kid in enumerate(kids[gen])]
        fitness.sort(key=lambda x: x[0])
        bestfitperstep[gen] = [-fitness[-1][0], kids[gen][fitness[-1][1]]]

    return bestfitperstep

# ==========================================
# 5. GAME: Expression Collection & Weighting
# ==========================================

def roughness(f, z_grid):
    f_vals = f(z_grid)
    dz     = z_grid[1] - z_grid[0]
    d2 = (-f_vals[4:] + 16 * f_vals[3:-1] - 30 * f_vals[2:-2] + 16 * f_vals[1:-3] - f_vals[:-4]) / (12 * dz ** 2)
    return np.sum(d2 ** 2) * dz


def _process_chain(bf_params, final_chi2, grammar, X, Y, sY, func_prior, qchi, z_grid, simplify):
    """Worker function to process SymPy expressions in parallel."""
    xs = sym.Symbol('x')
    
    # 1. Build symbolic expression
    ga_expr = make_function(xs, bf_params, grammar, func_prior)
    y0      = compute_offset(bf_params, grammar, X, Y, sY, func_prior, qchi)
    expr    = (ga_expr + y0) if qchi == 0 else (ga_expr * y0)
    
    if simplify:
        expr = sym.nsimplify(expr, rational=True)
        
    # 2. Lambdify with scalar-protection (Prevents shape bugs if expr is constant)
    f_raw = sym.lambdify(xs, expr, ['numpy', 'scipy'])
    def f_safe(z):
        res = f_raw(z)
        return np.full_like(z, res) if np.isscalar(res) else res
        
    # 3. Evaluate Roughness
    R = roughness(f_safe, z_grid)
    
    return final_chi2, R, expr, f_safe

def collect_expressions(run, bfps, grammarchain,
                        X, Y, sY, func_prior, z_grid,
                        qchi=1, n_jobs=-1, simplify=True):
    """Build offset-applied symbolic expressions and compute roughness fully in parallel."""
    results = Parallel(n_jobs=n_jobs)(
        delayed(_process_chain)(
            run[i][-1][1], float(bfps[i][-1]), grammarchain[i], 
            X, Y, sY, func_prior, qchi, z_grid, simplify
        )
        for i in range(len(run))
    )

    chi2_arr  = np.array([r[0] for r in results])
    R_arr     = np.array([r[1] for r in results])
    expr_list = [r[2] for r in results]
    f_list    = [r[3] for r in results]

    return chi2_arr, R_arr, expr_list, f_list

def compute_game_weights(chi2_arr, R_arr, expr_list, f_list,
                         lambda_min=1e-30, lambda_max=1e10, K=10_000):
    """
    Run the full GAME pipeline: L-curve elbow selection, weights and averaged function.

    For each lambda on a log grid, the model minimising S_j(lambda) = chi2_j + lambda*R_j
    is identified. The subset of models that are optimal for at least one lambda forms the
    lower envelope. The elbow of this envelope in log(R)-log(chi2) space marks the optimal
    trade-off between goodness-of-fit and smoothness. lambda_elbow is then chosen as the
    geometric mean of the two breakpoints bracketing the elbow model, placing it strictly
    inside the interval where that model is the unique minimiser (see paper Sec. 2.3.1).
    """
    xs          = sym.Symbol('x')
    lambda_grid = np.logspace(np.log10(lambda_min), np.log10(lambda_max), K)

    # --- Step 1: build lower envelope ---
    J           = chi2_arr[None, :] + lambda_grid[:, None] * R_arr[None, :]
    min_indices = np.argmin(J, axis=1)
    env_indices = np.unique(min_indices)

    chi2_env = chi2_arr[env_indices]
    R_env    = R_arr[env_indices]
    order    = np.argsort(R_env)
    env_indices, chi2_env, R_env = env_indices[order], chi2_env[order], R_env[order]

    # --- Step 2: find lambda_elbow ---
    if len(env_indices) == 1:
        # Only one model is ever optimal: lambda is irrelevant, any value works.
        lambda_elbow = lambda_min

    elif len(env_indices) == 2:
        # Two-point envelope: the single breakpoint lambda where they tie.
        dR       = R_env[1] - R_env[0]
        lam_star = (chi2_env[0] - chi2_env[1]) / dR if dR != 0 else -1.0
        # If lam_star <= 0, one model dominates for all lambda > 0.
        lambda_elbow = (lambda_min if lam_star <= 0
                        else float(np.clip(lam_star, lambda_min, lambda_max)))

    else:
        # --- General case (3+ envelope models) ---
        #
        # The envelope is sorted by INCREASING R:
        #   index 0   = smoothest model (small R, high chi2)
        #   index M-1 = roughest  model (large R, low  chi2)
        #
        # Physical intuition: at large λ the roughness penalty R dominates, so smoother
        # models win; at small λ the data fit chi2 dominates, so rougher models win.
        # The elbow is the model that balances both — the optimal trade-off point.
        #
        # Step A: find k_star — the elbow model index.
        # Work in log(R)-log(chi2) space (matching the L-curve plot axes).
        # The elbow is the envelope point farthest from the straight line connecting
        # the two endpoints, measured as perpendicular distance.
        xl = np.log(R_env)
        yl = np.log(chi2_env)
        p1 = np.array([xl[0],  yl[0]])
        p2 = np.array([xl[-1], yl[-1]])
        v  = (p2 - p1) / np.linalg.norm(p2 - p1)   # unit vector along the diagonal
        dists = [np.linalg.norm(np.array([xl[j], yl[j]]) - p1
                                - np.dot(np.array([xl[j], yl[j]]) - p1, v) * v)
                    for j in range(1, len(xl) - 1)]     # skip endpoints (index 0 and M-1)
        k_star = 1 + int(np.argmax(dists))           # +1 because we skipped index 0

        # Step B: compute the analytical breakpoints between consecutive envelope models.
        #
        # lam_breaks[i] is the unique λ where models i and i+1 are equally good,
        # i.e. where S_i(λ) = S_{i+1}(λ):
        #   chi2_i + λ·R_i = chi2_{i+1} + λ·R_{i+1}
        #   → λ = (chi2_i − chi2_{i+1}) / (R_{i+1} − R_i)
        #
        # Because R increases and chi2 decreases along the envelope, each breakpoint
        # is positive, and lam_breaks is a DECREASING sequence:
        #   lam_breaks[0] > lam_breaks[1] > ... > lam_breaks[M-2]
        # This means model 0 (smoothest) becomes optimal only at very large λ,
        # and model M-1 (roughest) is optimal only at very small λ.
        lam_breaks = np.array([
            (chi2_env[i] - chi2_env[i + 1]) / (R_env[i + 1] - R_env[i])
            for i in range(len(R_env) - 1)
        ])

        # Step C: determine the valid λ interval for k_star.
        #
        # Model k_star is the unique minimiser of S_j(λ) for λ in (lam_lower, lam_upper):
        #
        #   lam_lower = lam_breaks[k_star]:   below this λ, the rougher neighbour
        #               (k_star+1) has lower S and wins over k_star.
        #
        #   lam_upper = lam_breaks[k_star-1]: above this λ, the smoother neighbour
        #               (k_star-1) has lower S and wins over k_star.
        #
        # Because lam_breaks is decreasing, lam_lower < lam_upper always holds.
        # At the boundary models (k_star=0 or k_star=M-1), one neighbour does not
        # exist, so the corresponding bound is open (replaced by lambda_min/max).
        lam_upper = lam_breaks[k_star - 1] if k_star > 0              else lambda_max
        lam_lower = lam_breaks[k_star]     if k_star < len(lam_breaks) else lambda_min

        if 0 < k_star < len(lam_breaks):
            # Interior elbow: place lambda_elbow at the geometric mean of the two
            # breakpoints, which is the log-centre of (lam_lower, lam_upper).
            lambda_elbow = np.sqrt(lam_upper * lam_lower)
        elif k_star == 0:
            # Smoothest model is the elbow. Its valid range is (lam_lower, +∞).
            # Use 10× the lower breakpoint as a representative value safely inside.
            lambda_elbow = lam_lower * 10.0
        else:
            # Roughest model is the elbow. Its valid range is (0, lam_upper).
            # Use 1/10 of the upper breakpoint — symmetric to the k_star=0 case.
            lambda_elbow = lam_upper / 10.0

        lambda_elbow = float(np.clip(lambda_elbow, lambda_min, lambda_max))


    # --- Step 3: compute weights (paper Eq. 2.23) ---
    # w_j = exp(-1/2 * (S_j(lambda_elbow) - S_min(lambda_elbow)))
    S     = chi2_arr + lambda_elbow * R_arr
    S_min = S.min()
    w     = np.exp(-0.5 * (S - S_min))  
    w    /= w.sum()

    # --- Step 4: build f_GAME callable (paper Eq. 2.24) ---
    # f_GAME(x) = sum_j w_j * f_j,GA(x) — a numerical weighted average.
    def f_game(z):
        z    = np.asarray(z, dtype=float)
        vals = np.array([f(z) for f in f_list])  # shape (N_conf, len(z))
        return np.einsum('j,jk->k', w, vals)

    return w, lambda_elbow, f_game

# ==========================================
# 6. Utilities
# ==========================================

def generate_mock(ndat, xlims, use_uniform_z_space, fiducial_function,
                  error_fact, include_sigma_redshift=False, fix_extremes=False):
    if use_uniform_z_space:
        xdata = np.linspace(xlims[0], xlims[1], ndat)
    else:
        if fix_extremes:
            inner = np.sort(np.random.uniform(xlims[0], xlims[1], ndat - 2))
            xdata = np.concatenate(([xlims[0]], inner, [xlims[1]]))
        else:
            xdata = np.sort(np.random.uniform(xlims[0], xlims[1], ndat))

    fid_vals = fiducial_function(xdata)
    errors   = (np.abs(error_fact * fid_vals * (1 + xdata))
                if include_sigma_redshift else np.abs(error_fact * fid_vals))
    ydata    = np.random.normal(fid_vals, scale=errors)
    return xdata, ydata, errors

# ==========================================
# 7. Error Reconstruction (Path Integral)
# ==========================================

def _parapoly(x, a, grade, use_cpl=False):
    var = x / (1.0 + x) if use_cpl else x
    return np.polyval(a[::-1], var)

def _chi2_total_wrapper(params, X, Y, sY, funcGA, grade, use_cpl, qchi=1):
    poly_val = _parapoly(X, params, grade, use_cpl)
    ga_val   = funcGA(X)
    factor   = 1.0 / (np.sqrt(2) * sY)
    CI_i     = 0.5 * (erf(factor * (poly_val + ga_val - Y))
                      + erf(factor * (poly_val - ga_val + Y)))
    chi2_CI  = np.sum((CI_i - erf(1.0 / np.sqrt(2))) ** 2)

    # Regularisation term: require the perturbed reconstruction fGA ± δf to remain
    # statistically consistent with the data using the same marginalisation as the GA.
    fx  = funcGA(X) + _parapoly(X, params, grade, use_cpl)
    w_  = 1.0 / sY ** 2
    if qchi == 0:   # additive marginalisation (Eq. 2.8)
        A = np.sum(((Y - fx) / sY) ** 2)
        B = np.sum((Y - fx) * w_)
        C = np.sum(w_)
    else:           # multiplicative marginalisation (Eq. 2.9) — default for H(z)
        A = np.sum((Y / sY) ** 2)
        B = np.sum(Y * fx * w_)
        C = np.sum(fx ** 2 * w_)
    chi2_fit = A - B ** 2 / C
    return chi2_CI + chi2_fit

def compute_dfuncGA(X, Y, sY, funcGA, grade=1, use_cpl=False, starting_guess=1e-2, qchi=1):
    result = minimize(
        _chi2_total_wrapper,
        x0=[starting_guess] * (grade + 1),
        args=(X, Y, sY, funcGA, grade, use_cpl, qchi),
        method='L-BFGS-B',
        tol=1e-12,
        options={'maxiter': 100_000},
    )
    if not result.success:
        warnings.warn(f"Path-integral minimisation did not converge: {result.message}")
    return lambda x: _parapoly(x, result.x, grade, use_cpl)

# ==========================================
# 8. Plotting
# ==========================================

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset

def plot_chi2_evolution(bfps, chi2_threshold, input_params,
                        seedchain, crosschain, mutchain, grammarchain,
                        dpi=100, nchains_shown=7):
    nchains = input_params['Nchains']
    ngens   = input_params['Ngens']
    gens    = np.arange(1, ngens + 1)

    all_final = [float(row[-1]) for row in bfps]
    min_chi2  = min(all_final)
    idx_min   = all_final.index(min_chi2)

    fig, ax = plt.subplots(dpi=dpi)
    ax.axhline(chi2_threshold, lw=2.5, ls='--', color='C0', zorder=1000,
               label=fr'$\chi^2_{{\rm thr}} = {chi2_threshold:.3f}$')

    plotted = 0
    for i in range(nchains):
        color = f'C{i + 1}'
        label = None
        if i == idx_min:
            color = f'C{nchains_shown + 2}'
            label = (fr'$\chi^2_{{\rm min}} = {min_chi2:.3f}$'
                     f'\nseed={seedchain[i]}, cross={crosschain[i]},'
                     f' mut={mutchain[i]}, gramm={grammarchain[i]}')
        elif plotted < nchains_shown:
            label  = (f'seed={seedchain[i]}, cross={crosschain[i]},'
                      f' mut={mutchain[i]}, gramm={grammarchain[i]}')
            plotted += 1
        elif plotted == nchains_shown:
            label   = r'$...$'
            plotted += 1
        ax.plot(gens, bfps[i], lw=1.5, color=color, alpha=0.5, label=label)

    ax.set(xlabel=r'$n_{gen}$', ylabel=r'$\chi^2$', xscale='log', yscale='log')
    ax.legend(frameon=False, bbox_to_anchor=(1., 0.5), loc='center left')

    axins = inset_axes(ax, width="45%", height="45%", loc='upper right')
    axins.axhline(chi2_threshold, lw=3.5, ls='--', color='C0', zorder=1000)
    for i in range(nchains):
        c = f'C{nchains_shown + 2}' if i == idx_min else f'C{i + 1}'
        axins.plot(gens, bfps[i], lw=1.5, color=c, alpha=0.5)

    flat_vals = [v for chain in bfps for v in chain]
    axins.set_xlim(0.4 * ngens, ngens + 1)
    axins.set_ylim(0.98 * min(min(flat_vals), chi2_threshold), 1.05 * chi2_threshold)
    axins.tick_params(axis='x', rotation=45)
    mark_inset(ax, axins, loc1=3, loc2=4, fc='none', ec='C0', lw=1.0, alpha=0.8)
    plt.show()