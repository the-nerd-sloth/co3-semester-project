"""
CO3 Project -- Spring 2026
Fair water distribution in rural Nepal.

Simulated Annealing: constraint-preserving metaheuristic
following the pseudocode in CO3 Chapter 7.
Requires the parameter vectors (n, d, b, S, a, w) and
the objective function defined in lagrange_newton.py.
"""

import numpy as np

# ---------------------------------------------------------------------------
# Parameters (identical to Lagrange-Newton baseline)
# ---------------------------------------------------------------------------
n = 20
d = np.full(n, 225.0)   # demand per household             [L/day]
b = np.full(n, 100.0)   # WHO-aligned minimum              [L/day]
S = 3500.0              # total available spring yield      [L/day]

a = np.linspace(1.00, 1.25, n)   # delivery-loss factors
w = np.linspace(1.00, 1.80, n)   # vulnerability weights

# ---------------------------------------------------------------------------
# Objective (weighted sum of squared shortfalls)
# ---------------------------------------------------------------------------
def objective(x):
    return np.sum(w * (d - x) ** 2)

# ---------------------------------------------------------------------------
# SA hyper-parameters
# ---------------------------------------------------------------------------
T_0       = 500.0     # initial temperature
k_max     = 50_000    # total iteration budget
step_size = 20.0      # max water shift per move [L/day]

# ---------------------------------------------------------------------------
# Feasible initial point
# Distribute S proportionally to (d_i - b_i), then clip to [b, d].
# ---------------------------------------------------------------------------
def sa_make_x0():
    x = b.copy()
    remaining = S - np.dot(a, x)          # source water still to assign
    capacity  = np.dot(a, d - b)          # total assignable headroom
    if remaining > 0 and capacity > 0:
        fill = np.minimum(d - x, remaining * a * (d - b) / capacity)
        x   += fill / a
    return np.clip(x, b, d)

# ---------------------------------------------------------------------------
# Constraint-preserving transfer move
# Transfers delta L/day from household i to j, adjusted by loss factors
# so that sum_k a_k * x_k stays exactly equal to S.
#   a_i * (-delta) + a_j * (delta * a_i / a_j) = 0  =>  sum unchanged
# The feasible window is computed BEFORE sampling to guarantee
# b_i <= x_i <= d_i without any post-hoc clipping.
# ---------------------------------------------------------------------------
def sa_perturb(x):
    x_new = x.copy()
    i, j  = np.random.choice(n, size=2, replace=False)

    max_take     = x_new[i] - b[i]               # how much i can give up
    max_give     = d[j]  - x_new[j]              # how much j can absorb
    max_give_src = max_give * a[j] / a[i]        # translated to i-side

    limit = min(max_take, max_give_src, step_size)
    if limit < 1e-9:
        return x_new                              # move not possible

    delta      = np.random.uniform(0, limit)     # sample within window
    x_new[i]  -= delta
    x_new[j]  += delta * a[i] / a[j]            # preserve sum(a*x) = S

    # NOTE: no np.clip here -- clipping would silently break sum(a*x) = S
    return x_new

# ---------------------------------------------------------------------------
# Simulated Annealing -- Steps 1-5 from CO3 Chapter 7
# ---------------------------------------------------------------------------
np.random.seed(42)

# Step 1: initialise
x_cur  = sa_make_x0()
f_cur  = objective(x_cur)
x_best = x_cur.copy()
f_best = f_cur

for k in range(k_max):

    # Step 2: propose new candidate (always feasible by construction)
    x_test = sa_perturb(x_cur)
    f_test = objective(x_test)

    # Step 3: accept deterministically if better
    if f_test <= f_cur:
        x_cur, f_cur = x_test, f_test

    # Step 4: accept with Metropolis probability if worse
    else:
        # Step 5: linear cooling schedule (CO3 Chapter 7)
        T_k = T_0 * (1 - (k + 1) / k_max)      # T_{k+1} = T_0*(1-(k+1)/k_max)
        if T_k > 1e-10 and np.exp(-(f_test - f_cur) / T_k) >= np.random.rand():
            x_cur, f_cur = x_test, f_test

    # Track global best across all visited states
    if f_cur < f_best:
        f_best = f_cur
        x_best = x_cur.copy()

x_sa = x_best

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
print(f"SA objective f*     : {f_best:.4f}")
print(f"Total delivered     : {np.sum(x_sa):.2f} L/day")
print(f"Constraint |Ax - S| : {abs(np.dot(a, x_sa) - S):.2e}")
print(f"Std dev (equity)    : {np.std(x_sa):.4f} L/day")
print(f"Max unmet demand    : {np.max(d - x_sa):.2f} L/day")
