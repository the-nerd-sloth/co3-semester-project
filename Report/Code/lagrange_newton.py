"""
CO3 Project -- Spring 2026
Fair water distribution in rural Nepal.

Lagrange-Newton baseline: gradient-based, constrained QP
solved via scipy.optimize.minimize (trust-constr).
"""

import numpy as np
from scipy.optimize import minimize, LinearConstraint, Bounds

# --- parameters --------------------------------------------------------------
n = 20
d = np.full(n, 225.0)   # demand per household                [L/day]
b = np.full(n, 100.0)   # WHO-aligned minimum per household   [L/day]
S = 3500.0              # total available spring yield        [L/day]

# delivery-loss factor range (terrain/pipe overhead).
# a_i in [1.00, 1.25]  means 0%-25% overhead at the source.
a = np.linspace(1.00, 1.25, n)

# vulnerability priority range (equity weights).
# Higher weight -> shortages for that household cost more.
w = np.linspace(1.00, 1.80, n)

# --- sanity checks -----------------------------------------------------------
assert n > 0
assert np.all(d >= b), "demand must be >= minimum basic water"
assert S > 0
assert np.all(a >= 1.0)
assert np.all(w > 0.0)

S_min = np.sum(a * b)
S_max = np.sum(a * d)
print(f"Feasible source interval: [{S_min:.1f}, {S_max:.1f}] L/day")
print(f"Chosen total supply S   : {S:.1f} L/day")
assert S_min <= S, "infeasible: supply below total minimum requirement"
if S > S_max:
    print("Note: supply above total demand; some water may remain unused.")

# --- objective and gradient --------------------------------------------------
def objective(x):
    return np.sum(w * (d - x) ** 2)

def objective_grad(x):
    return 2.0 * w * (x - d)

# --- constraints and bounds --------------------------------------------------
A = a.reshape(1, -1)                          # source-side balance matrix
eq_constr = LinearConstraint(A, lb=S, ub=S)   # sum_i a_i x_i == S
bounds    = Bounds(lb=b, ub=d)                # b_i <= x_i <= d_i

# --- initial guess (uniform, clipped into the box) ---------------------------
x0 = np.clip(np.full(n, S / n), b, d)

# --- solve -------------------------------------------------------------------
res = minimize(
    fun=objective,
    x0=x0,
    jac=objective_grad,
    method="trust-constr",
    constraints=[eq_constr],
    bounds=bounds,
    options={"verbose": 0},
)
x_opt = res.x

# --- report ------------------------------------------------------------------
print(f"\nTotal delivered         : {np.sum(x_opt):.2f} L/day")
print(f"Average per household   : {np.mean(x_opt):.2f} L/day")
print(f"Households below 100 L  : {int(np.sum(x_opt < 100.0))}")
print("\nHousehold allocations (L/day):")
for i in range(n):
    print(f"  Household {i + 1:2d}: {x_opt[i]:.2f}")
