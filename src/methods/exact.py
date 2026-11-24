from enum import Enum
from time import perf_counter
from typing import cast

import mip

from process.dataset import Problem

TIMEOUT = 600
EPS = 1e-6


class Cuts(Enum):
    NO_ONE = 0
    MIP_ONLY = 1
    OURS_ONLY = 2
    BOTH = 3


def _bin_val(v: mip.Var) -> int:
    """
    Sanity check for the variables.

    # Args:
        v (Var): Problem variable.

    # Returns:
    The value stored in the variable.
    """

    assert v.x is not None
    return round(v.x)


def solve_k(
    problem: Problem, Uo: list[int], k: int, z: tuple[mip.Numeric, int], cuts: Cuts
) -> tuple[mip.Model, list[mip.Var], list[mip.Var]]:
    """
    Construct the Order Batching problem formulated by MeLi, adapted to be linear; to do this, the number of aisles is fixed and passed as the parameter `k`.

    In terms of model optimization, we know that for the current `k`, the solution will only be worthwhile if the number of items in the selected orders is greater than the value of the best solution found so far multiplied by `k`. This leads to the LB constraint, where LB becomes `z' * k`.

    # Args:
        problem (Problem): Dataset information.
        Uo (list[int]): Total number of items in each order.
        k (int): Number of aisles.
        z (tuple[Numeric, int]): Numerator and number of aisles in the best solution found.
        cuts (Cuts): Cuts to be implemented.

    # Returns:
    The problem that remains unresolved and the variables associated with it. The orders are the first, and the aisles are the second.
    """

    prob = mip.Model(f"Order Batching with {k} aisles.", mip.MAXIMIZE)

    x = [prob.add_var(f"x_{o}", var_type=mip.BINARY) for o in range(problem.o)]
    y = [prob.add_var(f"y_{a}", var_type=mip.BINARY) for a in range(problem.a)]

    # Objective function.
    prob.objective = mip.xsum(Uo[o] * x[o] for o in range(problem.o))

    # Constraint 1 - respect the limits.
    prob += mip.xsum(Uo[o] * x[o] for o in range(problem.o)) >= max(
        problem.lb, z[0] * k // z[1] + 1
    )
    prob += mip.xsum(Uo[o] * x[o] for o in range(problem.o)) <= problem.ub

    # Constraint 2 - the aisles have the items to fulfill the orders.
    for i in range(problem.i):
        uoi = mip.xsum(problem.orders[o].get(i, 0) * x[o] for o in range(problem.o))
        uai = mip.xsum(problem.aisles[a].get(i, 0) * y[a] for a in range(problem.a))
        prob += uoi <= uai

    # Constraint 3 - the number of aisles must be equal to k.
    prob += mip.xsum(y[a] for a in range(problem.a)) == k

    # Cuts.
    if cuts in (Cuts.BOTH, Cuts.OURS_ONLY):
        prob.cuts_generator = CoverCutGenerator(x, Uo, problem.ub)

    return prob, x, y


def solve(problem: Problem, cuts: Cuts) -> str:
    """
    Solve the |A| integer problems for Order Batching. The results are saved directly to the variable `problem`.

    # Args:
        problem (Problem): Dataset information.
        cuts (Cuts): Cuts to be implemented.

    # Returns:
        The status of the solution found: Optimal, Not Solved, Feasible.
    """

    Uo = [sum(problem.orders[o].values()) for o in range(problem.o)]
    optimal: tuple[tuple[mip.Numeric, int], list[int], list[int]] = ((0, 1), [], [])

    n_calls = 0
    n_cuts = 0

    k = 1
    status = "Not Solved"
    start = perf_counter()
    while k <= problem.a:
        remaining = TIMEOUT - (perf_counter() - start)
        if remaining <= 0:
            if optimal[0][0] > 0:
                status = mip.OptimizationStatus.FEASIBLE.name
            break

        # The best possible value for z* is UB, so if the ratio UB / k is less than or equal to the best value of z found, no other value will be better.
        if problem.ub * optimal[0][1] <= optimal[0][0] * k:
            break

        prob, x, y = solve_k(problem, Uo, k, optimal[0], cuts)
        prob.verbose = 0

        if cuts in (Cuts.NO_ONE, Cuts.OURS_ONLY):
            prob.cuts = 0

        stats = prob.optimize(max_seconds=remaining)

        # Update the values if the result of the current LP is better.
        if stats in (mip.OptimizationStatus.OPTIMAL, mip.OptimizationStatus.FEASIBLE):
            status = stats.name
            z = prob.objective_value
            assert z is not None

            z = round(z)
            if z * optimal[0][1] > optimal[0][0] * k:
                optimal = (
                    (z, k),
                    [_bin_val(x[o]) for o in range(problem.o)],
                    [_bin_val(y[a]) for a in range(problem.a)],
                )

        if cuts in (Cuts.BOTH, Cuts.OURS_ONLY):
            prob.cuts_generator = cast(CoverCutGenerator, prob.cuts_generator)
            n_calls += prob.cuts_generator.n_calls
            n_cuts += prob.cuts_generator.n_cuts

        k += 1
    end = perf_counter()

    problem.result["orders"] = [i for i, valor in enumerate(optimal[1]) if valor == 1]
    problem.result["aisles"] = [i for i, valor in enumerate(optimal[2]) if valor == 1]
    problem.result["objective"] = round(optimal[0][0] / optimal[0][1], 2)
    problem.result["time"] = end - start

    if cuts in (Cuts.BOTH, Cuts.OURS_ONLY):
        print(f"Generate Covers Calls: {n_calls}.")
        print(f"Cuts found: {n_cuts}")

    return status


def generate_covers(
    pairs: list[tuple[int, float]], Uo: list[int], ub: int
) -> list[int] | None:
    """
    Given the fractional solution (only variables in the node), try to find a minimally violated covering.

    # Args:
    - pairs (list[tuple[int, float]]): orders indices and fractional value.
    - Uo (list[int]): Total number of items in each order.
    - ub (int): Upper bound.

    # Returns:
    List of order indices (`C`), or `None` if none is found.
    """

    xs = dict(pairs)

    # Order indices, from the largest x* to the smallest
    indices = sorted(xs, key=lambda o: xs[o], reverse=True)
    print(indices)

    # Accumulate Uo until it exceeds ub.
    C = []
    uo = 0
    for o in indices:
        C.append(o)
        uo += Uo[o]
        if uo > ub:
            break
    else:
        return None

    # Attempts to remove from the SMALLEST x* to the LARGEST.
    for o in sorted(C, key=lambda o: xs[o]):
        if uo - Uo[o] > ub:
            C.remove(o)
            uo -= Uo[o]

    if sum(xs[o] for o in C) > len(C) - 1 + EPS:
        return C
    return None


class CoverCutGenerator(mip.ConstrsGenerator):
    def __init__(self, x: list[mip.Var], Uo: list[int], ub: int):
        self.x = x
        self.Uo = Uo
        self.ub = ub
        self.n_calls = 0
        self.n_cuts = 0

    def generate_constrs(self, model: mip.Model, depth: int = 0, npass: int = 0):
        self.n_calls += 1

        node: dict[int, mip.Var] = {}
        for o, v in enumerate(self.x):
            t = model.translate(v)
            if t is not None:
                node[o] = cast(mip.Var, t)

        pairs: list[tuple[int, float]] = []
        for o, v in node.items():
            val = v.x
            assert val is not None
            pairs.append((o, val))

        C = generate_covers(pairs, self.Uo, self.ub)
        if C is None:
            return

        self.n_cuts += 1
        model.add_cut(mip.xsum(node[o] for o in C) <= len(C) - 1)
