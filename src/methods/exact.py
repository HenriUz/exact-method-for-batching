from math import floor
from time import perf_counter

import pulp

from process.dataset import Problem

TIMEOUT = 600


def _bin_val(v: pulp.LpVariable) -> int:
    """
    Sanity check for the variables.

    # Args:
        v (LpVariable): Problem variable.

    # Returns:
    The value stored in the variable.
    """

    r = v.value()
    assert r is not None
    return round(r)


def solve_k(
    problem: Problem, Uo: list[int], k: int, z: float
) -> tuple[pulp.LpProblem, list[pulp.LpVariable], list[pulp.LpVariable]]:
    """
    Construct the Order Batching problem formulated by MeLi, adapted to be linear; to do this, the number of aisles is fixed and passed as the parameter `k`.

    In terms of model optimization, we know that for the current `k`, the solution will only be worthwhile if the number of items in the selected orders is greater than the value of the best solution found so far multiplied by `k`. This leads to the LB constraint, where LB becomes `z' * k`.

    # Args:
        problem (Problem): Dataset information.
        Uo (list[int]): Total number of items in each order.
        k (int): Number of aisles.
        z (float): Value of the best solution found so far.

    # Returns:
    The problem that remains unresolved and the variables associated with it. The orders are the first, and the aisles are the second.
    """

    prob = pulp.LpProblem(f"Order_Batching_with_{k}_aisles.", pulp.LpMaximize)

    x = [prob.add_variable(f"x_{o}", cat="Binary") for o in range(problem.o)]
    y = [prob.add_variable(f"y_{a}", cat="Binary") for a in range(problem.a)]

    # Objective function.
    prob += pulp.lpSum(Uo[o] * x[o] for o in range(problem.o))

    # Constraint 1 - respect the limits.
    prob += pulp.lpSum(Uo[o] * x[o] for o in range(problem.o)) >= max(
        problem.lb, floor(z * k) + 1
    )
    prob += pulp.lpSum(Uo[o] * x[o] for o in range(problem.o)) <= problem.ub

    # Constraint 2 - the aisles have the items to fulfill the orders.
    for i in range(problem.i):
        uoi = pulp.lpSum(problem.orders[o].get(i, 0) * x[o] for o in range(problem.o))
        uai = pulp.lpSum(problem.aisles[a].get(i, 0) * y[a] for a in range(problem.a))
        prob += uoi <= uai

    # Constraint 3 - the number of aisles must be equal to k.
    prob += pulp.lpSum(y[a] for a in range(problem.a)) == k

    return prob, x, y


def solve(problem: Problem) -> str:
    """
    Solve the |A| integer problems for Order Batching. The results are saved directly to the variable `problem`.

    # Args:
        problem (Problem): Dataset information.

    # Returns:
        The status of the solution found: Optimal, Not Solved, Time Limit.
    """

    Uo = [sum(problem.orders[o].values()) for o in range(problem.o)]
    optimal: tuple[float, list[int], list[int]] = (0.0, [], [])

    k = 1
    status = "Not Solved"
    start = perf_counter()
    while perf_counter() - start <= TIMEOUT and k <= problem.a:
        # The best possible value for z* is UB, so if the ratio UB / k is less than or equal to the best value of z found, no other value will be better.
        if problem.ub / k <= optimal[0]:
            break

        prob, x, y = solve_k(problem, Uo, k, optimal[0])
        stats = prob.solve(
            pulp.COIN_CMD(msg=False, timeLimit=TIMEOUT - (perf_counter() - start))
        )

        # Update the values if the result of the current LP is better.
        if stats.status_str == "Optimal" or stats.status_str == "Time Limit":
            status = stats.status_str
            z = pulp.value(prob.objective)
            assert z is not None

            z /= k
            if z > optimal[0]:
                optimal = (
                    z,
                    [_bin_val(x[o]) for o in range(problem.o)],
                    [_bin_val(y[a]) for a in range(problem.a)],
                )
        k += 1
    end = perf_counter()

    problem.result["orders"] = optimal[1]
    problem.result["aisles"] = optimal[2]
    problem.result["objective"] = round(optimal[0], 2)
    problem.result["time"] = end - start

    return status
