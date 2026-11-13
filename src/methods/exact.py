from time import perf_counter

import pulp

from process.dataset import Problem


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
    problem: Problem, Uo: list[int], k: int
) -> tuple[pulp.LpProblem, list[pulp.LpVariable], list[pulp.LpVariable]]:
    """
    Construct the Order Batching problem formulated by MeLi, adapted to be linear; to do this, the number of aisles is fixed and passed as the parameter `k`.

    # Args:
        problem (Problem): Dataset information.
        Uo (list[int]): Total number of items in each order.
        k (int): Number of aisles.

    # Returns:
    The problem that remains unresolved and the variables associated with it. The orders are the first, and the aisles are the second.
    """

    prob = pulp.LpProblem(f"Order_Batching_with_{k}_aisles.", pulp.LpMaximize)

    x = [prob.add_variable(f"x_{o}", cat="Binary") for o in range(problem.o)]
    y = [prob.add_variable(f"y_{a}", cat="Binary") for a in range(problem.a)]

    # Objective function.
    prob += pulp.lpSum(Uo[o] * x[o] for o in range(problem.o))

    # Constraint 1 - respect the limits.
    prob += pulp.lpSum(Uo[o] * x[o] for o in range(problem.o)) >= problem.lb
    prob += pulp.lpSum(Uo[o] * x[o] for o in range(problem.o)) <= problem.ub

    # Constraint 2 - the aisles have the items to fulfill the orders.
    for i in range(problem.i):
        uoi = pulp.lpSum(problem.orders[o].get(i, 0) * x[o] for o in range(problem.o))
        uai = pulp.lpSum(problem.aisles[a].get(i, 0) * y[a] for a in range(problem.a))
        prob += uoi <= uai

    # Constraint 3 - the number of ailes must be equal to k.
    prob += pulp.lpSum(y[a] for a in range(problem.a)) == k

    return prob, x, y


def solve(problem: Problem):
    """
    Solve the |A| linear problems for Order Batching. The results are saved directly to the variable `problem`.

    # Args:
        problem (Problem): Dataset information.
    """

    Uo = [sum(problem.orders[o].values()) for o in range(problem.o)]
    optimal: tuple[float, list[int], list[int]] = (0.0, [], [])

    start = perf_counter()
    for k in range(1, problem.a + 1):
        prob, x, y = solve_k(problem, Uo, k)
        stats = prob.solve(pulp.COIN_CMD(msg=False))

        # Update the values if the result of the current LP is better.
        if stats.status_str == "Optimal":
            z = pulp.value(prob.objective)
            assert z is not None

            z /= k
            if z > optimal[0]:
                optimal = (
                    z,
                    [_bin_val(x[o]) for o in range(problem.o)],
                    [_bin_val(y[a]) for a in range(problem.a)],
                )
    end = perf_counter()

    problem.result["orders"] = optimal[1]
    problem.result["aisles"] = optimal[2]
    problem.result["objective"] = optimal[0]
    problem.result["time"] = end - start
