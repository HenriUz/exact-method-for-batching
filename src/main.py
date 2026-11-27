import argparse
from random import seed

from methods.exact import Cuts, solve
from methods.sbpso import SBPSO
from process.dataset import Problem


def arguments():
    parser = argparse.ArgumentParser(
        description="Exact and Heuristic Methods for Order Batching."
    )

    parser.add_argument("instance", type=str, help="Path to the instance.")
    parser.add_argument(
        "--algorithm",
        choices=["bb", "sbpso"],
        required=True,
        type=str,
        help="algorithm to be executed (Branch-and-Bound or Set-Based PSO).",
    )
    parser.add_argument(
        "--cuts",
        choices=[0, 1, 2, 3],
        type=int,
        default=Cuts.BOTH,
        help="cuts to be made (0: no one; 1: MIP only; 2: ours only; 3: both) - optional for Branch-and-Bound, with a default value of 3.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="random seed - optional for SBPSO, with a default value of 0.",
    )
    parser.add_argument(
        "--size",
        type=int,
        default=50,
        help="swarm size - optional for SBPSO, with a default value of 50.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=600,
        help="maximum number of iterations - optional for SBPSO, with a default value of 600.",
    )
    parser.add_argument(
        "--c1",
        type=float,
        default=0.9297,
        help="coefficient 1 - optional for the SBPSO, with a default value of 0.9297.",
    )
    parser.add_argument(
        "--c2",
        type=float,
        default=0.2266,
        help="coefficient 2 - optional for the SBPSO, with a default value of 0.2266.",
    )
    parser.add_argument(
        "--c3",
        type=float,
        default=1.3086,
        help="coefficient 3 - optional for the SBPSO, with a default value of 1.3086.",
    )
    parser.add_argument(
        "--c4",
        type=float,
        default=2.1526,
        help="coefficient 4 - optional for the SBPSO, with a default value of 2.1526.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=7,
        help="tournament - optional for SBPSO with a default value of 7.",
    )

    return parser.parse_args()


def main():
    args = arguments()

    seed(args.seed)

    problem = Problem(args.instance)
    if args.algorithm == "bb":
        status, n_calls, n_cuts = solve(problem, Cuts(args.cuts))
        print(status, n_calls, n_cuts)
    else:
        SBPSO(
            problem,
            args.size,
            args.iterations,
            args.c1,
            args.c2,
            args.c3,
            args.c4,
            args.k,
        )

    print(problem.result)


if __name__ == "__main__":
    main()
