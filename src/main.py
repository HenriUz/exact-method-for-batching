import argparse


def arguments():
    parser = argparse.ArgumentParser(
        description="Exact and Heuristic Methods for Order Batching."
    )

    parser.add_argument("instance", type=str, help="Path to the instance.")
    parser.add_argument(
        "--algorithm",
        choices=["b&b", "sbpso"],
        required=True,
        type=str,
        help="algorithm to be executed (Branch-and-Bound or Set-Based PSO).",
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


if __name__ == "__main__":
    main()
