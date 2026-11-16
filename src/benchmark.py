import argparse
import os
import random
import statistics
import sys
from time import perf_counter

from methods.exact import solve
from methods.sbpso import SBPSO
from process.dataset import Problem
from utils import checker

bks = {
    "instance_0001": 15.0,
    "instance_0002": 2.0,
    "instance_0003": 12.0,
    "instance_0004": 3.5,
    "instance_0005": 177.88,
    "instance_0006": 691.0,
    "instance_0007": 392.25,
    "instance_0008": 162.94,
    "instance_0009": 4.42,
    "instance_0010": 17.11,
    "instance_0011": 16.85,
    "instance_0012": 11.25,
    "instance_0013": 117.38,
    "instance_0014": 181.64,
    "instance_0015": 149.33,
    "instance_0016": 85.0,
    "instance_0017": 36.5,
    "instance_0018": 117.2,
    "instance_0019": 202.0,
    "instance_0020": 5.0,
}


def arguments():
    parser = argparse.ArgumentParser(
        description="Run the SBPSO on all the seeds listed in the provided file and save the results."
    )

    parser.add_argument(
        "instance", type=str, help="path to the folder containing the instances."
    )
    parser.add_argument("seed", type=str, help="path to the random seed file.")
    parser.add_argument(
        "--algorithm",
        choices=["bb", "sbpso"],
        required=True,
        type=str,
        help="algorithm to be executed (Branch-and-Bound or Set-Based PSO).",
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


def load_seeds(seed_path: str) -> list[int]:
    seeds = []

    try:
        with open(seed_path, "r") as file:
            for line_number, line in enumerate(file, start=1):
                line = line.strip()

                if not line:
                    continue

                try:
                    seed = int(line)
                except ValueError:
                    print(f"Error: invalid format on the line {line_number}: '{line}'")
                    sys.exit(1)

                seeds.append(seed)
        return seeds
    except FileNotFoundError:
        print(f"Error: file '{seed_path}' not found.")
        sys.exit(1)
    except OSError as e:
        print(f"Error opening the file: {e}")
        sys.exit(1)


def bench_sbpso(args: argparse.Namespace):
    seeds = load_seeds(args.seed)

    with open(f"./benchmarks/{args.algorithm}.csv", "+w") as file:
        file.write("instance,min value,max value,mean values,stdev,gap,mean time\n")
        file.close()

    for instance, best_obj in bks.items():
        print(f"Dataset: {instance}")

        times = []
        objectives = []

        start = perf_counter()
        for i in seeds:
            random.seed(i)

            problem = Problem(f"{args.instance}/{instance}.txt")
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
            problem.save_solution("./solution.txt")

            is_feasible, objective_value = checker.main(
                f"{args.instance}/{instance}.txt", "./solution.txt"
            )

            if not is_feasible or problem.result["objective"] != objective_value:
                print(
                    f"Something is wrong: {is_feasible}, {problem.result['objective']}, {objective_value}"
                )
                sys.exit(1)

            objectives.append(problem.result["objective"])
            times.append(problem.result["time"])

        end = perf_counter()
        print(f"Time: {end - start}\n")

        min_obj = min(objectives)
        max_obj = max(objectives)
        mean_obj = statistics.mean(objectives)
        stdev_obj = statistics.stdev(objectives)

        gap = round(100 * (best_obj - max_obj) / best_obj, 2)
        if gap == 0:
            gap = 0.00

        mean_time = statistics.mean(times)

        with open(f"./benchmarks/{args.algorithm}.csv", "+a") as file:
            file.write(
                f"{instance},{min_obj},{max_obj},{mean_obj},{stdev_obj},{gap},{mean_time}\n"
            )
            file.close()

    os.remove("./solution.txt")


def bench_bb(args):
    with open(f"./benchmarks/{args.algorithm}.csv", "+w") as file:
        file.write("instance,status,objective,gap,time\n")
        file.close()

    for instance, best_obj in bks.items():
        print(f"Dataset: {instance}")

        problem = Problem(f"{args.instance}/{instance}.txt")
        status = solve(problem)
        print(problem.result)

        if problem.result["objective"] != 0.0:
            problem.save_solution("./solution.txt")
            is_feasible, objective_value = checker.main(
                f"{args.instance}/{instance}.txt", "./solution.txt"
            )
            if not is_feasible or problem.result["objective"] != round(objective_value, 2):
                print(
                    f"Something is wrong: {is_feasible}, {problem.result['objective']}, {objective_value}"
                )
                sys.exit(1)

        print(f"Time: {problem.result['time']}\n")

        gap = round(100 * (best_obj - problem.result["objective"]) / best_obj, 2)
        if gap == 0:
            gap = 0.00

        with open(f"./benchmarks/{args.algorithm}.csv", "+a") as file:
            file.write(
                f"{instance},{status},{problem.result['objective']},{gap},{problem.result['time']}\n"
            )
            file.close()

    os.remove("./solution.txt")


if __name__ == "__main__":
    args = arguments()

    # Benchmark.
    if args.algorithm == "bb":
        bench_bb(args)
    else:
        bench_sbpso(args)
