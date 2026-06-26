#!/usr/bin/env python3

import numpy as np
import sys
from os.path import dirname

sys.path.append(dirname(__file__))
import lib


if __name__ == "__main__":
    for name, datas in (
        ("ICAI", list(lib.get_data(
            name="ICAI",
            paths="icai_trees",
            test_only=False,
        ))),
        ("ICAI+", list(lib.get_data(
            name="ICAI+",
            paths="trees",
            test_only=False,
        ))),
    ):

        bigtab = []
        print(name)
        print("===")
        print()
        for config, n_principles, dataset, data in datas:

            # ensure we are fair to invalid values
            data[np.logical_and(data != "a", data != "b")] = "a"

            names = ["base", "llm ", "majo", "prio", "lgbm", "dt"]
            agreement_matrix = np.full((len(names), len(names)), 1.0)

            for i, d1 in enumerate(data):
                for j, d2 in enumerate(data):
                    if i != j:
                        agreement_matrix[i, j] = agreement_matrix[j, i] = np.sum(d1 == d2) / len(d1)

            bigtab.append((
                dataset,
                config,
                n_principles,
                *agreement_matrix[0][1:],
                *(agreement_matrix[0][1] - agreement_matrix[0][2:]),
                *agreement_matrix[1][2:],
                *agreement_matrix[2][3:],
                *agreement_matrix[3][4:],
                *agreement_matrix[4][5:],
            ))

        # train comes before test
        bigtab = sorted([x for x in bigtab if "train" in x[0]]) + sorted([x for x in bigtab if "train" not in x[0]])

        header = (
            "dataset",
            "config",
            "n. principles",
            "mean coverage",
            "mean accuracy",
            "stddev coverage",
            "stddev accuracy",
            "llm",
            "majority",
            "priority",
            "lgbm",
            "dt",
            "Δ-maj",
            "Δ-pri",
            "Δ-lgbm",
            "Δ-dt",
            "llm x maj",
            "llm x pri",
            "llm x lgbm",
            "llm x dt",
            "maj x pri",
            "maj x lgbm",
            "maj x dt",
            "pri x lgbm",
            "pri x dt",
            "lgbm x dt",
        )

        lib.print_aligned(lib.str_align([header, tuple("---" for _ in header)] + bigtab))
        print()

        for dataset in reversed(sorted({x[0] for x in bigtab})):
            for head, col in list(zip(header, zip(*[x for x in bigtab if x[0] == dataset])))[3:]:
                print(f"{dataset}: avg. {head}: {100*np.mean(col):+.1f} +/- {100*lib.mean_ci(np.array(col)):.1f}")
            print()

            n_principless = {x[2] for x in bigtab}
            if len(n_principless) > 1:
                for n_principles in sorted(n_principless):
                    for head, col in list(zip(header, zip(*[x for x in bigtab if x[0] == dataset and x[2] == n_principles])))[3:]:
                        print(f"{dataset}: avg. {head}: {100*np.mean(col):+.1f} +/- {100*lib.mean_ci(np.array(col)):.1f}")
                print()
