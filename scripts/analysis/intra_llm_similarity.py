import numpy as np
import sys
import itertools
from os.path import dirname
from treecai.principles.lib import principle_similarity, get_selected_annotators
from pathlib import Path

sys.path.append(dirname(__file__))
import lib


def print_stats(agreements):
    for t, s in (
        ("μ cov", np.mean(agreements[:, 0])),
        *((
            ("σ cov", np.sqrt(np.var(agreements[:, 0]))),
            ("+ cov", lib.mean_ci(agreements[:, 0])),
        ) if agreements.shape[0] > 1 else tuple()),
        ("μ acc", np.mean(agreements[:, 1])),
        *((
            ("σ acc", np.sqrt(np.var(agreements[:, 1]))),
            ("+ acc", lib.mean_ci(agreements[:, 1])),
        ) if agreements.shape[0] > 1 else tuple()),
    ):
        print(f"{t}: {100*s:.1f}")


exp_suffix = "intra_random"
for dataset in ("train", "test"):

    agreement_stats = []
    all_agreements = []

    for exp_dir in Path("exp/saved_outputs").rglob(exp_suffix):
        pretty_config = str(exp_dir).removeprefix("exp/saved_outputs/").removesuffix(exp_suffix).removesuffix("/")
        ap_data = lib.get_ap_data(lib.get_ap_data_file(exp_dir, dataset))

        annotators = get_selected_annotators(ap_data["annotators"], default_to_all=True)
        votes = np.array([
            [
                v["pref"] or v.get("no_pref_reason")
                for v in c["annotations"][aid]["votes"]
            ]
            for c in ap_data["comparisons"]
            for aid in annotators.keys()
        ]).T

        agreements = np.mean(np.array([
            principle_similarity(votes[i], votes[j], coverage_set="all")
            for i, j in itertools.combinations(range(len(votes)), 2)
        ]), axis=0).reshape(1, -1)

        all_agreements.append(agreements)

        print(f"{dataset}/{pretty_config}/{len(annotators)} annotators:")
        print_stats(all_agreements[-1])
        print()


    print("===")
    print(f"{dataset}:")
    print_stats(np.concatenate(all_agreements, axis=0))
    print()
