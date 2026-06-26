import sys
from os.path import dirname
from pathlib import Path
from scipy.stats import spearmanr
from tqdm import tqdm
from os import environ
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText

environ["LOG"] = environ.get("LOG", "WARN")  # suppress annoying treecai log
from treecai.exp.config import PrincipleSelectionConfig
from treecai.exp.stages.generate_trees import select_best_principles
from treecai.principles.lib import hash_string

sys.path.append(dirname(__file__))
import lib

gens = {"icai": "icai_trees", "icai+": "trees"}
datasets = ("train", "test")
# exps = ("llm_30_majority_selected", "majority_30_better",)
exps = ("llm_30_majority_selected", "majority_30_better", "priority_30_majority_selected", "lightgbm_tuned", "dt_tuned")
stat_keys = ("accuracy", "relevance", "real_accuracy")

ensure_total = 18

runs = {generation: list(Path("exp/saved_outputs").rglob(gendir)) for generation, gendir in gens.items()}
n_runs = sum(len(x) for x in runs.values())

tab = [("dataset", "statistic", "average", "executor", "generation", "n", "stat", "pval")]

# for n_principles in (30, None):
for n_principles in (30,):
    ap_data = {}
    bar = tqdm(total=len(datasets) * n_runs, desc=f"Getting intial data for {n_principles} principles")
    for dataset in datasets:
        ap_data[dataset] = {}
        for generation, gendir in gens.items():
            for p in runs[generation]:
                _ap_data = lib.get_ap_data(lib.get_ap_data_file(p / "majority_30_better", dataset))

                if n_principles:
                    assert n_principles == 30
                    # get same best ones for fair comparison
                    _aids = set(hash_string(p) for v in (v for v in _ap_data["annotators"].values() if v["type"] == "constitution") for p in v["constitution"])
                    _ap_data["annotators"] = {
                        aid: annotator for aid, annotator in _ap_data["annotators"].items()
                        if aid in _aids
                    }
                ap_data[dataset][p] = _ap_data
                bar.update(1)

    bar = tqdm(total=len(datasets) * len(exps) * (n_runs + len(gens)), desc=f"Processing for {n_principles} principles")
    for dataset in datasets:
        for i, exp in enumerate(exps):
            for generation, gendir in gens.items():
                aggregate_stats = {}
                for p in runs[generation]:
                    constitution_annotations = lib.get_annotations(lib.get_ap_data_file(p / exp, dataset), "constitution")
                    pretty_path = str(p).removeprefix("exp/saved_outputs/").removesuffix(f"/{gendir}")
                    stats = lib.get_aggregate_stats_from_ap_data(ap_data[dataset][p], stat_keys, constitution_annotations)

                    if n_principles:
                        if stats["n_principles"] != n_principles:
                            print(stats["n_principles"], n_principles)
                        assert stats["n_principles"] == n_principles
                    else:
                        # sanity check
                        if stats["n_principles"] != (100 if generation == "icai" else 50):
                            print(stats["n_principles"])
                        assert stats["n_principles"] == (100 if generation == "icai" else 50)

                    for k, v in stats.items():
                        if i == 0 and k not in ("agreement", "n_principles"):
                            bar.write(f"{dataset}/{generation}/{n_principles}/{pretty_path}/{k}: {100*v:.1f}")
                        aggregate_stats[k] = aggregate_stats.get(k, [])
                        aggregate_stats[k].append(v)
                    bar.update(1)

                for avg in ("mean", "median"):
                    for stat_key in stat_keys:
                        x = aggregate_stats[f"{avg}_{stat_key}"]
                        y = aggregate_stats["agreement"]
                        stat, pval = spearmanr(x, y)

                        assert len(set(aggregate_stats["n_principles"])) == 1

                        assert len(x) == len(y)
                        if ensure_total:
                            assert len(x) == ensure_total

                        tab.append((dataset, stat_key, avg, exp, generation, aggregate_stats["n_principles"][0], f"{stat:+.3f}", f"{pval:.3f}"))
                        plt.figure()
                        plt.scatter(x, y)
                        plt.gca().add_artist(AnchoredText(f"$\\rho_s$ = {stat:+.3f}\np = {pval:.3f}", loc="upper left"))
                        plt.tight_layout()
                        plt.savefig(f"plots/correlation/{dataset}_{n_principles}_{stat_key}_{avg}_vs_agreement_{exp}_{generation}.pdf")
                        plt.close()

                bar.update(1)  # allow for time spent plotting

    bar.close()
lib.print_aligned(lib.str_align(tab))
