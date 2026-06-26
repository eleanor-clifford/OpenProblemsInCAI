from os import listdir
from os.path import isfile
from typing import Literal
from pathlib import Path
from tqdm import tqdm
from scipy import stats
import ast
import json
import numpy as np

from treecai.lib import log
from treecai.data.alpaca_eval import get_ap_data_from_alpaca_annotations
from treecai.principles.lib import flat_annotations, principle_stats, get_selected_annotators


def parse_alpaca_content(s):
    return ast.literal_eval(ast.parse(s).body[0].value.elts[1])["content"]


def alpaca_pref_to_ap_pref(pref):
    return {1: "a", 2: "b", -1: None}[int(pref)]


def get_ap_data_from_treecai_alpaca_dir(
    treecai_dir: str | Path,
    alpaca_dataset,
    annotator_config,
    base_ap_data=None,
    keep_annotators: set | Literal["all"] = "all"
):
    aa_topdir = Path(treecai_dir) / f"icai/full_classic_run/out/tmp/{alpaca_dataset}/annotator_configs/"
    valid_annotators = [x for x in listdir(str(aa_topdir)) if "constitutional" in x]
    if len(valid_annotators) == 0:
        return None
    assert len(valid_annotators) == 1
    aa_annotator_name = valid_annotators[0]
    full_aa_path = aa_topdir / aa_annotator_name / "annotations_seed0_configs.json"
    return get_ap_data_from_alpaca_annotations(full_aa_path, annotator_config, base_ap_data, keep_annotators)


def get_ap_data_file(treecai_dir, dataset):
    f = f"{treecai_dir}/step4_{dataset}_w_new_annotations_ap.json"
    if dataset == "train" and not isfile(f):
        if isfile(f2 := f"{treecai_dir}/step2_annotation_ap.json"):
            return f2
    return f


def get_ap_data(ap_data_file: str):
    if not isfile(ap_data_file):
        log.error(f"{ap_data_file} does not exist")
        exit(1)

    with open(ap_data_file) as f:
        return json.load(f)


def get_annotations(ap_data: str | dict, annotator_type):
    if isinstance(ap_data, str):
        ap_data = get_ap_data(ap_data)

    if annotator_type == "default":
        aid = ap_data["metadata"]["default_annotator"]
    else:
        aids = [aid for aid, annotator in ap_data["annotators"].items() if annotator["type"] == annotator_type]
        assert len(aids) == 1
        aid = aids[0]

    return [c["annotations"][aid]["pref"] for c in ap_data["comparisons"]]


def get_data(name, paths, get_llm_from_alpaca=None, test_only=True):
    n_principless = (30,)
    datasets = [] if test_only else [("train", "trainset")]
    datasets.append(("test", "testset-0"))
    exp_dirs = []
    for p in Path("exp/saved_outputs").rglob(paths):
        if "/ablations/" not in str(p):
            exp_dirs.append(p)

    for base_dir in tqdm(exp_dirs, desc=f"Processing {name} results"):
        for n_principles in n_principless:
            for dataset, aa_dataset in datasets:
                pretty_config = str(base_dir).removeprefix("exp/saved_outputs/").removesuffix(f"/{paths}")

                aggregate_stats = [
                    get_annotations(get_ap_data_file(base_dir / "dt_tuned", dataset), "default")
                ]
                annotationss = [
                    get_annotations(get_ap_data_file(base_dir / "dt_tuned", dataset), "default"),
                ]
                if get_llm_from_alpaca:
                    llm_ap_data = get_ap_data_from_treecai_alpaca_dir(
                        base_dir.parent / get_llm_from_alpaca.format(n_principles),
                        aa_dataset,
                        {"type": "constitution"},
                        get_ap_data(get_ap_data_file(base_dir.parent / get_llm_from_alpaca.format(n_principles), dataset)),
                    )
                    annotationss.append(get_annotations(llm_ap_data, "constitution"))
                else:
                    annotationss.append(get_annotations(get_ap_data_file(base_dir / f"llm_{n_principles}_majority_selected", dataset), "constitution"))

                for t in f"majority_{n_principles}_better", f"priority_{n_principles}_majority_selected", "lightgbm_tuned", "dt_tuned":
                    annotationss.append(get_annotations(get_ap_data_file(base_dir / t, dataset), "constitution"))

                assert all([len(x) == len(annotationss[0]) for x in annotationss])
                prefss = []

                for i in range(len(annotationss[0])):
                    prefss.append(tuple(x[i] for x in annotationss))

                yield pretty_config, n_principles, dataset, np.array(prefss).T


def get_aggregate_stats_from_ap_data(ap_data: dict, stat_keys, constitution_annotations=None):
    human_annotations = np.array(get_annotations(ap_data, "default"))
    constitution_annotations = constitution_annotations or np.array(get_annotations(ap_data, "constitution"))
    constitution_annotations[np.logical_and(constitution_annotations != "a", constitution_annotations != "b")] = "a"

    comparisons = {c["id"]: c for c in ap_data["comparisons"]}

    selected_principles = get_selected_annotators(ap_data["annotators"], default_to_all=True)

    all_principle_stats = {
        aid: principle_stats(human_annotations, flat_annotations(comparisons, aid))
        for aid, annotator in selected_principles.items()
    }

    return dict(
        agreement = np.sum(human_annotations == constitution_annotations) / len(human_annotations),
        n_principles = len(all_principle_stats),
        **{
            f"mean_{k}": np.mean([x[k] for x in all_principle_stats.values()])
            for k in stat_keys
        },
        **{
            f"median_{k}": np.median([x[k] for x in all_principle_stats.values()])
            for k in stat_keys
        },
        **{
            f"stddev_{k}": np.sqrt(np.var([x[k] for x in all_principle_stats.values()]))
            for k in stat_keys
        },
    )



def str_align(lines, sep="  ", maps=dict()):
    maps = {
        # str: lambda v: re.sub(r"^(\s*)-(0)", r"\1\2", v),
        float: "{:+.3f}",
    } | maps

    def map_fn(val):
        for k, v in maps.items():
            if isinstance(val, k):
                if isinstance(v, str):
                    return v.format(val)
                else:
                    return v(val)
        return str(val)

    lines = [[map_fn(col) for col in line] for line in lines]
    widths = np.max([[len(col) for col in line] for line in lines], axis=0)
    aligned = [[col + (w - len(col)) * " " for w, col in zip(widths, line)] for line in lines]
    return aligned


def pretty_matrix(names, matrix, maps={}):
    return str_align(
        [("", *names)] +
        [(name, *line) for name, line in zip(names, matrix)],
        maps=maps,
    )


def print_aligned(aligned, *args, **kwargs):
    for line in aligned:
        print(*line, sep="  ", *args, **kwargs)


def mean_ci(array: np.ndarray, axis=0, df=None):
    n = array.shape[axis]
    return stats.t.ppf(0.975, df=(df or (n - 1))) * stats.sem(array, axis=axis)


def principle_similarity_ci(
    first_principle_annotations: np.ndarray,
    second_principle_annotations: np.ndarray,
):
    if first_principle_annotations.shape != second_principle_annotations.shape:
        raise ValueError("number of annotations must match!")

    first_coverage = np.logical_or(first_principle_annotations == "a", first_principle_annotations == "b")
    second_coverage = np.logical_or(second_principle_annotations == "a", second_principle_annotations == "b")

    coverage_set = np.logical_and(first_coverage, second_coverage)
    coverage_agreement = first_coverage == second_coverage
    vote_agreement = first_principle_annotations[coverage_set] == second_principle_annotations[coverage_set]

    return (np.mean(coverage_agreement), np.mean(vote_agreement)), (mean_ci(coverage_agreement), mean_ci(vote_agreement))
