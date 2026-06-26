import json
import numpy as np
import sys
from os.path import dirname
from treecai.principles.lib import hash_string
from pathlib import Path

sys.path.append(dirname(__file__))
import lib


def print_matrices(cfg_names, matrices, title, ci=None):
    if (novar := len(matrices.shape) == 3):
        matrices = np.expand_dims(matrices, 0)

    for i, mtype in enumerate(("coverage", "accuracy")):
        m = matrices[:, :, :, i]
        mn = np.mean(m, axis=0)

        def np_pctfmt(mtx):
            return np.char.mod('%.1f', 100 * mtx)

        str_mtx = np_pctfmt(mn)
        if not (ci is None and novar):
            _ci = lib.mean_ci(m) if ci is None else ci[:, :, i]
            str_mtx = str_mtx + ' +/- ' + np_pctfmt(_ci)  # no += because maximum string length must increase

        print(f"{title}/{mtype}")
        lib.print_aligned(lib.pretty_matrix(cfg_names, str_mtx))
        print()


print_last = ""

for name, run_sets in (
    ("ICAI", (("blend_random_icai",),)),
    ("ICAI+", (("blend_random",),)),
):
    print(name)
    print("===")
    print()
    for dataset in ("train", "test"):
        matrices = []
        models = []

        for config in ["alpaca_eval", "prism", "arena"]:
            annotation_configs = None
            flat_annotationss = {}
            for runs in run_sets:
                ap_datas = [
                    lib.get_ap_data(lib.get_ap_data_file(Path(f"exp/saved_outputs/{config}/{run}"), dataset))
                    for run in runs
                ]

                _annotation_configs = {}
                for apd in ap_datas:
                    for a in [c for a in apd["annotators"].values() if (c := a.get("annotation_config")) is not None]:
                        h = hash_string(json.dumps(a))
                        if h not in _annotation_configs:
                            _annotation_configs[h] = a

                if annotation_configs is None:
                    annotation_configs = _annotation_configs
                else:
                    assert annotation_configs == _annotation_configs

                annotatorss = {
                    cfg_id: {aid: annotator for apd in ap_datas for aid, annotator in apd["annotators"].items() if annotator.get("annotation_config") == a_cfg}
                    for cfg_id, a_cfg in annotation_configs.items()
                }

                principless = [
                    [a["description"] for a in b.values()]
                    for b in annotatorss.values()
                ]

                pids = [[hash_string(p) for p in ps] for ps in principless]
                assert all(p == pids[0] for p in pids)

                assert all(x == principless[0] for x in principless)
                assert all(len(apd["comparisons"]) == len(ap_datas[0]["comparisons"]) for apd in ap_datas)
                assert all(apd["comparisons"][i]["id"] == ap_datas[0]["comparisons"][i]["id"] for apd in ap_datas for i in range(len(apd["comparisons"])))

                annotationss = {
                    cfg_id: [
                        [
                            {k: v for apd in ap_datas for k, v in apd["comparisons"][i]["annotations"].items()}[aid]["pref"]
                            for i in range(len(ap_datas[0]["comparisons"]))
                        ]
                        for aid in annotators.keys()
                    ]
                    for cfg_id, annotators in annotatorss.items()
                }

                flat_annotationss = {
                    cfg_id: np.concatenate([flat_annotationss.get(cfg_id, []), *annotations])
                    for cfg_id, annotations in annotationss.items()
                }

            assert all(len(v) == len(list(flat_annotationss.values())[0]) for v in flat_annotationss.values())

            agreement_matrix = np.full((len(annotation_configs), len(annotation_configs), 2), 1.0)
            ci_matrix = np.full((len(annotation_configs), len(annotation_configs), 2), 0.0)

            for i, d1 in enumerate(flat_annotationss.values()):
                for j, d2 in enumerate(flat_annotationss.values()):
                    if i != j:
                        agreement_matrix[i, j], ci_matrix[i, j] = agreement_matrix[j, i], ci_matrix[j, i] = lib.principle_similarity_ci(d1, d2)

            matrices.append(agreement_matrix)
            models.append([cfg["model"].split('/')[-1] for cfg in annotation_configs.values()])
            print_matrices(models[-1], agreement_matrix, f"{config}/{dataset}", ci_matrix)

        assert all(m == models[0] for m in models)
        assert all(m.shape == matrices[0].shape for m in matrices)

        matrices = np.array(matrices)
        print_matrices(models[-1], matrices, f"{dataset}")
        assert matrices.shape[0] * matrices.shape[1] == 18
        mtx_tril = matrices.transpose(1, 2, 0, 3)[np.tril_indices_from(matrices[0, :, :, 0], -1)].reshape(-1, 2)
        full_mean = np.char.mod('%.1f', 100 * np.mean(mtx_tril, axis=0))
        full_ci = np.char.mod('%.1f', 100 * lib.mean_ci(mtx_tril, axis=0, df=(matrices.shape[0] * matrices.shape[1] - 1)))
        print_last += f"{name}/{dataset} cov. agreement: {full_mean[0]} +/- {full_ci[0]}\n"
        print_last += f"{name}/{dataset} vote agreement: {full_mean[1]} +/- {full_ci[1]}\n"

print(print_last)
