import json
import numpy as np
from constree.principles.lib import get_selected_annotators, log
from string import Template
from os.path import dirname
from os import makedirs

train_data = {
    "alpaca_eval": "data/alpaca_eval_train.json",
    "arena": "data/processed/misc/arena2025_arena-human-preference-140k_5000chars_english_no-sep-prompt_800samples_train_ap.json",
    "prism": "data/processed/misc/prism_800samples_train_ap.json",
}

test_data = {
    "alpaca_eval": "data/alpaca_eval_test.json",
    "arena": "data/processed/misc/arena2025_arena-human-preference-140k_5000chars_english_no-sep-prompt_400samples_test_ap.json",
    "prism": "data/processed/misc/prism_400samples_test_ap.json",
}

np.random.seed(0)

for dataset in ("alpaca_eval", "prism", "arena"):
    for model in ("gemini", "deepseek", "gpt4o-mini", "gpt4o", "gpt5.4-mini", "gpt5.4-nano"):
        exp_dir = dirname(__file__) + f"/../../../saved_outputs/{dataset}/{model}/icai_plus"

        with open(f"{exp_dir}/step2_annotation_ap.json") as f:
            ap_data = json.load(f)

        annotators = get_selected_annotators(ap_data["annotators"])

        descendants = {
            k: v
            for k, v in annotators.items()
            if "parent" in v["discovery"]
        }

        original = {
            k: v
            for k, v in annotators.items()
            if "parent" not in v["discovery"]
        }

        select = 12
        if len(descendants) < select:
            log.warning(f"Fewer improved principles, decreasing selection num to {len(descendants)} for {dataset}/{model}")
            select = len(descendants)
        if len(original) < select:
            log.warning(f"Fewer unimproved principles, decreasing selection num to {len(original)} for {dataset}/{model}")
            select = len(original)

        for items, name in ((original, "filtered_only"), (descendants, "improved_only"), (annotators, "all")):
            makedirs(dirname(__file__) + f"/{dataset}/{model}", exist_ok=True)
            open(dirname(__file__) + f"/{dataset}/{model}/{name}.yml", "w").write(
                Template(open(dirname(__file__) + "/base.yml").read()).substitute(
                    TRAIN_DATA=train_data[dataset],
                    TEST_DATA=test_data[dataset],
                    EXP_DIR=f"exp/saved_outputs/{dataset}/{model}/icai_plus",
                    ALLOWED_IDS = str([str(x) for x in np.random.choice(list(items.keys()), select, replace=False)])
                )
            )
