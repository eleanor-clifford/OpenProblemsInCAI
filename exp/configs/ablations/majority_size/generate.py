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

models = {
    "gemini": "openrouter/google/gemini-2.5-flash",
    "deepseek": "openrouter/deepseek/deepseek-chat-v3.1",
    "gpt4o-mini": "openrouter/openai/gpt-4o-mini-2024-07-18",
    "gpt4o": "openrouter/openai/gpt-4o",
    "gpt5.4-mini": "openrouter/openai/gpt-5.4-mini",
    "gpt5.4-nano": "openrouter/openai/gpt-5.4-nano",
}

cluster_size = {
    1: 20,
    2: 20,
    5: 20,
    10: 80,
    20: 80,
    30: 100,
    40: 100,
    50: 100,
}

for dataset in train_data.keys():
    for model in models.keys():
        for num_principles in cluster_size.keys():
            base_input = dict(
                TRAIN_DATA=train_data[dataset],
                TEST_DATA=test_data[dataset],
                NUM_CLUSTERS=cluster_size[num_principles],
                MAX_PRINCIPLES=num_principles,
            )

            icai_input = dict(
                MODEL=models[model],
                EXP_DIR=f"exp/saved_outputs/{dataset}/{model}/icai_trees/majority_30_better",
            )

            plus_input = dict(
                EXP_DIR=f"exp/saved_outputs/{dataset}/{model}/icai_plus",
            )

            for base, ip in (
                ("icai_plus", plus_input),
                ("icai", icai_input),
            ):
                makedirs(dirname(__file__) + f"/{base}/{dataset}/{model}", exist_ok=True)
                open(dirname(__file__) + f"/{base}/{dataset}/{model}/{num_principles:02d}.yml", "w").write(
                    Template(open(dirname(__file__) + f"/{base}/base.yml").read()).substitute(
                        base_input | ip
                    )
                )
