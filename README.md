# Open Problems in Constitutional Preference Reconstruction

## Setup

To set up this repository, first create a new virtualenv:

```
python3 -m venv venv
. ./venv/bin/activate
```

Then clone and install `treecai` and its dependencies:
```
git clone https://github.com/eleanor-clifford/treecai
pip install ./treecai
```

Finally, fill in your api keys in `alpaca_eval_secrets.yaml`, `openai_configs.yaml`, and `secrets.toml`

## Running

To run a single experiment, run:

```
treecai --config exp/config/some/config.yml
```

To run multiple experiments, a convenience script is provided which spawns multiple experiments, each in their own tmux window. E.g. run the following from inside a tmux session:

```
./scripts/tmux_spawn exp/config/some/config1.yml exp/config/some/config2.yml ...
```

Another script is used to save complete results for use by other configs:

```
./scripts/save_latest  # saves all completed runs
```

As some experiments depend on others, experiments should be run in the following order. Configurations are named this way for historical reasons.

```
./scripts/tmux_spawn exp/configs/**/treecai  # start base ICAI+ experiments
./scripts/tmux_spawn exp/configs/**/icai  # start base ICAI experiments

# wait for completion
./scripts/save_latest

./scripts/tmux_spawn exp/configs/**/trees/*  # start ICAI+ executors
./scripts/tmux_spawn exp/configs/**/icai_trees/*  # start ICAI executors
./scripts/tmux_spawn exp/configs/**/blend*  # start cross-llm experiments
./scripts/tmux_spawn exp/configs/**/intra_random.yml  # start intra-llm experiments

python3 exp/configs/ablations/updated_vs_not/generate.py  # generate configs for refinement ablation
./scripts/tmux_spawn exp/configs/ablations/**/*.yml  # start all ablations

# wait for completion
./scripts/save_latest
```

## Analysis

To run the analysis scripts, run:

```
./scripts/run_analysis
```

Unfortunately, due to time constraints and prioritising quality and correctness
of data reporting in the paper, these analysis scripts are much more brittle
than the main experiment code, and will only function correctly if the
experiments they are analysing have all been fully generated.

## Results

The final constitutions generated for the experiments in the main paper can be
seen in the exp/constitutions directory.
