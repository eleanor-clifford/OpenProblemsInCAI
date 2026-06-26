## LMArena experiments

Experiment on LMArena (formerly known as "Chatbot Arena") data of pairwise human preference votes over text outputs by popular LLMs. Original dataset was collected in 2025 and [released on HuggingFace](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k).

### Data prepocessing

The file name describes the pre-processing done to the dataset:

```
data/processed/misc/arena2025_arena-human-preference-140k_5000chars_english_no-sep-prompt_5000samples_test_ap.json
```

Let's break this down:
- `arena2025`: our own internal name for this dataset
- `arena-human-preference-140k`: huggingface name of dataset ([see here](https://huggingface.co/datasets/lmarena-ai/arena-human-preference-140k))
- `5000chars`: maximum length of conversations in characters, any longer conversations are truncated. This helps avoid errors during annotation due to running out of available tokens (and reduces cost). For each datapoint, there is metadata included to indicate whether the conversation was truncated.
- `english`: to best work with our english prompts, we selected english conversations based on the originally included language metadata. Note that this metadata is not perfect, and sometimes conversations that just include one english word but otherwise are in a different language are also included.
- `no-sep-prompt`: the dataset does not provide a separate prompt, but rather includes the initial user prompt and subsequent user inputs as part of the responses. This setup is to treat user input equally, whether it is the initial prompt or subsequent user responses.
- `5000samples`: 5000 randomly sampled datapoints (of datapoints satisfying prior requirements)
- `test`: test dataset
- `ap.json`: AnnotatedPairs json format

Notebook used for data preparation: [link](https://github.com/rdnfn/feedback-forensics/blob/1c11c6d51059afd97699efa5bbb4983215efa6ba/notebooks/06_lmarena2025-140k_dataprep.ipynb).