# Sources and parser changes

- Official assignment: https://gist.github.com/neur-lab/38deabdfcde9e6dbacf362ab8059eb41
- Submission template: https://gist.github.com/neur-lab/483852e1f9d8d52f54627e600677c2f9
- Qwen evaluation recipe: https://github.com/QwenLM/Qwen3-VL#evaluation-reproduction
- Qwen model: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct
- vLLM chat API: https://docs.vllm.ai/en/latest/api/vllm/entrypoints/llm/
- MMMU parser source: https://github.com/MMMU-Benchmark/MMMU/blob/aa9b70da92c2825b3d544d1a11b36856bd92f6c3/mmmu/utils/eval_utils.py

`vendor/mmmu_eval_utils.py` is an unmodified copy from the commit above.
The upstream license is included as `vendor/MMMU_LICENSE`.

`mc_parse()` in `eval_mmmu.py` adapts the upstream multiple-choice parser:
the same bracket / standalone-letter / option-text matching order is used;
the last match is selected when several candidate options are found.
Differences: no-match returns `None` (scored wrong), not a random option;
empty option strings are not treated as matches; parse mode and candidate list are recorded.
Open-ended extraction, normalization and scoring use the unmodified upstream functions.
Empty model responses are always wrong. Predictions are never repaired using the gold answer.
Open-ended candidate lists are sorted only for stable JSON output; scoring is unchanged.

This is a documented team pipeline, not a claim to reproduce every detail of Qwen's internal evaluation.
Image labels and question templates are newly designed; both are saved in the run artifacts.
