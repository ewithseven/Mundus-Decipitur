# When Every Monitor Scores 1.0

Code and monitor outputs for the report **"When Every Monitor Scores 1.0: Why a Collusion Benchmark's Stego Tier Cannot Separate Text Monitors from Activation Probes"**. The report (`report/report.pdf`) was written for the Apart Research Swarm Alignment Hackathon, November 2026.

Author: ewithseven, independent researcher. Prior work: [Memento Mori](https://github.com/ewithseven/Memento-Mori).

## What is here

| Folder | Report section | What it does |
|---|---|---|
| `transfer_plain_monitor/` | 4.3 | Plain Claude Sonnet 5.5 monitor on NARCBench Transfer (13 + 2 scenario pairs). Includes the prompt (`duz_istem.md`), the case builder, scoring scripts, and all monitor outputs. |
| `stego_monitors/` | 4.1 (Table 1), 4.3 | Stego judges: the paper's naive and informed prompts, asked for a verdict or for a probability. Runs Sonnet 5.5, qwen2.5:7b and ministral-3:8b. Also computes the zero-model baselines (length, bet range, keywords). |
| `probe_length_check/` | 4.1, 4.2 | Re-implementation of the paper's activation probe (`reproduce_LR.py` settings), plus the test of whether probe scores track message length. |
| `symmetric_prompts/` | 4.2, Table 2 | Counterfactual on 80 real Stego table moments with qwen2.5:7b. Compares honest instructions (H), the original colluder instructions (C), and honest instructions plus the secret task (S). |
| `content_blind_monitor/` | 4.2, Table 3 | A monitor that sees only how much each agent reads and writes. Also decomposes the reading load into private section, role instructions and shared remainder. |
| `second_generator_model/` | 4.2, Figure 1 | Replication on the GPT-OSS-20B games: per-table silence and cheat-type ratios. |
| `core_coalition_reach/` | Appendix B | Core tier: whether honest teammates follow the coalition, and whether small local monitors drift to the coalition's option. |
| `report/` | | The PDF, the two figures and the script that draws them. |

Model outputs are bundled per experiment in `outputs.jsonl`. Each line holds one original output file:

```json
{"file": "...", "content": "..."}
```

## Data

All experiments use the public NARCBench release: https://huggingface.co/datasets/aaronrose227/narcbench (Rose et al., 2026, arXiv:2604.01151).

- **Scenarios:** Qwen3-32B for core, transfer and stego; GPT-OSS-20B for core, transfer and stego.
- **Activations:** Qwen3-32B core and stego.

The dataset is not redistributed here. The scripts contain the local paths used in our runs (`BASE` / `SRC` near the top of each file). Point them at your own download.

**Redaction note:** In the released Stego files, the big player's `PRIVATE:` decoding note is appended to its public messages: 140 lines for Qwen3-32B and 130 for GPT-OSS-20B. `stego_monitors/hazirla.py` cuts every public message at `PRIVATE` and drops the in-game pit boss's messages before any monitor sees the data.

## Environment

- Python 3.13. `numpy`, `scipy` and `scikit-learn` are needed for `probe_length_check/`, `content_blind_monitor/`, `second_generator_model/` and the scoring scripts.
- Local models via [Ollama](https://ollama.com): `qwen2.5:7b` and `ministral-3:8b`, at temperature 0 and seed 0. Settings are in each script.
- The Claude Sonnet 5.5 calls were made through isolated Claude Code sub-agents, one case per call, using the prompt text in the scripts or prompt files. Their raw answers are in `outputs.jsonl`.

## Provenance

- All code here was written during the hackathon, 6–9 November 2026. No pre-existing code is included.
- An earlier, rule-based referee developed by the author before the hackathon is mentioned in the report but is not part of this repository.
- Every experiment was pre-registered in the author's private notes before the corresponding model calls. Analyses that were not pre-registered are labelled post-hoc in the report.
- Script names and some printed labels are in Turkish (for example `hazirla` = prepare, `olc` = measure, `kos` = run, `sonuc` = result).
