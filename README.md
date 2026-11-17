# FedPhysio — code, predictions and results

Code and results for the revised manuscript

> **FedPhysio: XAI-Guided Modality Selection and Reliability-Aware Federated Learning for Multimodal Digital Health**
> A. Ali, J. Huang, F. Saeed, M. Islam — *Frontiers in Digital Health*, manuscript 1998634 (under revision)

This repository lets anyone check every number in the revised paper. Each reported metric comes from **saved per-window predictions** on the held-out subjects, and `audit/recompute_metrics.py` recomputes all of them independently of the training code.

---

## 1. What is in this repository

| Path | Content |
|---|---|
| `notebooks/fedphysio_v2_1.ipynb` | The notebook that produced the revised results, **with its execution log** (per-run accuracy, validation score, selected configuration, consistency checks). |
| `src/fedphysio_v2_1.py` | The same code as a script (used by the parallel background workers). |
| `results/tables/` | All result tables of the revision (LaTeX bodies and CSV), written by the notebook. |
| `results/reference_first_corrected_run/` | Tables of the first corrected run (original pipeline): the seven reference federated methods, scenarios, the τ and backbone analyses, the attribution study. |
| `results/predictions/` | Per-window predictions of every reported run (see Section 4). |
| `audit/recompute_metrics.py` | Independent recomputation of every published number from the predictions. |
| `figures/` | Data figures of the paper as editable SVG/PDF, generated from the tables. |

The WESAD dataset is **not** redistributed. Download it from its authors (Schmidt et al., ICMI 2018) and set `DATA_ROOT` in the notebook.

---

## 2. Main results (strict leave-one-subject-out, 15 folds, 6,569 held-out windows)

| Protocol | Accuracy | Macro-F1 | Cohen's κ | AUC (macro) |
|---|---|---|---|---|
| **Strict LOSO** (primary) | **0.8835** | **0.8396** | **0.8023** | 0.9450 |
| Strict LOSO + causal 30-s smoothing | 0.8901 | 0.8487 | 0.8133 | 0.9511 |
| Subject-calibrated, 60 s per condition | 0.9123 | 0.8829 | 0.8485 | 0.9793 |
| Subject-calibrated, 120 s per condition | 0.9134 | 0.8833 | 0.8460 | 0.9853 |

**Robustness scenarios** (pooled κ): good 0.802, average 0.686, worst 0.596. In the worst case, the reliability bit flags 95.6 % of corrupted client-rounds and 3.2 % of clean ones.

**Reference federated methods** were trained in the first corrected run with the original pipeline, on the same test windows. They reach 77.6–84.6 % accuracy; the best is coordinate-wise Median (84.61 %, κ 0.738). They are **reference results, not a matched comparison** (see the paper).

---

## 3. Integrity and leakage controls

- **The held-out subject is never used for any choice.** It plays no part in normalisation statistics of other subjects, sensor-configuration choice, early stopping, checkpoint choice, prior correction or any hyperparameter. It is evaluated once, after training.
- **The test stream is label-independent.** Windows are spaced every 5 s for every class. Class-dependent spacing (5 / 2.5 / 1 s) is used for training windows only.
- **The split is made before windowing.** Each client's recording is split in time within each condition, 90 % training and 10 % validation.
- **No signal is mixed across subjects.** All augmentation pairs windows of the same class from the same client.
- **Normalisation is label-free.** The held-out subject is normalised with its own unlabelled recording, so the main protocol is *retrospective and subject-calibrated*. This is stated in the paper.
- **Pre-specified settings.** τ = 0.5, the scenario definitions and the clock-feature rule (|ρ| ≥ 0.5, computed on training subjects, label-free) were fixed before the reported runs.
- **Automatic consistency checks.** Pooled accuracy must equal support-weighted recall, and the per-subject confusion matrices must sum to the pooled matrix. Both pass for every protocol; see the notebook output.
- **Single seed.** Every result comes from one training run (seed 42). Every SD is between subjects.
- **Development history, for transparency.** The submitted results (97.52 %) could not be reproduced under the corrected protocol; the submitted design reaches 78.31 %. The method was then revised. An initial run of the revised pipeline revealed a batch-normalisation mismatch and session-clock features, which were fixed with label-free rules applied to all folds *before* the reported runs.

---

## 4. Reproduce or audit the results

**Audit only (minutes, no GPU):**
```bash
pip install -r requirements.txt
python audit/recompute_metrics.py --runs results/predictions --tables results/tables
```
The script prints every protocol, scenario and sensor configuration next to the published table and ends with `ALL PUBLISHED NUMBERS REPRODUCED`.

**Full re-run** (about 18–25 h on one RTX 3090 with one process):
1. Download WESAD and set `DATA_ROOT` in the first configuration cell of `notebooks/fedphysio_v2_1.ipynb`.
2. Choose the GPU with the environment variable `FEDPHYSIO_GPU` (default 3 in the notebook) and run all cells. To split the folds across parallel processes, start `python src/fedphysio_v2_1.py` several times with `FEDPHYSIO_WORKER=0/3`, `1/3`, `2/3`; the workers write the same final tables.
3. After it finishes, run `bash collect_predictions.sh fedphysio_v2_results` and then the audit script.

Interrupted runs resume from their last communication round.

---

## 5. Software and hardware used

Python 3.9, PyTorch 1.13.1 (CUDA 11.7), NumPy, SciPy, pandas, scikit-learn, Matplotlib; one NVIDIA RTX 3090 (24 GB), FP32.

## 6. Citation

Please cite the article once it is published. Until then, cite this repository and the manuscript number above.

## License

MIT (code). WESAD is subject to its own licence.
