# FoodLens — Final Review (Prompt 12)

_Generated 2026-07-15 · all quality gates green · clean-clone verified_

## Quality gates

| Gate | Command | Result |
|------|---------|--------|
| Tests | `pytest tests/ -q` | ✅ **40 passed** (incl. 2 checkpoint-gated) |
| Lint | `ruff check .` | ✅ **All checks passed** |
| Types | `mypy src config.py` | ✅ **no issues in 24 files** |

## Definition of Done (PROJECT_SPEC §10) — item by item

| # | Item | Status | Evidence |
|---|------|:------:|----------|
| 1 | `pip install -r requirements.txt` + train effnet runs clean | ✅ PASS | `models/effnet_best.pt`, `reports/history_effnet.json` (10 epochs, val-F1 0.878) |
| 2 | `streamlit run app/streamlit_app.py` end-to-end demo | ✅ PASS | 4 tabs (Şəkil/Mətn/Gündəlik/Model), light theme, webcam+upload, Grad-CAM shown |
| 3 | `pytest` green | ✅ PASS | 40 passed in ~32 s |
| 4 | `reports/metrics.json` shows both models | ✅ PASS | simple + effnet entries with top1/top5/macroF1/params/cpu-ms |
| 5 | README with diagram, results table, limitations, how-to-run | ✅ PASS | `README.md` (Mermaid flowchart, results table, ethics, AZ+EN) |
| 6 | Grad-CAM overlays saved | ✅ PASS | `reports/gradcam/` 10 overlays incl. 2 misclassified |
| 7 | Every user-visible string in Azerbaijani | ✅ PASS | app labels, advisor/summarizer templates, disclaimer — all AZ |

## Headline results (test set, 6 250 images)

| Model | Top-1 | Top-5 | Macro-F1 | Params | CPU ms |
|-------|------:|------:|---------:|-------:|-------:|
| SimpleCNN (from scratch) | 0.504 | 0.843 | 0.486 | 395 801 | 11.8 |
| EfficientNet-B0 (transfer) | **0.923** | **0.989** | **0.922** | 4 039 573 | 21.9 |

- NLP: parser 20/20, RAG hit-rate 8/10 (`reports/nlp_eval.md`).
- Portion: MAE ≈ 159 g, MAPE ≈ 131 % vs nominal servings (`reports/portion_validation.md`).

## Clean-clone test

`git clone --depth 1` into a fresh directory → checkpoints present (16.5 MB effnet,
1.6 MB simple) → `Predictor("effnet")` loads and runs a forward pass returning 25
classes. All deliverables (README, presentation outline, notebook, reports,
gradcam) present in the clone. ✅

---

## The 3 weakest parts (and how to defend them)

### 1. Portion estimation (MAPE ≈ 131 %)
**Weakness:** grams are unreliable — small foods (donuts, waffles) are wildly
over-estimated because Food-101 close-up crops are not real 26 cm plates, and a
2-D area cannot see food height.
**Defence:** This is an *ill-posed* problem — single-image monocular mass
estimation without a fiducial marker is fundamentally under-constrained. We do not
hide it: we measured it (`portion_validation.md`), documented four failure modes,
attach a `confidence` flag to every estimate, and the UI shows the S/M/L bucket
next to the gram figure. Correct engineering response to an unsolvable-as-stated
problem is transparency, not a fake precise number. Fix path: a reference object
(coin/card) or a depth sensor would make this tractable.

### 2. SimpleCNN baseline is weak (50 % top-1)
**Weakness:** the from-scratch CNN is not competitive.
**Defence:** That is the *point* of the baseline — it exists to quantify how much
transfer learning buys us (2× the accuracy for ~10× the params). A weak,
honestly-reported baseline is more scientifically useful than a tuned one that
blurs the comparison. Training curves and both confusion matrices are in the
notebook to show the gap is systematic, not noise.

### 3. Food-101 Western-food bias & single-dish assumption
**Weakness:** the 25 classes are Western; Azerbaijani dishes (qutab, dolma, plov)
are absent, so the model will confidently mis-classify them. The pipeline also
assumes one food per plate.
**Defence:** This is a dataset/scope limitation, stated up front in the README
ethics section and the non-goals. The architecture is class-agnostic — adding AZ
classes is a data-collection task, not a redesign. Multi-dish detection is listed
as future work (would need an object detector before the classifier). We chose to
ship a solid, honest 25-class system rather than a shaky 50-class one.

---

## Non-negotiables — confirmed

- ✅ `LLM_PROVIDER=template` default → demo runs offline, zero API keys.
- ✅ CNN / CV / NLP layers are all real (no API shortcuts).
- ✅ Seeds fixed at 42 everywhere.
- ✅ Errors reported honestly (portion MAE, RAG misses, weak baseline).
- ✅ Repo public: https://github.com/AsimGasimov/AI_FinalProject
