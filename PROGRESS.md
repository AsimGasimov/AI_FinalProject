# PROGRESS.md — live build checklist

- [x] STOP 1 — docs read, plan approved, hardware detected (i7-10750H, GTX 1660 Ti 6GB, py3.13)
- [x] Prompt 1 — skeleton: dirs, config.py, run.ps1, requirements, gitignore, GitHub repo
- [x] Prompt 2 — nutrition_db.json (25 classes), dataset.py, prepare_data.py; Food101 download ← IN PROGRESS (~60%)
- [x] Prompt 3 (code) — models.py (SimpleCNN 396K, EffNet 4.04M), train.py CLI; smoke test pending download
- [x] Prompt 4 (code) — evaluate.py + gradcam.py + predict.py (runs after training)
- [x] Prompt 5 — CV: quality.py, segment.py, portion.py + 16 tests PASS + cv_debug.py
- [x] Prompt 6 — NLP: meal_parser.py + 20 tests PASS (threshold 0.55→0.75, see DECISIONS #10)
- [x] Prompt 7 — 9 AZ guideline docs, retriever, llm providers, advisor, summarizer (template output verified)
- [x] Prompt 8 — pipeline.py, db.py, api.py + tests (38 pass, 2 skip until checkpoint exists)
- [x] Prompt 9 (code) — Streamlit demo written (4 tabs, AZ, custom CSS); live check after training
- [ ] → STOP 2+3: smoke test ETA + class distribution + 3 nutrition entries
- [ ] Full training: SimpleCNN + EfficientNet-B0
- [ ] Evaluate + Grad-CAM artefacts on real checkpoints
- [ ] Prompt 10 — portion_validation.md, nlp_eval.md, results.ipynb
- [ ] Prompt 11 — README + PRESENTATION_OUTLINE.md
- [ ] Prompt 12 — pytest/ruff/mypy, DoD checklist, clean-clone test → STOP 4
- [x] GitHub: pushed after each step (github.com/AsimGasimov/AI_FinalProject)
