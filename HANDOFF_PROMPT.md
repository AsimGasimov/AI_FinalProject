# HANDOFF PROMPT — copy everything below the line into your Claude Code

---

You are taking over **FoodLens**, a finished AI-Engineering course project. It is
**complete and working** — all code, trained models, evaluation, docs and a
one-click demo are done and pushed. Your job is to get it running, understand it
well enough to defend or extend it, and (optionally) improve it. Work
autonomously; do not ask me to write code.

## Get the code

```
git clone https://github.com/AsimGasimov/AI_FinalProject.git
cd AI_FinalProject
```

The trained checkpoints are **already in the repo** (`models/effnet_best.pt`,
`models/simple_best.pt`) — you do NOT need to retrain. All report artefacts
(`reports/`) and the executed notebook are included too.

## Read these first, fully, in this order

1. `README.md` — what it is, architecture diagram, results, how to run
2. `FINAL_REVIEW.md` — quality-gate status, DoD checklist, the 3 weakest parts + how to defend them
3. `CLAUDE.md` — binding project rules (tech stack, layout, coding standards)
4. `PROJECT_SPEC.md` — the complete technical specification
5. `DECISIONS.md` — 11 logged engineering decisions; do NOT re-litigate them
6. `PRESENTATION_OUTLINE.md` — 12 Azerbaijani slides with speaker notes
7. `BASLA.md` — how the owner runs the demo (one-click launchers, Windows)

## Set up and verify (any OS)

```
pip install -r requirements.txt
python scripts/prepare_data.py     # Food-101 25-class subset + nutrition DB
pytest -q                          # expect 40 passed
streamlit run app/streamlit_app.py # demo at http://localhost:8501
```

On Windows the owner just double-clicks the desktop **FoodLens** shortcut
(`START.bat`) — venv check + API + demo + browser all launch automatically.

> The demo runs **fully offline** — `LLM_PROVIDER=template` is the default, no API
> key needed. This is deliberate (defence-day insurance). Keep it that way.

## Current verified state

- **CV (OpenCV):** quality gate (blur/brightness) → HoughCircles plate detection
  → GrabCut segmentation → area-to-grams portioning.
- **CNN (PyTorch):** from-scratch `SimpleCNN` baseline vs `EfficientNet-B0`
  transfer, with Grad-CAM via manual hooks.
  - Test set (6 250 imgs): **EfficientNet top-1 0.923 / macro-F1 0.922**;
    SimpleCNN 0.504 / 0.486. (The weak baseline is intentional — it quantifies
    the transfer-learning gain.)
- **NLP:** Azerbaijani meal parser/NER + hybrid cosine-plus-lexical RAG. Parser
  20/20, RAG hit-rate 8/10 (misses documented honestly in `reports/nlp_eval.md`).
- **Serving:** FastAPI + SQLite; Streamlit demo (4 tabs, all Azerbaijani).
- **Quality gates all green:** `pytest` 40 passed · `ruff check .` clean ·
  `mypy src config.py` clean.

## If you want to extend it (optional, pick any)

1. **Add Azerbaijani dishes** (qutab, dolma, plov…): collect images, add classes
   to `config.CLASSES` + `data/nutrition_db.json`, re-run
   `python -m src.cnn.train --model effnet --epochs 10 --bs 32 --num-workers 2`.
2. **Improve portioning** — the weakest part (MAPE ≈ 131 %): add a reference-object
   (coin/card) scale so the cm/px estimate stops assuming a fixed 26 cm plate.
3. **Multi-dish detection**: run an object detector before the classifier so one
   photo with several foods works (currently one-dish-per-plate).

## Non-negotiable rules (if you touch the code)

- Every user-facing string and generated advice stays **Azerbaijani**. Code,
  comments and commits stay in English.
- `LLM_PROVIDER=template` stays the default — never make the demo require a network.
- Never replace the CNN/CV/NLP layers with API shortcuts; the grade depends on
  them being real.
- Do NOT train with `--mixed-precision` on a GTX 16xx GPU — it NaNs BatchNorm
  (see `DECISIONS.md` #11) and gives no speedup (no tensor cores).
- Report errors honestly (portion MAE, RAG misses); examiners reward acknowledged
  limitations. Seeds stay at 42. `git commit` after each meaningful step.
- Windows host uses `run.ps1` (not a Makefile); on macOS/Linux call the same
  `python -m ...` commands directly.
