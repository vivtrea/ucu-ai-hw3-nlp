# HW3 — AI-Generated Text Detection (UCU AI 2026)

Classify English/German texts as **human**, **ai** or **fusion** (human text partly rewritten by an LLM). Metric: macro-F1.
Kaggle: [ai-generated-text-detection-hw3-ucu-ai-2026](https://www.kaggle.com/competitions/ai-generated-text-detection-hw3-ucu-ai-2026).

**Result:** val macro-F1 **0.9705** — LightGBM stack over TF-IDF, mDeBERTa-v3 encoders (incl. per-window stats and a human-vs-fusion specialist) and LM/Binoculars features. Best single model: 0.9305.

## Structure

```
data/raw/      competition data (not committed)
data/splits/   split.csv from 02 — the fixed group-aware val split used everywhere
notebooks/     01–05, run on Kaggle, saved with outputs
reports/       result tables (*.csv) and figures/
```

## Run (Kaggle, in order)

Each notebook's first cell lists its settings and inputs; attach earlier notebooks' outputs via *Add Input -> Your Work -> Notebooks*. GPU = T4x2 with Internet on. Seed 42.

| Notebook | What | HW |
|---|---|---|
| `01_eda` | class balance, lengths, groups, fusion stylometry | CPU |
| `02_baseline` | **val split**, TF-IDF + linear models | CPU |
| `02b_baseline_full` | best baseline on all 245k rows | CPU |
| `03_encoder` | mDeBERTa-v3, 40k rows; head / head+tail / chunks | GPU |
| `03b_encoder_full` | mDeBERTa-v3, all rows; window stats (**reference**, 0.9305) | GPU |
| `03c_encoder_human_fusion` | binary human-vs-fusion specialist | GPU |
| `04_my_approach` | LM-perplexity features, stacking, class weights | GPU |
| `04b_my_approach_v2` | + Binoculars + specialist → **final submission** | GPU |
| `05_explainability` | TF-IDF weights, Integrated Gradients, SHAP | GPU |

02b, 03, 03b and 03c need only 02 and can run in any order.
