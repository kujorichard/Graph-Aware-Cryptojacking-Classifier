# Streamlit Demo Setup — Cryptojacking Detection

This Streamlit app is a UI wrapper around the existing `graph_aware_classifier` project. It does not train a new model or replace the Random Forest, SHAP, or LLM implementation.

## Recommended folder layout

Keep the existing project package at the project root and put the Streamlit-related files inside `demo/`:

```text
YourProject/
├── CryptojackingDetection.ipynb
├── graph_aware_classifier/
│   ├── sample_transformer.py
│   ├── sample_explainer.py
│   ├── llm_analyzer.py
│   ├── ... trained model / vocabulary files ...
│   ├── .env
│   └── occurences/
│       └── <sample-stem>_vocabulary.json
└── demo/
    ├── streamlit_app.py
    └── STREAMLIT_SETUP.md
```

**Do not move or copy the `graph_aware_classifier` package into `demo/`.** The app searches its parent directories for the existing package, so this layout is supported.

Keep all trained model files, vocabulary files, package modules, occurrence lookups, and `.env` configuration in the locations already used by your notebook.

## Install and run

Activate the same Python environment used for the notebook, then install Streamlit if needed:

```bash
python -m pip install "streamlit>=1.55,<2"
```

From the project root, run:

```bash
streamlit run demo/streamlit_app.py
```

Alternatively, enter the `demo/` directory and run `streamlit run streamlit_app.py`.

If the project's existing dependencies are not installed in that environment, install them using the project's dependency file first.

## Using the app

1. Upload the **preprocessed JSON sample** that the notebook passes to `transform_sample()`.
2. Keep its matching occurrence lookup in `graph_aware_classifier/occurences/`. The lookup filename is based on the uploaded sample's filename stem and uses the notebook's existing `<sample-stem>_vocabulary.json` convention.
3. Click **Run detection pipeline**.
4. Inspect the visible stages: sample transformation, Random Forest prediction and tree view, SHAP feature contributions, n-gram-to-function occurrence lookup, and conditional LLM analysis.

The current notebook does not pass a raw `.exe` directly to `transform_sample()`. This app preserves that behavior; upload a preprocessed JSON sample, not an executable.

## LLM behavior

- The LLM stage is called only when Random Forest predicts class `1` (cryptojacking), matching the notebook's conditional behavior.
- Configure the LLM backend/model using the existing `graph_aware_classifier/.env` settings.
- The LLM provides an explanation after the classifier's prediction; it is not a second classifier or replacement verdict.
- Missing occurrence data or LLM configuration is reported in the UI.

## What is visible

- Stage-by-stage progress.
- Random Forest class and predicted probability for class `1`.
- A depth-limited visualization and readable rules for one constituent decision tree (not the entire forest).
- The top 20 SHAP-ranked n-grams, frequencies, and contributions.
- Function names associated with influential n-grams when lookup data is available.
- LLM assessment tier, confidence score, explanation, key indicators, recommended action, and per-feature assessments when returned.
