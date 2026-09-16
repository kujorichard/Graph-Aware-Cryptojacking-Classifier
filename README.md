# Cryptojacking Detection Thesis

This repository contains the data-processing and machine-learning pipeline for detecting cryptojacking in binary samples. The pipeline converts samples into assembly and graph representations, extracts graph-aware n-grams, builds feature datasets, and trains or evaluates Random Forest classifiers.

## Repository folders

### `dissassembly/`

Contains the Ghidra-based headless disassembly pipeline. It stages and optionally unpacks samples, runs Ghidra analysis, exports assembly plus control-flow graph (CFG) data, and records sample provenance. `reaching_defs.py` adds reaching-definitions data-flow graph (DFG) information, while `gcb_graph_input.py` prepares graph-guided input for GraphCodeBERT-style processing. The folder also contains Ghidra scripts, intermediate outputs, logs, and local sample-processing artifacts.

This folder is ignored by Git because it contains generated analysis data, local Ghidra projects, and potentially sensitive raw samples.

### `graph_ngrams/`

Contains the graph-aware n-gram extraction pipeline. The loaders and models read CFG and DFG representations, the extractors traverse those graphs to produce n-gram sequences, and the vocabulary, shingling, vectorization, and dataset-builder modules turn the extracted sequences into classifier-ready features. The `dataset/` subfolder stores graph JSON inputs used by this stage.

### `graph_aware_classifier/`

Contains the graph-aware classifier workflow, including the notebook for experimenting with the Random Forest models, stored models and vocabularies for different n-gram sizes, sample prediction and explanation scripts, and utilities for extracting n-grams from individual samples. Its `dataset/` folder stores graph samples used by the classifier workflow.

### `ml/`

Contains the conventional machine-learning dataset workflow. The dataset loader reads benign and cryptojacking graph samples, the feature-matrix and vocabulary scripts create sparse feature datasets for 2-, 3-, and 4-grams, and the notebooks and supporting code train and evaluate Random Forest models.

The contents of `ml/dataset/benign/` and `ml/dataset/cryptojacking/` are ignored by Git because they are local sample datasets. The directory structure can still be created locally when preparing data.

## Typical processing flow

1. Place benign and cryptojacking samples in the local input directories used by the disassembly pipeline.
2. Run `dissassembly/disassemble.py` to stage samples and generate assembly and CFG records with Ghidra.
3. Run `dissassembly/reaching_defs.py` to add DFG information, then use `dissassembly/gcb_graph_input.py` when preparing graph-guided model input.
4. Use `graph_ngrams/` to extract CFG- and DFG-guided n-grams and build vocabularies or feature matrices.
5. Use the scripts and notebooks in `ml/` or `graph_aware_classifier/` to train, evaluate, explain, or run the stored Random Forest models.


Dependencies for the Python Dissassembly components are listed in `dissassembly/requirements.txt`.