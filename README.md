# Journal Recommendation System

This repository contains the final delivery version of the data mining project. The selected recommendation method is a TF-IDF and cosine-similarity baseline, and the selected topic clustering model is the accepted `K=46` K-Means solution derived from a broad K-selection analysis.

## Project Summary
- Loads article abstracts, author keywords, subjects, and keyword plus fields from `CompSciencePub.sqlite`.
- Recommends the top 5 journals for a user-provided abstract.
- Evaluates recommendation quality with a leakage-free hold-out split.
- Builds a domain-aware clustering feature space and compares `K=2..100` plus `K=45`.
- Uses the accepted elbow-based `K=46` decision for the final clustering outputs.
- Summarizes an experimental BERTopic alternative without making it part of the main delivery pipeline.

## How To Run
1. Install the main requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Generate the final outputs:
   ```bash
   python main.py
   ```
3. Regenerate the notebook:
   ```bash
   python create_notebook.py
   ```
4. Build the clean submission folder:
   ```bash
   python build_submission.py
   ```

Optional BERTopic experiments require extra dependencies:

```bash
pip install -r requirements-bertopic.txt
```

## Delivery Contents
The clean submission build includes:
- source code
- Jupyter notebook
- IEEE report source and PDF
- final outputs and figures
- `data/CompSciencePub.sqlite`
- BERTopic experiment source code only

The clean submission excludes caches, logs, temporary files, raw BERTopic artifacts, and unrelated exploratory folders.

## GitHub
GitHub repository: `https://github.com/al1berk/datamining_finalproject.git`
