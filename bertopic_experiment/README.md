This folder is independent from the homework deliverable.

It contains exploratory BERTopic experiments that use:
- project text preprocessing
- either TF-IDF + TruncatedSVD embeddings or sentence-transformer embeddings
- UMAP
- HDBSCAN
- BERTopic's c-TF-IDF topic representation

Run with the local experiment environment:

```bash
./.venv/bin/python run_bertopic_experiment.py
./.venv/bin/python run_bertopic_experiment.py --embedding-backend sentence_transformer
```

Outputs are written into `results/<backend>/`.
