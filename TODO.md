# TODO — Semantic Search Engine

## Completed

- [x] MS MARCO v1.1 data loading and passage deduplication
- [x] Document embedding with `all-MiniLM-L6-v2` (384-dim, GPU/CPU auto-select)
- [x] Local cache system — embeddings, FAISS indexes, BM25 tokens, metadata JSON
- [x] FAISS HNSW index (`IndexHNSWFlat`) for approximate nearest-neighbour search
- [x] FAISS Exact index (`IndexFlatIP`) as brute-force baseline
- [x] BM25 lexical search (`rank-bm25`)
- [x] Dense (BERT) semantic search
- [x] Hybrid search — min-max normalised fusion: `alpha * bert + (1-alpha) * bm25`
- [x] Evaluation: `Precision@k` and `Recall@k` across configurable k values
- [x] ANN vs Exact timing and quality comparison
- [x] Query-type analysis (factual / short / complex buckets)
- [x] PCA 2-D embedding visualisation
- [x] CLI runner (`main.py`) with `--eval-queries`, `--k-values`, `--top-k`, `--alpha`, `--query`, `--model`, `--skip-eval`
- [x] Evaluation report saved to `outputs/evaluation_report.json`
- [x] Streamlit app with 5 tabs: Search, Comparison View, Analysis Dashboard, PCA Visualization, How To Read Results
- [x] Custom HNSW C++ implementation (`hnsw_bind.cpp`) exposed via pybind11
- [x] FAISS vs custom-HNSW comparison tab in Streamlit
- [x] Streamlit Cloud deployment: `packages.txt` (`build-essential`, `g++`, `clang`) + auto-compiler selection in `setup.py`
- [x] Lazy HNSW load with detailed build diagnostics on failure

---

## Known Bugs / Issues

- [ ] Custom HNSW uses **squared Euclidean distance** but embeddings are L2-normalised — inner product would be equivalent and consistent with the FAISS dense backend. Scores shown in the HNSW tab are negated distances, which differ in scale from FAISS inner-product scores.
- [ ] HNSW index is **rebuilt from scratch on every app cold-start** (no persistence to disk). With the full 50 k subset this adds ~30–60 s startup time.
- [ ] Streamlit Cloud HNSW build may still fail on certain runner images even with `packages.txt` entries — see README §15 redeploy checklist.
- [ ] `cache/` is tracked in git for the demo profile (subset = 3,900). Large binary files will cause GitHub warnings if the subset is switched to 50,000 and cache is not excluded via `.gitignore`.

---

## Pending / Future Work

### Metrics
- [ ] Add **MRR** (Mean Reciprocal Rank) to `evaluate_k()` and the dashboard
- [ ] Add **NDCG@k** (Normalised Discounted Cumulative Gain)
- [ ] Expose `MAP` (Mean Average Precision) for completeness

### Search & Retrieval
- [ ] Persist custom HNSW index to disk (save/load via pickle or a custom binary format) to avoid rebuild on each cold-start
- [ ] Add **cross-encoder re-ranking** as a fourth retrieval mode (retrieve with BM25/BERT, re-rank with cross-encoder)
- [ ] Experiment with **Word2Vec / GloVe** embeddings (referenced in `pipeline.txt` but not yet implemented)
- [ ] Grid-search alpha (`0.0` → `1.0`) in the Streamlit Analysis tab and plot a precision curve

### CLI
- [ ] Add `--subset-size` flag to `main.py` so the corpus size can be set from the command line without editing source (currently hardcoded as `FIXED_SUBSET_SIZE`)
- [ ] Add `--output-dir` flag so report path root is configurable

### App / UX
- [ ] Add a **Download Report** button in the Analysis Dashboard to export the JSON evaluation report
- [ ] Show per-query example in the Analysis tab (not just aggregated metrics)
- [ ] Display HNSW index build time in the UI on first load

### Code Quality
- [ ] Add unit tests for `retrieval_engine.py` (tokeniser, hybrid fusion, min-max normalisation, cache validation)
- [ ] Add integration test that runs the full pipeline on a tiny synthetic corpus (no MS MARCO download required)
- [ ] Type-check with `mypy` and add to CI

### Deployment
- [ ] Document external storage option (e.g., Hugging Face Hub or S3) for the full 50 k cache artifacts so they don't need to be pushed to GitHub
- [ ] Add a `Makefile` / `justfile` with `make build-cache`, `make eval`, `make app` shortcuts
