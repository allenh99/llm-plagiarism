## LLM Plagiarism Detection

Classical ML classifiers (SVM, Naive Bayes, KNN) trained on text embeddings to distinguish between human-written vs. LLM-generated answers using the HC3 dataset.

### Project journal
-
  - Wrote setup and two workflows (OpenAI embeddings, local LLaMA embeddings).
  - Implemented `svm_openai.py` to build embeddings and persist `hc3_with_embeddings.pkl`.
  - Implemented `svm_openai_run.py` to train/evaluate SVM, Naive Bayes, KNN with grouped split.
  - Implemented `svm_llama.py` to fetch local embeddings via Ollama `llama3`.
  - Next: run OpenAI pipeline end-to-end and record baseline metrics below.

### Running metrics (update after each run)
- Baseline (OpenAI embeddings → SVM): TBD
- Naive Bayes (OpenAI embeddings): TBD
- KNN (OpenAI embeddings): TBD
- Notes: Grouped by `id` via `GroupShuffleSplit` to avoid question leakage.

### Next tasks
- [ ] Run `svm_openai.py` to generate `hc3_with_embeddings.pkl` at full/desired size.
- [ ] Run `svm_openai_run.py`, capture accuracy and classification report; update metrics above.
- [ ] Add persistence to `svm_llama.py` (e.g., `hc3_with_embeddings_llama.pkl`) for parity.
- [ ] Compare OpenAI vs. LLaMA embedding performance; add observations to journal.
- [ ] Add ROC-AUC and confusion matrices to evaluation output.
- [ ] Explore regularization/C parameter tuning for SVM.
- [ ] Consider train/dev/test split and cross-validation for stability.

### Repo layout
- `llama_methods/svm_openai.py`: Builds embeddings with OpenAI (`text-embedding-3-small`), prepares a flattened dataframe, and saves `hc3_with_embeddings.pkl`.
- `llama_methods/svm_openai_run.py`: Loads `hc3_with_embeddings.pkl`, performs a grouped train/test split by question `id`, trains SVM / Naive Bayes / KNN, and prints metrics.
- `llama_methods/svm_llama.py`: Generates embeddings locally via Ollama (`llama3`) at `http://localhost:11434/api/embeddings` and constructs feature/label arrays in-memory.

### Setup
1) Python environment
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

2) Datasets
- The scripts automatically download the HC3 dataset: `Hello-SimpleAI/HC3` split `all` via `datasets`.

3) OpenAI API (for `svm_openai.py`)
- Create a `.env` file in the project root with:
```bash
OPENAI_API_KEY=your_api_key_here
```

4) Local LLaMA embeddings (optional; for `svm_llama.py`)
- Install and run Ollama on macOS:
```bash
brew install ollama
ollama run llama3   # downloads the model on first run
```
- Ensure the embeddings API is available at `http://localhost:11434/api/embeddings`.

### Workflow A: OpenAI embeddings → train/eval
1) Build embeddings and save dataset
```bash
python llama_methods/svm_openai.py
```
- What it does:
  - Loads HC3, keeps rows with `chatgpt_answers` and excludes `source == "wikipedia"`.
  - Flattens into pairs per question: one human (`answer_source = 0`) and one ChatGPT (`answer_source = 1`).
  - Samples 1,000 rows for speed (adjustable in the script).
  - Calls OpenAI embeddings (`text-embedding-3-small`) and saves `hc3_with_embeddings.pkl` in the project root.

2) Train and evaluate classifiers
```bash
python llama_methods/svm_openai_run.py
```
- What it does:
  - Loads `hc3_with_embeddings.pkl`.
  - Uses `GroupShuffleSplit` grouped by question `id` to avoid leakage.
  - Trains SVM, Gaussian Naive Bayes, and KNN; prints accuracy and a classification report.

### Workflow B: Local LLaMA (Ollama) embeddings
```bash
python llama_methods/svm_llama.py
```
- What it does:
  - Loads and flattens HC3 (same filtering as above).
  - Requests embeddings from local Ollama (`model: llama3`).
  - Builds in-memory `X` (embeddings) and `y` (`answer_source`) arrays.
- Note: This script currently does not persist a `.pkl`. If you want parity with the OpenAI path, you can mirror the save step (e.g., `to_pickle("hc3_with_embeddings_llama.pkl")`).

### Labels
- `answer_source = 0`: human answers
- `answer_source = 1`: LLM (ChatGPT) answers

### Tips & troubleshooting
- OpenAI rate limits/errors: the script skips failed embeddings and logs a short message; re-run if needed.
- Ollama not responding: ensure the daemon is running and that `llama3` has been pulled via `ollama run llama3` at least once.
- Dataset size/time: increase or remove the sampling (`df_flat.sample(1000, ...)`) in `svm_openai.py` for better coverage.
- Reproducibility: random seeds are set where applicable (e.g., sampling, splits) for consistency.

### Output artifacts
- `hc3_with_embeddings.pkl`: Pickled pandas DataFrame containing `id`, `source`, `question`, `answer`, `answer_source`, and `embedding` (only created by `svm_openai.py`).

### License
See original dataset license for HC3. Model licenses apply (OpenAI terms, LLaMA via Ollama).