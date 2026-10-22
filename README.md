# Arabic Sentiment Analysis (AraBERT)

Classifies Arabic reviews as **negative**, **mixed** or **positive**. A fine-tuned AraBERT model
served by a FastAPI service and packaged with Docker.

> **Status: Modules 1-2 complete** (package, API, Docker, MLflow tracking, DVC pipeline,
> CI/CD). Best validation macro-F1 is 0.7573 (run `len256-wd0001`, registered as
> `ArabicSentiment` / `Production`). The API loads whatever model is placed in
> `models/arabert-sentiment/`.

## Run the service

Requires Docker. The model files are not stored in Git. Run `dvc pull` to fetch the data and
models, or place a trained model in `models/arabert-sentiment/`, then:

```bash
git clone https://github.com/jihadhamd/Arabic-Sentiment-Analysis_withMlops-.git
cd Arabic-Sentiment-Analysis_withMlops-
docker compose up
```

The image is also on Docker Hub (it contains the code but **not** the model, which is mounted
from `models/`):

```bash
docker pull jihadhamy/arabic-sentiment:latest
```

## API

```bash
curl localhost:8000/health
# {"status":"healthy"}

curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
  -d '{"text": "تجربة سيئة جدا ولن أعود مرة أخرى"}'
# {"label":"negative","confidence":0.5189,"model_version":"0.1.0"}
```

Empty, blank or missing `text` returns **422**. Interactive docs: http://localhost:8000/docs

## Data

- **Dataset:** `ar_reviews_100k`. https://www.kaggle.com/datasets/abedkhooli/arabic-100k-reviews.
- **Size and balance:** 99,999 reviews, exactly 33,333 per class (negative, mixed, positive). No
  missing values, no duplicates, no review with two different labels.
- **Splits:** 79,999 train / 10,000 validation / 10,000 test, stratified, seed 42. The test split
  is opened only with `--evaluate-test`.
- **Review length:** median 38 AraBERT tokens, 95th percentile 257, 99th percentile 565 (5,000-review
  sample). Share of reviews that fit without truncation: 128 tokens 85.2%, 256 tokens 95.0%,
  512 tokens 98.7%. AraBERT reads at most 512 tokens. The first baseline used `max_length` 128;
  the best run uses 256.
- **Short reviews:** 731 reviews (0.73%) have one or two words.
- **Topics:** the data mixes several kinds of reviews, including hotels and books. A keyword check
  found a similar topic mix in all three classes.

Full analysis: `notebooks/01_eda_ar_reviews_100k.ipynb`.

## Development

```bash
python3 -m venv .venv && source .venv/bin/activate   # or: uv venv
pip install -e ".[train,dev]"                         # install PyTorch for your hardware first
pre-commit install
pytest
python -m src.data     # build the train/val/test splits
python -m src.train    # fine-tune; see --help for options
```

Layout: `src/` (config, text, data, model, train, api, schemas), `tests/`, `notebooks/`.
Training and serving share one text-cleaning function (`src/text.py`) to avoid train/serve skew.

## Module 2: Experiment Tracking, Data Versioning & CI/CD

### Experiment tracking (MLflow on DagsHub)
All runs are logged to the `arabic-sentiment` experiment on DagsHub MLflow.
Each run records `model_name`, `lr`, `batch_size`, `epochs`, `max_length`,
`weight_decay`, `accuracy`, `f1_macro`, training time and peak GPU memory,
and the trained model itself.

| Run | Setup | Val accuracy | Val macro-F1 |
|---|---|---|---|
| baseline | len128, lr 2e-5, bs16, 2 ep | 0.7493 | 0.7492 |
| len256-lr2e-5 | len256, lr 2e-5 | 0.7549 | 0.7547 |
| len512-lr2e-5 | len512, lr 2e-5 | 0.7561 | 0.7557 |
| len256-wd001 | len256, lr 5e-5, wd 0.01 | 0.7562 | 0.7563 |
| dvc-training (DVC pipeline) | len256, lr 5e-5, wd 0.01 | 0.7570 | 0.7563 |
| **len256-wd0001 (best)** | len256, lr 5e-5, wd 0.001 | **0.7568** | **0.7573** |

Comparison of all runs: `reports/mlflow_comparison.png`.

The best model is registered in the MLflow model registry as
`ArabicSentiment` and promoted to the `Production` alias.

### Data and pipeline versioning (DVC)
The pipeline has two stages, `prepare` and `train`, defined in `dvc.yaml`.
The data, processed splits and trained model are tracked by DVC, with the
DagsHub S3-compatible storage as the remote.

```bash
dvc pull     # fetch data and models
dvc repro    # reproduce preparation and training
dvc status   # clean when everything is up to date
dvc push     # upload results to the remote
```

Reproducibility check: two runs of the pipeline with the same parameters
(`len256-wd001` and `dvc-training`) gave validation macro-F1 of 0.7563 and
0.7563, well inside the required ±0.005.

### CI/CD (GitHub Actions)
**`CI` workflow:** on pull requests and on pushes to `main`: ruff lint,
pytest, Docker build and a smoke test. On a push to `main` only, the image
is tagged with the commit SHA and `latest` and pushed to Docker Hub
(`jihadhamy/arabic-sentiment`).

**Quality gate (`Model quality` workflow):** pulls
`models/dvc-training/metrics.json` from the DVC remote and runs
`scripts/check_model_quality.py`. The build fails if `val_f1_macro` drops
more than 0.005 below the baseline of 0.7573
(`reports/model_quality_baseline.json`, reference run `len256-wd0001`),
i.e. below about 0.7523.

### Environment variables for training
Set `MLFLOW_TRACKING_URI`, `MLFLOW_TRACKING_USERNAME` and
`MLFLOW_TRACKING_PASSWORD` (a DagsHub token) before running training.

## Known limitations

- The Docker image does not contain the model. Run `dvc pull` or mount a model in `models/`.
- The class balance is artificial. Real traffic will not be balanced, so scores on this data will
  look better than live performance. Drift monitoring is planned for Module 5.
- "Mixed" is probably a middling or two-sided opinion, which is not the same as "neutral". This is
  a hypothesis from reading examples, not a verified fact.
- The API checks the shape and size of the input, not its language. Non-Arabic text still gets a
  label, and that label means nothing.
- Reviews longer than `max_length` tokens are truncated.
- The Docker image uses CPU-only PyTorch. GPU serving comes in Module 3.
