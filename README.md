# Arabic-Sentiment-Analysis_withMlops-
# Arabic Sentiment Analysis (AraBERT)

Classifies Arabic reviews as **negative**, **mixed** or **positive**. A fine-tuned AraBERT model
served by a FastAPI service and packaged with Docker.

> **Status: Module 1 complete** (package, API, Docker). The service currently serves a
> **placeholder model**: 2,000 training reviews, 1 epoch, validation macro-F1 0.55. A tracked
> baseline trained on all the data comes in Module 2. Do not read the placeholder's numbers as
> the project's final performance.

## Run the service

Requires Docker. The model files are not stored in Git. Until DVC is added in Module 2, place a
trained model in `models/arabert-sentiment/`, then:

```bash
git clone https://github.com/jihadhamd/Arabic-Sentiment-Analysis_withMlops-.git
cd Arabic-Sentiment-Analysis_withMlops-
docker compose up
```

The image is also on Docker Hub (it contains the code but **not** the model, which is mounted
from `models/`):

```bash
docker pull <jihadhamy>/arabic-sentiment:0.1.0
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
  512 tokens 98.7%. AraBERT reads at most 512 tokens. The baseline uses `max_length` 128.
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

## Known limitations

- The served model is a placeholder (see the status note above).
- The class balance is artificial. Real traffic will not be balanced, so scores on this data will
  look better than live performance. Drift monitoring is planned for Module 5.
- "Mixed" is probably a middling or two-sided opinion, which is not the same as "neutral". This is
  a hypothesis from reading examples, not a verified fact.
- The API checks the shape and size of the input, not its language. Non-Arabic text still gets a
  label, and that label means nothing.
- Reviews longer than `max_length` tokens are truncated.
- The Docker image uses CPU-only PyTorch. GPU serving comes in Module 3.
