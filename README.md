# Saudi real-estate price prediction — research prototype

A Python project exploring **property-listing price regression** using numeric property attributes, location categories and Arabic listing text. It contains a TensorFlow training script, a Flask prediction API, a separate Flask form client, and historical performance notes. This repository is **source-only**: the database and trained model/preprocessor/scaler are not tracked, so the web app does not run from a fresh checkout.

**Contributors credited in the original README:** Omar Saqr and Aabed Elghadbaan. The repository does not document a finer-grained division of work.

## Architecture and source map

1. [`src/scripts/preprocess_data.py`](src/scripts/preprocess_data.py): optional Arabic text lemmatization in a SQLite `Listings` table (mutates the specified database; back it up first).
2. [`src/scripts/train_model.py`](src/scripts/train_model.py): reads `PandA.db`, splits listings, fits numeric/categorical/TF-IDF preprocessing on training data, trains a dense regression network, reports test metrics, and writes three artifacts into the **current working directory**.
3. [`src/web/server.py`](src/web/server.py): loads the three artifacts and three JSON maps **from the current working directory**, then exposes `/api/metadata`, `/api/predict`, and `/api/feedback` on port 5000.
4. [`src/web/client.py`](src/web/client.py) and [`src/web/templates/index.html`](src/web/templates/index.html): Arabic-text processing and a browser form on port 8000, making requests to the API on port 5000.
5. [`src/scripts/check_assets.py`](src/scripts/check_assets.py): presence-only check for training and server assets; [`tests/`](tests/) contains dependency-light checks.

## Reproducibility / quickstart

The original project pins older dependencies, and the complete environment has **not** been reproduced here. Use a compatible isolated Python environment and review `requirements.txt` before installation; the training file uses `OneHotEncoder(sparse_output=False)`, which requires scikit-learn 1.2 or newer. TensorFlow 2.9.0 has older Python/platform wheel constraints. No installed environment, downloadable dataset rights, or model weights have been verified.

```bash
git clone https://github.com/omarsaqr12/Real-Estate-Price-prediction-in-Saudi.git
cd Real-Estate-Price-prediction-in-Saudi
python -m unittest discover -s tests -v
python src/scripts/check_assets.py --mode train
```

Get the historical `PandA.db` dataset from the [originally documented external folder](https://drive.google.com/drive/folders/1PT3MuIW0eej5w4jTOENe_C3g1o3o7LdN) **only if you have access and permission**. Its current availability, provenance, license and contents have not been independently verified. Put it at the repository root; inspect its tables, columns and permissions before executing preprocessing or training. `src/scripts/preprocess_data.py` currently uses the separate filename `database.db`, so explicitly reconcile/backup the database rather than assuming these scripts share a file.

After installing dependencies and providing a compatible, preprocessed database, the historical training command is `python src/scripts/train_model.py`. Training is expensive and **was not run for this review**. It writes `price_prediction_model.keras`, `preprocessor.pkl` and `y_scaler.pkl` in the working directory. Do not load model or pickle files from untrusted sources.

To run the historical demo, first put those three generated files **and** copies of `src/data/{category_mapping,city_mapping,district_mapping}.json` in the repository root; check `python src/scripts/check_assets.py --mode serve`, then start `python src/web/server.py` and `python src/web/client.py` in separate terminals and open `http://localhost:8000`. This is **local experimental code only**, not a publicly deployable service: the server binds to all interfaces in Flask debug mode, permits broad CORS, accepts untrusted feedback and runs automatic retraining without a hardened validation pipeline. Do not expose it on a network or supply real user data.

## Results and what they establish

[`docs/performance_metrics.md`](docs/performance_metrics.md) records **historical, unverified figures** for a neural-network run: R² 0.9235; MAE 179,996.90 SAR; RMSE 303,539.51 SAR; average percentage error 16.11%. The source computes regression metrics on a random held-out 20% split, but no immutable dataset snapshot, stored predictions, model artifact, environment lockfile, or reproduction log is committed. These figures are **reported, not independently reproduced or established for new properties**. Duplicate/relisted properties and temporal or geographic generalization have not been audited.

The previous README also discussed nine candidate algorithms and five-fold cross-validation. The checked-in training source implements **one** network, with 128/64/32 hidden units and TF-IDF capped at 5,000 terms; it does **not** contain the nine-model comparison or establish five-fold validation of those numbers. Do not interpret those historical comparisons as executable results in this repository.

## Known defects / development priorities

- The database-name mismatch (`database.db` versus `PandA.db`) and absence of model assets block a turnkey installation. The file-presence checker does not verify model compatibility.
- The server assumes specific fitted categorical encoder columns, while the training script allows missing input columns; the same feature schema must be verified against the real database before claiming prediction compatibility.
- Feedback can be submitted without authentication or linkage to a verified sale. Automatic fine-tuning on selectively submitted, potentially inaccurate examples is **experimental and not validated**; the 20% deviation threshold is not evidence that the system improves.
- `setup.py` describes source packaging only; no non-existent CLI entry points or unprovided license grant are claimed. The repository has no `LICENSE` file; permission to reuse data and other assets must be checked separately.

See [`docs/dataset_info.md`](docs/dataset_info.md) for the original dataset notes and [`REPOSITORY_SETUP.md`](REPOSITORY_SETUP.md) for the historical organization log (not current runtime instructions).
