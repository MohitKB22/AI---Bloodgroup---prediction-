# BloodPrint Research Platform

[![CI](https://img.shields.io/badge/CI-GitHub%20Actions-blue)](.github/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-style AI platform -- ensemble computer vision, a FastAPI backend, and a React
dashboard -- built around a fingerprint image classification task.

> **⚠️ Research prototype, not a medical device.** This project explores a statistically
> contested hypothesis (that fingerprint patterns predict ABO blood group). It is not validated
> for, and must never be used for, any medical, clinical, or blood-donation/transfusion decision.
> **Read [MODEL_CARD.md](MODEL_CARD.md) before using or extending this project.**

---

## What's actually in this repo

This is an engineering demonstration first, a fingerprint classifier second. The interesting
problems solved here -- leakage-safe evaluation, calibrated confidence, explainability, an
authenticated API, a real frontend, containerized infra, CI -- are the same ones any deployed
image classifier faces, independent of whether *this particular* prediction target turns out to
be real. See [MODEL_CARD.md](MODEL_CARD.md) for the honest assessment of the underlying
hypothesis, and swap in a different (validated) fingerprint task if you want the scaffolding
without the caveat.

## Architecture

```
                      ┌──────────────┐
                      │    nginx     │  (edge reverse proxy)
                      └──────┬───────┘
                 ┌───────────┴────────────┐
                 ▼                        ▼
        ┌─────────────────┐      ┌─────────────────┐
        │  React frontend │      │  FastAPI backend │
        │  (nginx-served) │      │  (gunicorn +     │
        └─────────────────┘      │   uvicorn)       │
                                  └───┬────┬────┬────┘
                       ┌──────────────┘    │    └───────────────┐
                       ▼                   ▼                    ▼
              ┌────────────────┐  ┌───────────────┐   ┌──────────────────┐
              │  PostgreSQL    │  │     Redis      │   │  ml/ package     │
              │  (users,       │  │   (caching,    │   │  (preprocessing +│
              │   predictions) │  │  rate limiting)│   │   ensemble       │
              └────────────────┘  └───────────────┘   │   inference)     │
                                                        └──────────────────┘
```

`ml/` is a plain Python package shared by training (offline) and serving (imported directly by
the FastAPI backend) -- one implementation of preprocessing and inference, not two copies that
can drift apart.

## Features

**ML pipeline** (`ml/`)
- Fingerprint preprocessing: CLAHE, denoising, Gabor-filter ridge enhancement with orientation/
  frequency field estimation, block-variance segmentation, ROI extraction, and an NFIQ-inspired
  image quality score.
- Ensemble of EfficientNetV2, Vision Transformer, and ConvNeXt (via `timm`), combined by a
  learned weighted-softmax or stacking meta-learner.
- Transfer learning, mixed precision, focal loss + label smoothing, cosine LR schedule with
  warmup, gradient clipping, early stopping, checkpointing.
- Subject-grouped `StratifiedGroupKFold` cross-validation (see [MODEL_CARD.md](MODEL_CARD.md) for
  why this matters here specifically).
- Test-time augmentation, temperature-scaling calibration, Grad-CAM (CNN backbones) and attention
  rollout (ViT) explainability.
- MLflow experiment tracking, TensorBoard logging.

**Backend** (`backend/`)
- FastAPI, async SQLAlchemy 2.0 (PostgreSQL in prod, SQLite for local dev/tests), JWT auth
  (access + refresh tokens), Redis caching with an automatic in-memory fallback, Alembic
  migrations, Swagger/ReDoc docs, Prometheus metrics, structured JSON logging, PDF report
  generation, rate limiting.

**Frontend** (`frontend/`)
- React 18 + Vite + Tailwind, drag-and-drop upload, a calibration dial visualizing raw vs.
  temperature-scaled confidence, per-class probability charts, Grad-CAM/attention viewer,
  prediction history, analytics, downloadable PDF reports, dark mode (default) and light mode,
  4-language i18n (English, Spanish, French, Hindi).

**Infra**
- Docker + Docker Compose (postgres, redis, backend, frontend, mlflow, nginx), GitHub Actions CI
  (ML tests, backend tests, lint, frontend build, Docker build check).

## Project structure

```
.
├── ml/                     # Training + inference, importable by both offline scripts and the API
│   ├── config/             # Central config (class names, hyperparameters)
│   ├── preprocessing/      # CLAHE, denoise, ridge enhancement, segmentation, ROI, quality
│   ├── data/                # Dataset manifest, leakage-safe splitting, augmentation, Dataset class
│   ├── models/              # Backbone wrappers, ensemble, focal loss
│   ├── training/            # Training engine, schedulers, callbacks, ensemble fitting, CLI
│   ├── evaluation/          # Metrics, calibration, Grad-CAM, attention rollout, report generation
│   ├── inference/           # Serving-time predictor (used by the backend) + TTA
│   └── tests/                # Pytest suite (preprocessing, losses, splits, calibration, etc.)
├── backend/
│   ├── app/
│   │   ├── api/v1/endpoints/  # auth, users, predictions, health
│   │   ├── core/               # config, security (JWT/bcrypt), logging, exceptions
│   │   ├── db/                  # SQLAlchemy engine/session
│   │   ├── models/              # ORM models (User, Prediction)
│   │   ├── schemas/             # Pydantic request/response schemas
│   │   ├── services/            # ML inference wrapper, cache, PDF report generation
│   │   └── middleware/          # Error handling, Prometheus metrics
│   ├── alembic/                 # DB migrations
│   └── tests/                   # Pytest + httpx async test suite
├── frontend/
│   └── src/
│       ├── api/                 # Axios client with JWT refresh
│       ├── components/          # UploadDropzone, ConfidenceDial, ProbabilityChart, etc.
│       ├── context/             # Auth + theme
│       ├── i18n/                 # en/es/fr/hi translations
│       └── pages/                # Dashboard, History, Analytics, About, Auth
├── infra/nginx/             # Edge reverse-proxy config
├── .github/workflows/       # CI pipeline
├── docker-compose.yml
├── MODEL_CARD.md
└── README.md
```

## Quick start (Docker Compose)

```bash
cp .env.example .env              # then edit BP_SECRET_KEY and POSTGRES_PASSWORD
docker compose up --build
```

This starts Postgres, Redis, the backend API, the React frontend, an MLflow tracking server, and
an nginx edge proxy on port 80. **The backend will report `"model": "not loaded"` on
`/api/v1/health/ready` until you train a model** (see below) and mount it at
`ml_artifacts/model_bundle/bundle.json` -- the platform is fully functional (auth, DB, frontend,
docs) without a model; predictions alone return `503` until one is trained.

## Local development (without Docker)

### 1. ML package

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r ml/requirements.txt
```

Train a model (requires a `dataset_blood_group/` directory -- see [MODEL_CARD.md](MODEL_CARD.md)
for the expected layout and the scientific caveat before sourcing real data):

```bash
python -m ml.training.train --dataset-root dataset_blood_group
```

This runs subject-grouped k-fold CV per backbone, fits the ensemble combiner and calibration on
out-of-fold predictions, evaluates once on a held-out test set, and writes
`artifacts/model_bundle/bundle.json` plus a full metrics report and plots.

Run the ML test suite:

```bash
PYTHONPATH=. pytest ml/tests/ -v
```

### 2. Backend

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env   # SQLite by default -- no Postgres/Redis needed for local dev
alembic upgrade head
uvicorn app.main:app --reload
```

API docs: `http://localhost:8000/api/docs` (Swagger) or `/api/redoc`.

Run the backend test suite (uses an isolated SQLite DB and a deliberately-unready model, so it
never requires a trained bundle):

```bash
pytest tests/ -v
```

### 3. Frontend

```bash
cd frontend
npm install
npm run dev
```

Vite's dev server proxies `/api` and `/static` to `localhost:8000` (see `vite.config.js`).

## Changelog

- **UI copy refinement.** All dashboard/analytics/disclaimer text was rewritten for a more
  natural, professional tone (e.g. "Run prediction" → "Analyze Fingerprint", "Analytics" →
  "Prediction Insights"). Wording only -- no layout, color, spacing, typography, or functionality
  changed. Applied consistently across all four locales (`frontend/src/i18n/{en,es,fr,hi}.json`),
  not just English.

## Future work

- **Login / registration UI.** The backend fully implements and enforces JWT auth (register,
  login, refresh, `/auth/me`) -- see `backend/app/api/v1/endpoints/auth.py` and
  `backend/tests/test_auth.py`. The frontend currently skips showing a login screen: on first
  load, `AuthContext` silently provisions a random "guest" account so trying a prediction has zero
  signup friction. That account is real (a real row in `users`, a real JWT), it's just never shown
  to the person using the app. To bring back a visible login/register flow: restore the
  `ProtectedRoute` wrapper and the `/login`, `/register` routes in `frontend/src/App.jsx` (both
  page components and the guard are still in the codebase, just unwired) and re-add a logout
  control to `Navbar`.
- **Email/password recovery, OAuth providers, and account settings** are not implemented.
- **Server-side analytics aggregation.** `AnalyticsPage` currently aggregates client-side over the
  most recent 100 predictions; a real deployment with many users would want a dedicated aggregate
  endpoint instead.

## Troubleshooting

- **`ModuleNotFoundError: No module named 'greenlet'`** during `alembic upgrade head` or on
  backend startup: SQLAlchemy's async engine requires `greenlet`, which is pinned explicitly in
  `backend/requirements.txt`. If it's still missing after `pip install -r requirements.txt`,
  check your Python version -- `greenlet` (and other compiled-extension packages like `bcrypt`,
  `asyncpg`) can lag behind brand-new CPython releases before prebuilt wheels are available. This
  project is developed and tested against **Python 3.12** (see `backend/.python-version`); if
  you're on something newer (3.13/3.14+) and hit install failures on compiled dependencies,
  installing 3.12 alongside your system Python (e.g. via `pyenv install 3.12`) is the fastest fix.
- **`ModuleNotFoundError: No module named 'ml'`** when starting the backend: this was a real bug
  in earlier versions of this repo (a path-resolution off-by-one in
  `backend/app/services/ml_service.py`) and is fixed as of this version -- `ml_service.py` now
  locates the project root by searching upward for a directory containing both `ml/` and
  `backend/`, rather than assuming a fixed directory depth. If you see this error, confirm you're
  running the code from this section onward and that `ml/__init__.py` exists.
- **`"model": "not loaded"` from `/api/v1/health/ready`**: expected until you train a model (see
  "Train a model" above) or set `BP_MODEL_BUNDLE_PATH` to point at an existing `bundle.json`.
  Every other feature (auth, history, docs) works without a trained model; only the predict
  endpoint itself returns `503` until one exists.

## Tech stack

| Layer | Choices |
|---|---|
| ML | PyTorch, timm (EfficientNetV2 / ViT / ConvNeXt), scikit-learn, OpenCV, albumentations, MLflow, TensorBoard |
| Backend | FastAPI, SQLAlchemy 2.0 (async), PostgreSQL/SQLite, Redis, Alembic, bcrypt + python-jose, Prometheus, reportlab |
| Frontend | React 18, Vite, Tailwind CSS, recharts, react-dropzone, react-i18next, react-router |
| Infra | Docker, Docker Compose, nginx, gunicorn + uvicorn workers, GitHub Actions |

## Why PyTorch + timm instead of the original TensorFlow/Keras stack

The original project used TensorFlow/Keras. This rebuild uses PyTorch + `timm` instead, because
`timm` gives EfficientNetV2, ViT, and ConvNeXt pretrained ImageNet weights behind one consistent
API -- native `tf.keras.applications` doesn't cleanly cover all three today. Everything downstream
(mixed precision, Grad-CAM, attention rollout, TTA) follows from that choice.

## License

MIT -- see [LICENSE](LICENSE). This covers the code. It does not, and cannot, certify the
underlying prediction task as scientifically or medically valid -- see
[MODEL_CARD.md](MODEL_CARD.md).
