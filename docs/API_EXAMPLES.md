# API Usage Examples

Full interactive docs live at `/api/docs` (Swagger) once the backend is running. This is a quick
curl-based walkthrough of the main flow: register → login → predict → history → PDF report.

Base URL below assumes local dev (`http://localhost:8000`); swap in your deployed host otherwise.

## 1. Register

```bash
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{"email": "researcher@example.com", "password": "correct-horse-battery-staple", "full_name": "A. Researcher"}'
```

## 2. Log in

```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "researcher@example.com", "password": "correct-horse-battery-staple"}'
```

Response includes `access_token` (short-lived) and `refresh_token` (longer-lived). Export the
access token for the following requests:

```bash
export TOKEN="<access_token from the response above>"
```

## 3. Run a prediction

```bash
curl -X POST "http://localhost:8000/api/v1/predictions?use_tta=true&explain=true" \
  -H "Authorization: Bearer $TOKEN" \
  -F "file=@/path/to/fingerprint.png"
```

Returns a `503` with a clear `detail` message if no model bundle has been trained yet (see the
main README's "Train a model" section) -- this is expected on a fresh checkout, not a bug.

## 4. List prediction history

```bash
curl "http://localhost:8000/api/v1/predictions?page=1&page_size=20" \
  -H "Authorization: Bearer $TOKEN"
```

## 5. Get a specific prediction

```bash
curl "http://localhost:8000/api/v1/predictions/<prediction_id>" \
  -H "Authorization: Bearer $TOKEN"
```

## 6. Download the PDF report

```bash
curl "http://localhost:8000/api/v1/predictions/<prediction_id>/report.pdf" \
  -H "Authorization: Bearer $TOKEN" \
  -o report.pdf
```

## 7. Refresh an expired access token

```bash
curl -X POST http://localhost:8000/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d "{\"refresh_token\": \"<refresh_token>\"}"
```

## 8. Health checks

```bash
curl http://localhost:8000/api/v1/health/live    # process is up
curl http://localhost:8000/api/v1/health/ready   # process + DB + model are all usable
```

## 9. Prometheus metrics

```bash
curl http://localhost:8000/metrics
```
