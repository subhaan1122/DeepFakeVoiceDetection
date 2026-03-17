# Deepfake Voice Detection

Deepfake Voice Detection is a full-stack project with a FastAPI backend and a React (Vite) frontend for audio-based fake/real voice analysis.

## Project Layout

- backend: FastAPI API service and inference orchestration
- frontend: React + Vite user interface
- app.py and training scripts: original workflow and model/training utilities
- artifacts, features, plots: local generated outputs (ignored from Git)

## What Was Recently Updated

- Frontend mobile optimization:
	- improved touch target sizes for interactive controls
	- better small-screen layout behavior
	- safe-area handling for notched mobile devices
- Repository hygiene:
	- added and hardened .gitignore
	- removed generated artifacts and metadata files from tracking

## Run Backend (PowerShell)

```powershell
cd D:/Deep Fake Voice Detection/backend
pip install -r requirements.txt
python run.py
```

Backend base URL:
- http://localhost:8000

## Run Frontend (PowerShell)

```powershell
cd D:/Deep Fake Voice Detection/frontend
npm install
npm run dev
```

Frontend URL:
- http://localhost:5173

## API Endpoints

- GET /api/v1/health
- GET /api/v1/ready
- GET /api/v1/models/info
- POST /api/v1/predict

## Quick Checks

```powershell
curl.exe -X GET "http://localhost:8000/api/v1/health"
curl.exe -X POST "http://localhost:8000/api/v1/predict" -F "file=@D:/path/to/test.wav"
```

## Privacy and Git Policy

The repository is configured to avoid committing local/private generated content, including:

- artifacts/
- plots/
- frontend/node_modules/
- frontend build/dev caches
- generated feature CSV files
- metadata CSV files that can contain local absolute paths

If you need to share dataset metadata, export a sanitized copy that removes personal paths or identifiers.

## Notes

- Inference requires model artifacts to exist locally in the expected artifacts path.
- This project keeps training/inference scripts and web app code in one repository.
