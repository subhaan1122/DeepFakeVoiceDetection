# Deepfake Voice Detection

This project now uses a FastAPI backend and React frontend, while reusing existing trained artifacts in the artifacts folder.

## Structure

- backend: FastAPI service
- frontend: React + Vite client
- artifacts: trained models and preprocessing objects
- features and plots: training outputs

## Backend Run (PowerShell)

cd D:/Deep Fake Voice Detection/backend
pip install -r requirements.txt
python run.py

Backend API: http://localhost:8000

## Frontend Run (PowerShell)

cd D:/Deep Fake Voice Detection/frontend
npm install
npm run dev

Frontend UI: http://localhost:5173

## Required Endpoints

- GET /api/v1/health
- GET /api/v1/ready
- GET /api/v1/models/info
- POST /api/v1/predict/audio
- POST /api/v1/predict/audio/batch

## Quick Verification

curl.exe -X GET "http://localhost:8000/api/v1/health"

curl.exe -X POST "http://localhost:8000/api/v1/predict/audio" -F "audio_file=@D:/path/to/test.wav"

## Compatibility Note

This system is an API/UI refactor of the existing inference stack and reuses existing trained model artifacts directly.
