# Quantum Drought Intelligence — Streamlit Frontend

This is a Streamlit frontend for the supplied Flask backend.

## Project files

Keep these files in the same project directory:
- `app.py` (this frontend)
- `backend.py`
- `heatmap_generator.py`
- the model files `gatekeeper.pth` and `drought_model.pth`
- the backend's required Python packages and any other project files

## Install

Use the same Python environment as the backend if possible:

```bash
pip install -r requirements-frontend.txt
```

Your backend also needs its own dependencies, including Flask, flask-cors, numpy, torch, PennyLane, OpenCV, and Pillow.

## Run

Terminal 1 — start the Flask API:

```bash
python backend.py
```

Terminal 2 — start Streamlit from the same project folder:

```bash
streamlit run app.py
```

Open the local Streamlit URL printed in the terminal (usually http://localhost:8501).

## API configuration

By default the frontend calls `http://localhost:5000`. To use a different backend address:

PowerShell:
```powershell
$env:DROUGHT_API_URL="http://127.0.0.1:5000"
streamlit run app.py
```

## Important notes from the supplied backend

- Prediction uses `POST /predict` with multipart form field `file`.
- Heatmap uses `POST /heatmap` with multipart form field `file` and returns a PNG.
- The frontend does not retrain the models.
- Ensure the trained weight files are present; the supplied backend warns if they are missing.
- The supplied `backend.py` calls `generate_heatmap(temp_path, "static/heatmap_result.png")`, while the uploaded `heatmap_generator.py` defines `generate_heatmap(image_path)` with one argument. That mismatch must be fixed in the backend or heatmap generator before the heatmap button can work. This frontend reports the error returned by the API.
- The API does not define a root `/` route, so the sidebar connection check may report HTTP 404 even when the server is reachable. A 404 response still means the server responded.
