# 🚦 Traffic-Aware-ETA-Prediction-Using-Temporal-Deep-Learning
End-to-end deep learning system that predicts Estimated Time of Arrival (ETA) using real-world freeway sensor data (PEMS-BAY), deployed via FastAPI and Docker. Includes multi-segment route ETA simulation inspired by Uber's dispatch infrastructure.

---

## ✨ Project Highlights

* 📅**Real-time ETA Prediction** using 5-min interval PEMS-BAY traffic speed data
* 🧠 **Deep Learning with PyTorch Lightning**: LSTM-based model
* 🌎 **Production-ready API**: FastAPI + Pydantic + Docker
* 📊 **Evaluation**: MAE ≈ 0.026 min, RMSE ≈ 0.068 min
* ⚖️ **Multi-segment Routing**: /predict\_route returns ETA for full route across multiple segments

---

## Step-by-Step Implementation

### 1. Environment Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Data Source: PEMS-BAY

* Dataset used: [PEMS-BAY.h5](https://zenodo.org/record/4263971)
* Data format: 5-min interval average speeds across 325 freeway sensors (May-July 2017)

### 3. Data Inspection

```bash
python src/data/load_and_inspect.py
```

* Inspects range, shape: (52116, 325)
* Detects missing/zero speed values

### 4. Cleaning and Normalization

```bash
python src/data/clean_pems.py
```

* Replaces 0 with NaN, interpolates
* Normalizes speed data (optional)

Data Note
To save GitHub storage, `data/cleaned_pems_bay.csv` is not included.

### 5. Create Model Input Windows

```bash
python src/data/make_windows.py
```

* Sliding window of 12 steps (1 hour) as input
* Simulates ETA (in minutes) = (1/speed) \* 60
* Filters outliers (ETA > 30 min)

### 6. Train LSTM Model

```bash
python src/model/train_lstm.py
```

* Uses PyTorch Lightning
* Saves model to models/lstm\_eta\_model.pth

### 7. Evaluate on Validation Set

```bash
python src/model/eval_lstm.py
```

* Logs MAE, RMSE
* Saves predictions to data/predictions.npz

### 8. Serve Model via FastAPI

```bash
uvicorn src.api.app:app --reload --reload-dir src
```

* `/predict` for single sensor
* `/predict_route` for multi-segment input

#### Example:

```json
{
  "segments": [
    [71.4, 71.6, 71.6, ..., 70.9],
    [65.4, 64.1, ..., 63.9],
    [58.3, ..., 57.8]
  ]
}
```

Returns:

```json
{
  "segment_etas": [0.84, 1.2, 1.5],
  "total_eta": 3.54
}
```

### 9. Dockerize API

```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
COPY src ./src
COPY models ./models
COPY data ./data
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t eta-api .
docker run -p 8000:8000 eta-api
```

---

## Deployment

* Hugging Face Spaces (via Docker image)

---

## Evaluation Metrics

```json
MAE:  0.026 minutes
RMSE: 0.068 minutes
```

---

## Future Work

* Add support for dynamic segment lengths
* Integrate GNN-based road graphs
* Use OpenStreetMap or simulated GPS traces for route pathing

---

## Screenshots 

* Swagger UI (/docs)
  ![Screenshot 2025-07-01 013617](https://github.com/user-attachments/assets/e51787f8-bc91-4a83-8a84-5efd786884da)

* Route ETA JSON input/output
  ![Screenshot 2025-06-30 234911](https://github.com/user-attachments/assets/cd0dafda-87ed-46fc-9fa2-663d87fe2c0b)

  ![Screenshot 2025-07-01 012127](https://github.com/user-attachments/assets/351af1d2-68f1-4201-aa83-7a359e4b0595)

* Graphs from eval\_lstm.py
  ![image](https://github.com/user-attachments/assets/cb6862f9-49e5-45cf-b848-016f2b58c2a6)

---

> ⚡ Ready to deploy | 📈 Evaluated | 📆 Reproducible | 🚀 Scalable
