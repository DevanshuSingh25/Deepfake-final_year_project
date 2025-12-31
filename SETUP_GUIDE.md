# Video Detection ML Integration - Setup Guide

## Quick Start

### 1. Backend Setup (FastAPI Server)

Run the automated setup script:

```powershell
.\setup_venv.bat
```

This will:
- Create a Python virtual environment
- Install all required dependencies (PyTorch, FastAPI, OpenCV, etc.)

**Manual Setup (alternative)**:
```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Start the Backend Server

```powershell
# Activate virtual environment (if not already activated)
venv\Scripts\activate

# Start FastAPI server
python -m uvicorn server:app --reload --port 8000
```

The server will be available at:
- API: `http://localhost:8000`
- Interactive docs: `http://localhost:8000/docs`

### 3. Start the Frontend

In a separate terminal:

```powershell
npm run dev
```

The frontend will be available at `http://localhost:8080`

---

## Production Deployment

### Environment Configuration

#### Backend (.env file)
```env
FRONTEND_URL=https://your-production-frontend.com
```

#### Frontend (.env or .env.local file)
```env
VITE_BACKEND_URL=https://your-production-backend.com
```

### CORS Configuration

The system is production-ready with flexible CORS:

**Development (default)**:
- Backend automatically allows: `localhost:8080`, `localhost:5173`
- Frontend automatically uses: `localhost:8000`

**Production**:
- Backend reads `FRONTEND_URL` from `.env` and adds it to allowed origins
- Frontend reads `VITE_BACKEND_URL` and uses it for API calls
- Leave variables blank for development, set them for production

---

## Available Models

The system includes 6 pre-trained models:

| Frames | Accuracy | File |
|--------|----------|------|
| 10 | 84% | `model_84_acc_10_frames_final_data.pt` |
| 20 | 90% | `model_90_acc_20_frames_FF_data.pt` |
| 40 | 95% | `model_95_acc_40_frames_FF_data.pt` |
| 60 | 97% | `model_97_acc_60_frames_FF_data.pt` |
| 80 | 97% | `model_97_acc_80_frames_FF_data.pt` |
| 100 | 97% | `model_97_acc_100_frames_FF_data.pt` |

The correct model is automatically selected based on the frame sampling rate chosen in the frontend.

---

## API Endpoints

### POST `/api/predict`
Upload a video for deepfake detection.

**Request**:
- `file`: Video file (multipart/form-data)
- `sequence_length`: Number of frames (10, 20, 40, 60, 80, 100)
- `face_focus`: Enable face detection (boolean)

**Response**:
```json
{
  "prediction": "REAL" | "FAKE",
  "confidence": 95.7,
  "sequence_length": 20,
  "device": "cpu"
}
```

### GET `/api/models`
List all available models.

**Response**:
```json
{
  "available_models": [
    {
      "filename": "model_84_acc_10_frames_final_data.pt",
      "frames": 10,
      "accuracy": "84%"
    },
    ...
  ],
  "total": 6
}
```

---

## Architecture

### Frontend
- **React + TypeScript + Vite**
- **SettingsPanel.tsx**: Dropdown with 10, 20, 40, 60, 80, 100 frame options
- **VideoDetect.tsx**: Handles file upload and API communication
- **api.ts**: API client with `predictVideo()` function

### Backend
- **server.py**: FastAPI application with CORS and prediction endpoint
- **model_utils.py**: Model loading, caching, and device detection
- **preprocessing.py**: Video frame extraction and face detection
- **Model Architecture**: ResNeXt50 + LSTM

---

## Troubleshooting

### "No module named 'torch'"
Run `pip install -r requirements.txt` in the virtual environment

### CORS errors in browser
1. Check that backend is running on port 8000
2. Check that frontend is running on port 8080
3. Verify `.env` configuration if using production URLs

### "Failed to load model"
Ensure all `.pt` model files are in the `models/` directory

### Out of memory errors
- Use lower frame counts (10 or 20 instead of 80 or 100)
- Models run on CPU by default; GPU support is automatic if CUDA is available

---

## File Structure

```
Deepfake-proj-main/
├── models/                    # Pre-trained ML models (.pt files)
├── reference code/            # Original Django preprocessing code
├── src/                       # React frontend source
│   ├── components/
│   │   └── SettingsPanel.tsx  # Frame rate dropdown (updated)
│   ├── pages/
│   │   └── VideoDetect.tsx    # Video upload & analysis (updated)
│   └── lib/
│       └── api.ts             # API client (updated)
├── server.py                  # FastAPI backend (NEW)
├── model_utils.py             # Model utilities (NEW)
├── preprocessing.py           # Video preprocessing (NEW)
├── requirements.txt           # Python dependencies (NEW)
├── setup_venv.bat            # Setup script (NEW)
├── .env                      # Backend config (UPDATED)
└── .env.local                # Frontend config (CREATE)
```

---

## Testing

1. **Start both servers** (backend on 8000, frontend on 8080)
2. **Navigate to** `/video` page
3. **Upload a video** (MP4, MOV, MKV, or WEBM)
4. **Select frame sampling rate** (default: 20 frames)
5. **Click "Analyze Video"**
6. **View results** with prediction (REAL/FAKE) and confidence percentage

---

## Dependencies

### Python (Backend)
- FastAPI 0.115.0
- Uvicorn 0.32.0
- PyTorch 2.5.1
- TorchVision 0.20.1
- OpenCV 4.10.0
- face-recognition 1.3.0

### Node.js (Frontend)
- React 18.3.1
- TypeScript 5.8.3
- Vite 5.4.19
- (See package.json for full list)

---

## License

This project is for educational and demonstration purposes.
