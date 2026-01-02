from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import tempfile
import shutil
from dotenv import load_dotenv
from model_utils import load_model, get_device
from preprocessing import preprocess_video, predict

# Audio deepfake detection imports (separate from video pipeline)
from audio_predict import predict_audio, AudioPredictionError
from audio_preprocessing import AudioValidationError, AudioLoadError

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Deepfake Detection API",
    description="Video and Audio deepfake detection API",
    version="1.0.0"
)

# CORS configuration for production-ready deployment
# Get frontend URL from environment variable (blank for development)
frontend_url = os.getenv("FRONTEND_URL", "").strip()

# Build CORS allowed origins list
allowed_origins = [
    "http://localhost:8080",      # Vite dev server default
    "http://localhost:5173",      # Alternative Vite port
    "http://127.0.0.1:8080",
    "http://127.0.0.1:5173",
]

# Add production frontend URL if specified
if frontend_url:
    allowed_origins.append(frontend_url)
    print(f"✓ Production frontend URL added to CORS: {frontend_url}")
else:
    print("✓ Development mode: Using localhost CORS origins")

# Configure CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("\n=== Deepfake Detection Server Configuration ===")
print(f"Allowed CORS origins: {allowed_origins}")
print(f"Device: {get_device()}")
print("=" * 50 + "\n")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "Deepfake Detection API",
        "version": "1.1.0",
        "device": get_device(),
        "capabilities": ["video", "audio"]
    }


@app.post("/api/predict")
async def predict_video_endpoint(
    file: UploadFile = File(...),
    sequence_length: int = Form(...),
    face_focus: bool = Form(True)
):
    """
    Predict whether a video is real or fake.
    
    Args:
        file: Video file to analyze
        sequence_length: Number of frames to extract (10, 20, 40, 60, 80, 100)
        face_focus: Whether to focus on faces (currently always enabled)
    
    Returns:
        JSON response with prediction result and confidence
    """
    temp_video_path = None
    
    try:
        # Validate sequence length
        valid_lengths = [10, 20, 40, 60, 80, 100]
        if sequence_length not in valid_lengths:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid sequence_length. Must be one of {valid_lengths}"
            )
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('video/'):
            raise HTTPException(
                status_code=400,
                detail="File must be a video"
            )
        
        print(f"\n{'='*50}")
        print(f"Processing video: {file.filename}")
        print(f"Sequence length: {sequence_length} frames")
        print(f"Face focus: {face_focus}")
        print(f"{'='*50}\n")
        
        # Save uploaded video to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_video_path = temp_file.name
        
        print(f"✓ Video saved to: {temp_video_path}")
        
        # Load model for the specified sequence length
        device = get_device()
        model = load_model(sequence_length, device)
        
        if model is None:
            raise HTTPException(
                status_code=500,
                detail=f"Failed to load model for {sequence_length} frames"
            )
        
        print(f"✓ Model loaded successfully")
        
        # Preprocess video
        print(f"⏳ Preprocessing video...")
        frames_tensor, preprocessed_images, face_cropped_images, faces_found = preprocess_video(
            temp_video_path,
            sequence_length,
            save_preprocessed=False  # Set to True if you want to save frames
        )
        
        print(f"✓ Preprocessing complete")
        
        # Make prediction
        print(f"⏳ Running prediction...")
        prediction_int, confidence = predict(model, frames_tensor, device)
        
        # Convert prediction to label
        prediction_label = "REAL" if prediction_int == 1 else "FAKE"
        
        print(f"\n{'='*50}")
        print(f"✓ PREDICTION: {prediction_label}")
        print(f"✓ CONFIDENCE: {confidence:.1f}%")
        print(f"{'='*50}\n")
        
        # Return response with frame images for frontend display
        # Limit to max 6 frames to keep response size reasonable
        display_frames = face_cropped_images[:6] if len(face_cropped_images) > 6 else face_cropped_images
        
        return JSONResponse(content={
            "prediction": prediction_label,
            "confidence": round(confidence, 1),
            "sequence_length": sequence_length,
            "device": device,
            "faces_found": faces_found,
            "total_frames_analyzed": len(face_cropped_images),
            "frame_images": display_frames
        })
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"\n❌ Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
    finally:
        # Clean up temporary file
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.unlink(temp_video_path)
                print(f"✓ Cleaned up temporary file")
            except Exception as e:
                print(f"⚠ Warning: Could not delete temporary file: {e}")


# =============================================================================
# AUDIO DEEPFAKE DETECTION ENDPOINT
# =============================================================================

@app.post("/api/audio/predict")
async def predict_audio_endpoint(file: UploadFile = File(...)):
    """
    Predict whether an audio file is real or fake (deepfake).
    
    Uses MelodyMachine/Deepfake-audio-detection-V2 (Wav2Vec2-based model).
    
    Args:
        file: Audio file to analyze (WAV, MP3, FLAC, M4A, OGG supported)
    
    Returns:
        JSON response with prediction result and confidence:
        {
            "prediction": "REAL" | "FAKE",
            "confidence": float (0-100),
            "model": "MelodyMachine/Deepfake-audio-detection-V2",
            "all_scores": {"real": float, "fake": float}
        }
    """
    temp_audio_path = None
    
    try:
        # Validate content type
        if file.content_type and not file.content_type.startswith('audio/'):
            raise HTTPException(
                status_code=400,
                detail="File must be an audio file"
            )
        
        print(f"\n{'='*50}")
        print(f"Processing audio: {file.filename}")
        print(f"Content type: {file.content_type}")
        print(f"{'='*50}\n")
        
        # Get file extension from filename
        file_ext = os.path.splitext(file.filename)[1] if file.filename else '.wav'
        
        # Save uploaded audio to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_audio_path = temp_file.name
        
        print(f"✓ Audio saved to: {temp_audio_path}")
        
        # Run audio prediction
        print(f"⏳ Running audio deepfake detection...")
        result = predict_audio(temp_audio_path, file.content_type)
        
        print(f"\n{'='*50}")
        print(f"✓ AUDIO PREDICTION: {result['prediction']}")
        print(f"✓ CONFIDENCE: {result['confidence']:.1f}%")
        print(f"{'='*50}\n")
        
        return JSONResponse(content=result)
    
    except AudioValidationError as e:
        print(f"\n❌ Audio validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except AudioLoadError as e:
        print(f"\n❌ Audio load error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    
    except AudioPredictionError as e:
        print(f"\n❌ Audio prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    except HTTPException:
        raise
    
    except Exception as e:
        print(f"\n❌ Error during audio prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Audio prediction failed: {str(e)}"
        )
    
    finally:
        # Clean up temporary file
        if temp_audio_path and os.path.exists(temp_audio_path):
            try:
                os.unlink(temp_audio_path)
                print(f"✓ Cleaned up temporary audio file")
            except Exception as e:
                print(f"⚠ Warning: Could not delete temporary audio file: {e}")


@app.get("/api/models")
async def list_available_models():
    """List all available models and their frame counts"""
    import glob
    models_dir = "models"
    model_files = glob.glob(os.path.join(models_dir, "*.pt"))
    
    models_info = []
    for model_path in model_files:
        filename = os.path.basename(model_path)
        try:
            parts = filename.split("_")
            accuracy = parts[1]
            frames = parts[3]
            models_info.append({
                "filename": filename,
                "frames": int(frames),
                "accuracy": f"{accuracy}%"
            })
        except (IndexError, ValueError):
            continue
    
    return {
        "available_models": sorted(models_info, key=lambda x: x["frames"]),
        "total": len(models_info)
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
