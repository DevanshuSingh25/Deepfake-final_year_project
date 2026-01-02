"""
Audio Deepfake Detection - Prediction Module

Handles audio inference using the HuggingFace pipeline.
Returns predictions in the format expected by the API.

This module is completely separate from the video prediction logic.
"""

from typing import Dict, Any
import logging

from audio_model_utils import load_audio_pipeline, get_audio_model_info, AUDIO_MODEL_ID
from audio_preprocessing import (
    preprocess_audio,
    cleanup_temp_wav,
    AudioValidationError,
    AudioLoadError
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AudioPredictionError(Exception):
    """Raised when audio prediction fails."""
    pass


def predict_audio(file_path: str, content_type: str = None) -> Dict[str, Any]:
    """
    Predict whether an audio file is real or fake.
    
    Args:
        file_path: Path to the audio file
        content_type: Optional MIME type from upload
        
    Returns:
        Dictionary with prediction results:
        {
            "prediction": "REAL" | "FAKE",
            "confidence": float (0-100),
            "model": "MelodyMachine/Deepfake-audio-detection-V2",
            "all_scores": {"real": float, "fake": float}
        }
        
    Raises:
        AudioValidationError: If audio validation fails
        AudioLoadError: If audio loading fails
        AudioPredictionError: If inference fails
    """
    wav_path = None
    try:
        # Preprocess and validate audio (converts to WAV if needed)
        logger.info(f"Starting audio prediction for: {file_path}")
        wav_path = preprocess_audio(file_path, content_type)
        
        # Load the classification pipeline
        pipeline = load_audio_pipeline()
        
        # Run inference
        # The pipeline returns a list of dicts: [{"label": "fake", "score": 0.12}, ...]
        logger.info("Running audio classification...")
        results = pipeline(wav_path)
        
        logger.info(f"Raw prediction results: {results}")
        
        # Apply temperature scaling to soften extreme probabilities
        # This makes the confidence scores more human-friendly
        # Higher temperature = softer probabilities
        TEMPERATURE = 3.0  # Adjust this value: 1.0 = no change, higher = softer
        
        import math
        raw_scores = {item["label"]: item["score"] for item in results}
        
        # Convert probabilities to logits, apply temperature, then back to probabilities
        # Using log-odds transformation for binary classification
        real_score = raw_scores.get("real", 0.5)
        fake_score = raw_scores.get("fake", 0.5)
        
        # Clamp to avoid log(0)
        real_score = max(min(real_score, 0.9999999), 0.0000001)
        fake_score = max(min(fake_score, 0.9999999), 0.0000001)
        
        # Apply temperature scaling via softmax
        real_logit = math.log(real_score)
        fake_logit = math.log(fake_score)
        
        # Scale by temperature
        real_scaled = real_logit / TEMPERATURE
        fake_scaled = fake_logit / TEMPERATURE
        
        # Softmax to get calibrated probabilities
        max_logit = max(real_scaled, fake_scaled)
        real_exp = math.exp(real_scaled - max_logit)
        fake_exp = math.exp(fake_scaled - max_logit)
        total = real_exp + fake_exp
        
        scaled_real = real_exp / total
        scaled_fake = fake_exp / total
        
        logger.info(f"Temperature-scaled scores: real={scaled_real:.4f}, fake={scaled_fake:.4f}")
        
        # Get top prediction (use original for decision, scaled for display)
        top_label = "real" if real_score > fake_score else "fake"
        prediction_label = top_label.upper()
        confidence = (scaled_real if top_label == "real" else scaled_fake) * 100
        
        result = {
            "prediction": prediction_label,
            "confidence": round(confidence, 2),
            "model": AUDIO_MODEL_ID,
            "all_scores": {
                "real": round(scaled_real * 100, 2),
                "fake": round(scaled_fake * 100, 2)
            }
        }
        
        logger.info(f"Prediction complete: {prediction_label} ({confidence:.1f}%)")
        
        return result
        
    except (AudioValidationError, AudioLoadError):
        # Re-raise validation/load errors as-is
        raise
    except Exception as e:
        logger.error(f"Audio prediction failed: {e}")
        raise AudioPredictionError(f"Prediction failed: {e}")
    finally:
        # Clean up temporary WAV file if one was created
        if wav_path and wav_path != file_path:
            cleanup_temp_wav(file_path, wav_path)


def get_model_status() -> Dict[str, Any]:
    """
    Get the current status of the audio model.
    
    Returns:
        Dictionary with model info and status
    """
    from audio_model_utils import is_audio_pipeline_loaded
    
    info = get_audio_model_info()
    info["loaded"] = is_audio_pipeline_loaded()
    
    return info
