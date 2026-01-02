"""
Audio Deepfake Detection - Model Utilities

Handles loading and caching of the HuggingFace audio classification pipeline.
Uses MelodyMachine/Deepfake-audio-detection-V2 (Wav2Vec2-based model).

This module is completely separate from the video detection model_utils.py.
"""

from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model configuration
AUDIO_MODEL_ID = "MelodyMachine/Deepfake-audio-detection-V2"
AUDIO_MODEL_TASK = "audio-classification"

# Global pipeline cache
_audio_pipeline = None


def load_audio_pipeline():
    """
    Load and cache the audio classification pipeline.
    
    Uses HuggingFace's pipeline API which handles:
    - Model downloading (first run only)
    - Feature extraction (Wav2Vec2FeatureExtractor)
    - Automatic resampling to 16kHz
    - Input normalization
    
    Returns:
        Pipeline object for audio classification
        
    Raises:
        RuntimeError: If pipeline loading fails
    """
    global _audio_pipeline
    
    if _audio_pipeline is not None:
        logger.info("Using cached audio classification pipeline")
        return _audio_pipeline
    
    try:
        from transformers import pipeline
        
        logger.info(f"Loading audio classification pipeline: {AUDIO_MODEL_ID}")
        
        # Load pipeline with CPU device (-1 forces CPU)
        # The pipeline handles all preprocessing automatically
        _audio_pipeline = pipeline(
            task=AUDIO_MODEL_TASK,
            model=AUDIO_MODEL_ID,
            device=-1  # Force CPU for compatibility
        )
        
        logger.info("Audio classification pipeline loaded successfully")
        return _audio_pipeline
        
    except Exception as e:
        logger.error(f"Failed to load audio pipeline: {e}")
        raise RuntimeError(f"Failed to load audio classification model: {e}")


def get_audio_model_info() -> dict:
    """
    Get information about the audio detection model.
    
    Returns:
        Dictionary with model metadata
    """
    return {
        "model_id": AUDIO_MODEL_ID,
        "task": AUDIO_MODEL_TASK,
        "architecture": "Wav2Vec2ForSequenceClassification",
        "base_model": "facebook/wav2vec2-base",
        "sample_rate": 16000,
        "labels": ["fake", "real"],
        "reported_accuracy": 0.997
    }


def is_audio_pipeline_loaded() -> bool:
    """Check if the audio pipeline is already loaded in memory."""
    return _audio_pipeline is not None


def unload_audio_pipeline() -> None:
    """
    Unload the audio pipeline from memory.
    Useful for freeing memory if needed.
    """
    global _audio_pipeline
    if _audio_pipeline is not None:
        _audio_pipeline = None
        logger.info("Audio pipeline unloaded from memory")
