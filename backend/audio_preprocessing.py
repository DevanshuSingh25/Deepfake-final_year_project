"""
Audio Deepfake Detection - Preprocessing Utilities

Handles audio file loading, validation, and preprocessing.
Converts non-WAV formats to WAV using ffmpeg for compatibility
with the HuggingFace pipeline.

This module is completely separate from the video preprocessing.py.
"""

import os
import subprocess
import tempfile
from typing import Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Audio configuration
REQUIRED_SAMPLE_RATE = 16000  # Wav2Vec2 requires 16kHz

# Supported audio formats
SUPPORTED_EXTENSIONS = {'.wav', '.mp3', '.flac', '.m4a', '.ogg', '.wma', '.aac'}
SUPPORTED_MIME_TYPES = {
    'audio/wav', 'audio/x-wav', 'audio/wave',
    'audio/mpeg', 'audio/mp3',
    'audio/flac', 'audio/x-flac',
    'audio/mp4', 'audio/x-m4a', 'audio/m4a',
    'audio/ogg', 'audio/vorbis',
    'audio/x-ms-wma',
    'audio/aac'
}


class AudioValidationError(Exception):
    """Raised when audio validation fails."""
    pass


class AudioLoadError(Exception):
    """Raised when audio loading fails."""
    pass


def validate_audio_file(file_path: str, content_type: Optional[str] = None) -> None:
    """
    Validate an audio file before processing.
    
    Args:
        file_path: Path to the audio file
        content_type: Optional MIME type from upload
        
    Raises:
        AudioValidationError: If validation fails
    """
    # Check file exists
    if not os.path.exists(file_path):
        raise AudioValidationError("Audio file not found")
    
    # Check file size (max 50MB)
    file_size = os.path.getsize(file_path)
    if file_size == 0:
        raise AudioValidationError("Audio file is empty")
    if file_size > 50 * 1024 * 1024:  # 50MB
        raise AudioValidationError("Audio file too large (max 50MB)")
    
    # Check extension
    _, ext = os.path.splitext(file_path)
    ext = ext.lower()
    if ext and ext not in SUPPORTED_EXTENSIONS:
        raise AudioValidationError(
            f"Unsupported audio format: {ext}. "
            f"Supported formats: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    
    # Check MIME type if provided
    if content_type:
        # Normalize content type (remove charset, etc.)
        base_type = content_type.split(';')[0].strip().lower()
        if base_type not in SUPPORTED_MIME_TYPES and not base_type.startswith('audio/'):
            raise AudioValidationError(
                f"Invalid content type: {content_type}. Must be an audio file."
            )
    
    logger.info(f"Audio file validated: {file_path} ({file_size} bytes)")


def convert_to_wav(input_path: str) -> str:
    """
    Convert audio file to WAV format using ffmpeg.
    
    The HuggingFace pipeline uses soundfile internally, which only supports
    WAV, FLAC, OGG. For other formats (MP3, M4A, AAC), we need to convert.
    
    Args:
        input_path: Path to the input audio file
        
    Returns:
        Path to the WAV file (either original or converted)
        
    Raises:
        AudioLoadError: If conversion fails
    """
    _, ext = os.path.splitext(input_path)
    ext = ext.lower()
    
    # Formats that soundfile can read directly
    if ext in {'.wav', '.flac', '.ogg'}:
        logger.info(f"File format {ext} is directly supported, no conversion needed")
        return input_path
    
    # Need to convert to WAV
    logger.info(f"Converting {ext} to WAV using ffmpeg...")
    
    # Create temp file for WAV output
    temp_wav = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
    temp_wav.close()
    output_path = temp_wav.name
    
    try:
        # Use ffmpeg to convert to WAV (16kHz, mono, PCM)
        result = subprocess.run(
            [
                'ffmpeg', '-y',  # Overwrite output
                '-i', input_path,  # Input file
                '-ar', str(REQUIRED_SAMPLE_RATE),  # Sample rate 16kHz
                '-ac', '1',  # Mono
                '-c:a', 'pcm_s16le',  # PCM 16-bit signed little-endian
                output_path
            ],
            capture_output=True,
            text=True,
            timeout=60
        )
        
        if result.returncode != 0:
            logger.error(f"ffmpeg conversion failed: {result.stderr}")
            raise AudioLoadError(f"Failed to convert audio: {result.stderr[:200]}")
        
        logger.info(f"Audio converted to WAV: {output_path}")
        return output_path
        
    except subprocess.TimeoutExpired:
        raise AudioLoadError("Audio conversion timed out")
    except FileNotFoundError:
        raise AudioLoadError("ffmpeg not found. Please install ffmpeg.")
    except Exception as e:
        raise AudioLoadError(f"Audio conversion failed: {e}")


def preprocess_audio(file_path: str, content_type: Optional[str] = None) -> str:
    """
    Preprocessing pipeline for audio files.
    
    This validates the file, converts to WAV if needed, and returns
    the path for the HuggingFace pipeline to process.
    
    Args:
        file_path: Path to uploaded audio file
        content_type: Optional MIME type from upload
        
    Returns:
        Path to the processed audio file (WAV format)
        
    Raises:
        AudioValidationError: If validation fails
        AudioLoadError: If conversion fails
    """
    # Validate file metadata
    validate_audio_file(file_path, content_type)
    
    # Convert to WAV if needed (for soundfile compatibility)
    wav_path = convert_to_wav(file_path)
    
    logger.info(f"Audio file ready for inference: {wav_path}")
    return wav_path


def cleanup_temp_wav(original_path: str, wav_path: str) -> None:
    """
    Clean up temporary WAV file if it was created during conversion.
    
    Args:
        original_path: Original input file path
        wav_path: WAV file path (may be same as original)
    """
    if wav_path != original_path and os.path.exists(wav_path):
        try:
            os.unlink(wav_path)
            logger.info(f"Cleaned up temporary WAV file: {wav_path}")
        except Exception as e:
            logger.warning(f"Could not delete temporary WAV: {e}")


def get_audio_duration(file_path: str) -> float:
    """
    Get the duration of an audio file in seconds.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Duration in seconds, or 0 if cannot be determined
    """
    try:
        result = subprocess.run(
            [
                'ffprobe', '-v', 'error', '-show_entries',
                'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1',
                file_path
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except Exception as e:
        logger.warning(f"Could not get audio duration: {e}")
    
    return 0.0
