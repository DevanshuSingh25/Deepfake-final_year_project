import torch
import torch.nn as nn
from torchvision import models
import glob
import os
from typing import Optional

class Model(nn.Module):
    """
    Video deepfake detection model using ResNeXt50 + LSTM architecture.
    Ported from reference code for production use.
    """
    
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        # Load pretrained ResNeXt50
        model = models.resnext50_32x4d(pretrained=True)
        # Remove the last two layers (avgpool and fc)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))


def get_accurate_model(sequence_length: int, models_dir: str = "models") -> Optional[str]:
    """
    Select the best model based on sequence length (frame count).
    
    Args:
        sequence_length: Number of frames to sample from video (10, 20, 40, 60, 80, 100)
        models_dir: Directory containing the model files
    
    Returns:
        Full path to the selected model file, or None if no model found
    """
    model_name = []
    sequence_model = []
    final_model = ""
    
    # Get all .pt model files
    list_models = glob.glob(os.path.join(models_dir, "*.pt"))
    
    if not list_models:
        print(f"No models found in {models_dir}")
        return None
    
    for model_path in list_models:
        model_name.append(os.path.basename(model_path))
    
    # Find models matching the sequence length
    for model_filename in model_name:
        try:
            # Model naming pattern: model_{accuracy}_acc_{frames}_frames_*.pt
            parts = model_filename.split("_")
            seq = parts[3]  # frames count is at index 3
            if int(seq) == sequence_length:
                sequence_model.append(model_filename)
        except (IndexError, ValueError):
            continue
    
    # Select model with highest accuracy if multiple found
    if len(sequence_model) > 1:
        accuracy = []
        for filename in sequence_model:
            acc = filename.split("_")[1]  # accuracy is at index 1
            accuracy.append(acc)
        max_index = accuracy.index(max(accuracy))
        final_model = os.path.join(models_dir, sequence_model[max_index])
    elif len(sequence_model) == 1:
        final_model = os.path.join(models_dir, sequence_model[0])
    else:
        print(f"No model found for sequence length {sequence_length}")
        return None
    
    return final_model


# Global model cache to avoid reloading
_model_cache = {}


def load_model(sequence_length: int, device: str = "cpu") -> Optional[Model]:
    """
    Load the model for the specified sequence length.
    Uses caching to avoid reloading the same model.
    
    Args:
        sequence_length: Number of frames (10, 20, 40, 60, 80, 100)
        device: 'cpu' or 'cuda'
    
    Returns:
        Loaded model ready for inference, or None if loading fails
    """
    cache_key = f"{sequence_length}_{device}"
    
    # Check cache first
    if cache_key in _model_cache:
        print(f"Using cached model for {sequence_length} frames")
        return _model_cache[cache_key]
    
    # Get the model path
    model_path = get_accurate_model(sequence_length)
    if not model_path:
        return None
    
    print(f"Loading model: {model_path}")
    
    try:
        # Initialize model
        model = Model(num_classes=2)
        
        # Load state dict
        if device == "cuda" and torch.cuda.is_available():
            model = model.cuda()
            model.load_state_dict(torch.load(model_path))
        else:
            model = model.cpu()
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        
        model.eval()
        
        # Cache the model
        _model_cache[cache_key] = model
        print(f"Model loaded successfully for {sequence_length} frames")
        
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def get_device() -> str:
    """Detect available device (GPU or CPU)"""
    return "cuda" if torch.cuda.is_available() else "cpu"
