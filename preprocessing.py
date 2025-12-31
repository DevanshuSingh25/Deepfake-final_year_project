import torch
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import numpy as np
import face_recognition
from typing import List, Generator
import os


# Image preprocessing parameters
IM_SIZE = 112
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Transform pipeline for video frames
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


class ValidationDataset(Dataset):
    """
    Dataset for processing a single video file for validation/prediction.
    Extracts frames, detects faces, and applies transformations.
    """
    
    def __init__(self, video_path: str, sequence_length: int = 60, transform=None):
        self.video_path = video_path
        self.transform = transform if transform else train_transforms
        self.sequence_length = sequence_length
    
    def __len__(self):
        return 1  # Single video
    
    def __getitem__(self, idx):
        frames = []
        
        # Extract frames from video
        for i, frame in enumerate(self.frame_extract(self.video_path)):
            # Detect face in frame
            faces = face_recognition.face_locations(frame)
            try:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except (IndexError, ValueError):
                # No face detected, use full frame
                pass
            
            frames.append(self.transform(frame))
            
            if len(frames) == self.sequence_length:
                break
        
        # If not enough frames, repeat the last frame
        if len(frames) < self.sequence_length:
            last_frame = frames[-1] if frames else torch.zeros(3, IM_SIZE, IM_SIZE)
            while len(frames) < self.sequence_length:
                frames.append(last_frame)
        
        frames = torch.stack(frames)
        frames = frames[:self.sequence_length]
        return frames.unsqueeze(0)
    
    def frame_extract(self, path: str) -> Generator[np.ndarray, None, None]:
        """Extract frames from video file"""
        vidObj = cv2.VideoCapture(path)
        success = True
        while success:
            success, image = vidObj.read()
            if success:
                yield image
        vidObj.release()


def preprocess_video(
    video_path: str,
    sequence_length: int,
    save_preprocessed: bool = False,
    output_dir: str = "temp_frames"
) -> tuple:
    """
    Preprocess video for model prediction.
    
    Args:
        video_path: Path to the video file
        sequence_length: Number of frames to extract
        save_preprocessed: Whether to save preprocessed images
        output_dir: Directory to save preprocessed images
    
    Returns:
        Tuple of (preprocessed_tensor, preprocessed_images_list, face_cropped_images_list)
    """
    preprocessed_images = []
    face_cropped_images = []
    
    # Create output directory if saving images
    if save_preprocessed and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read video
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frames.append(frame)
        else:
            break
    cap.release()
    
    print(f"Total frames extracted: {len(frames)}")
    
    # Process frames
    padding = 40
    faces_found = 0
    processed_frames = []
    
    for i in range(min(sequence_length, len(frames))):
        frame = frames[i]
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Save preprocessed image if requested
        if save_preprocessed:
            preprocessed_path = os.path.join(output_dir, f"frame_{i+1}.png")
            cv2.imwrite(preprocessed_path, cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR))
            preprocessed_images.append(preprocessed_path)
        
        # Face detection and cropping
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if len(face_locations) > 0:
            top, right, bottom, left = face_locations[0]
            # Apply padding
            top = max(0, top - padding)
            bottom = min(rgb_frame.shape[0], bottom + padding)
            left = max(0, left - padding)
            right = min(rgb_frame.shape[1], right + padding)
            
            frame_face = rgb_frame[top:bottom, left:right]
            
            # Save cropped face if requested
            if save_preprocessed:
                face_path = os.path.join(output_dir, f"face_{i+1}.png")
                cv2.imwrite(face_path, cv2.cvtColor(frame_face, cv2.COLOR_RGB2BGR))
                face_cropped_images.append(face_path)
            
            faces_found += 1
            processed_frame = frame_face
        else:
            # No face detected, use full frame
            processed_frame = rgb_frame
        
        # Apply transforms
        transformed_frame = train_transforms(processed_frame)
        processed_frames.append(transformed_frame)
    
    print(f"Faces detected: {faces_found}/{sequence_length}")
    
    # Handle case where not enough frames
    if len(processed_frames) < sequence_length:
        last_frame = processed_frames[-1] if processed_frames else torch.zeros(3, IM_SIZE, IM_SIZE)
        while len(processed_frames) < sequence_length:
            processed_frames.append(last_frame)
    
    # Stack frames into tensor
    frames_tensor = torch.stack(processed_frames[:sequence_length])
    frames_tensor = frames_tensor.unsqueeze(0)  # Add batch dimension
    
    return frames_tensor, preprocessed_images, face_cropped_images


def predict(model, img_tensor, device: str = "cpu"):
    """
    Make prediction on preprocessed video tensor.
    
    Args:
        model: Loaded PyTorch model
        img_tensor: Preprocessed video tensor
        device: 'cpu' or 'cuda'
    
    Returns:
        Tuple of (prediction, confidence)
        prediction: 0 for FAKE, 1 for REAL
        confidence: Confidence percentage (0-100)
    """
    sm = torch.nn.Softmax(dim=1)
    
    # Move tensor to device
    if device == "cuda":
        img_tensor = img_tensor.cuda()
    else:
        img_tensor = img_tensor.cpu()
    
    # Forward pass
    with torch.no_grad():
        fmap, logits = model(img_tensor)
        logits = sm(logits)
        _, prediction = torch.max(logits, 1)
        confidence = logits[0, int(prediction.item())].item() * 100
    
    return int(prediction.item()), confidence
