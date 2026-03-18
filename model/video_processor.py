"""
video_processor.py - Video Frame Extraction

Extracts key frames from uploaded videos for processing.
Skips near-duplicate frames to avoid redundant inference.
"""

import cv2
import numpy as np
from PIL import Image


class VideoProcessor:
    """
    Extract key frames from a video file or byte stream.
    
    Usage:
        processor = VideoProcessor(target_fps=2)
        frames = processor.extract_frames('dashcam.mp4')
        # frames = list of (frame_number, PIL.Image) tuples
    """
    
    def __init__(self, target_fps=2, similarity_threshold=0.95, max_frames=100):
        """
        Args:
            target_fps: Frames per second to sample (default 2)
            similarity_threshold: Skip frames more similar than this (0-1)
            max_frames: Maximum number of frames to return
        """
        self.target_fps = target_fps
        self.similarity_threshold = similarity_threshold
        self.max_frames = max_frames
    
    def extract_frames(self, video_path):
        """
        Extract key frames from a video file.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of dicts with:
                'frame_number': int
                'timestamp': float (seconds)
                'image': PIL.Image
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / video_fps if video_fps > 0 else 0
        
        # Calculate frame interval for target FPS
        frame_interval = max(1, int(video_fps / self.target_fps))
        
        frames = []
        prev_frame = None
        frame_idx = 0
        
        while cap.isOpened() and len(frames) < self.max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Only process frames at target interval
            if frame_idx % frame_interval != 0:
                frame_idx += 1
                continue
            
            # Skip near-duplicate frames
            if prev_frame is not None and self._is_similar(prev_frame, frame):
                frame_idx += 1
                continue
            
            # Convert BGR (OpenCV) to RGB (PIL)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            timestamp = frame_idx / video_fps if video_fps > 0 else 0
            
            frames.append({
                'frame_number': frame_idx,
                'timestamp': round(timestamp, 2),
                'image': pil_image
            })
            
            prev_frame = frame.copy()
            frame_idx += 1
        
        cap.release()
        
        return {
            'frames': frames,
            'video_fps': video_fps,
            'total_frames': total_frames,
            'duration': round(duration, 2),
            'extracted_count': len(frames)
        }
    
    def extract_frames_from_bytes(self, video_bytes, temp_path='_temp_video.mp4'):
        """
        Extract frames from video bytes (e.g., from an upload).
        
        Args:
            video_bytes: Video file content as bytes
            temp_path: Temporary file path for writing
            
        Returns:
            Same as extract_frames()
        """
        import tempfile
        import os
        
        # Write bytes to temp file
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as f:
            f.write(video_bytes)
            temp_path = f.name
        
        try:
            result = self.extract_frames(temp_path)
        finally:
            os.unlink(temp_path)
        
        return result
    
    def _is_similar(self, frame1, frame2):
        """
        Check if two frames are near-duplicates using histogram comparison.
        
        Uses normalized histogram correlation — fast and effective
        for detecting static/slow-moving scenes.
        """
        # Resize for faster comparison
        small1 = cv2.resize(frame1, (64, 64))
        small2 = cv2.resize(frame2, (64, 64))
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(small1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(small2, cv2.COLOR_BGR2GRAY)
        
        # Compare histograms
        hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
        
        cv2.normalize(hist1, hist1)
        cv2.normalize(hist2, hist2)
        
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        return similarity > self.similarity_threshold


if __name__ == "__main__":
    print("VideoProcessor - Frame Extraction")
    print("=" * 50)
    print("Usage: processor = VideoProcessor(target_fps=2)")
    print("       result = processor.extract_frames('video.mp4')")
