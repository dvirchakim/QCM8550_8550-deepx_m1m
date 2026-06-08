import cv2
import numpy as np
import time
import threading

class Source:
    """Abstract base class for all input sources."""
    def __init__(self):
        self.active = True

    def get_frame(self):
        """Return the next frame from the source. Must be implemented by subclasses."""
        raise NotImplementedError

    def start(self):
        self.active = True

    def stop(self):
        self.active = False
    
    def restart(self):
        pass

class LiveStreamSource(Source):
    """Handles live camera feeds (RTSP or local camera)."""
    def __init__(self, source_id):
        super().__init__()
        self.source_id = source_id
        self.cap = None
        self._connect()

    def _connect(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.source_id)
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.source_id}")

    def get_frame(self):
        if not self.active or not self.cap or not self.cap.isOpened():
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            # Attempt reconnection logic could go here
            print(f"Frame read failed for source {self.source_id}")
            self._connect()
            return None
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
        
    def restart(self):
        self._connect()

class FileSource(Source):
    """Handles looping video files."""
    def __init__(self, file_path):
        super().__init__()
        self.file_path = file_path
        self.cap = None
        self._connect()

    def _connect(self):
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(self.file_path)

    def get_frame(self):
        if not self.active or not self.cap or not self.cap.isOpened():
            return None
            
        ret, frame = self.cap.read()
        if not ret:
            # Loop video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                return None
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

class GenAISource(Source):
    """Handles the Stable Diffusion output stream."""
    def __init__(self):
        super().__init__()
        self.current_image = None
        self.lock = threading.Lock()
        # Initialize with a placeholder or black image
        self.current_image = np.zeros((512, 512, 3), dtype=np.uint8)

    def update_image(self, image):
        """Thread-safe update of the generated image."""
        with self.lock:
            self.current_image = image

    def get_frame(self):
        with self.lock:
            img = self.current_image.copy() if self.current_image is not None else None
        return img
