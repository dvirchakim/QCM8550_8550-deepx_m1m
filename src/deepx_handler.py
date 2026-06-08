import cv2
import time

class DeepXHandler:
    def __init__(self):
        self.model = None
        self.active = True

    def load_model(self, model_path):
        print(f"Loading DeepX model from {model_path}...")
        self.model = "DummyModel"
        self.active = True

    def restart(self):
        print("Restaring DeepX Handler...")
        self.active = False
        time.sleep(0.5) # Simulate restart delay
        self.active = True
        print("DeepX Handler restarted.")

    def process_frame(self, frame):
        """Draws a dummy bounding box on the frame."""
        if not self.active or frame is None:
            return frame # Return original if failed
        
        try:
            # Simulate random failure (rare)
            # if random.random() < 0.001: raise Exception("DeepX Driver Hang")

            processed = frame.copy()
            # Draw a red rectangle
            h, w, _ = processed.shape
            cv2.rectangle(processed, (50, 50), (w-50, h-50), (255, 0, 0), 2)
            cv2.putText(processed, "DeepX Inference", (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            return processed
        except Exception as e:
            print(f"DeepX Error: {e}")
            self.restart()
            return frame
