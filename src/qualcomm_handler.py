import numpy as np
import cv2

class QualcommHandler:
    def __init__(self):
        self.model_loaded = False

    def load_model(self):
        print("Loading Stable Diffusion on Qualcomm NPU...")
        self.model_loaded = True

    def generate_image(self, prompt):
        """Generates a dummy noise image representing SD output."""
        print(f"Generating image for prompt: {prompt}")
        # Create a random noise image
        img = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
        # Add text
        cv2.putText(img, "QCS8550 GenAI", (20, 480), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img
