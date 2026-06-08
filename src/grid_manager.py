import cv2
import numpy as np
import config

class GridManager:
    def __init__(self, rows=3, cols=3):
        self.rows = rows
        self.cols = cols
        self.width = config.WINDOW_WIDTH
        self.height = config.WINDOW_HEIGHT
        
        # Create a black canvas
        self.canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        # Calculate tile dimensions
        self.tile_w = self.width // self.cols
        self.tile_h = self.height // self.rows
        
        self.window_name = config.WINDOW_TITLE
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        # On embedded, force fullscreen or specific size
        # cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    def update_tile(self, index, frame):
        if frame is None:
            return
            
        # Calculate grid position
        r = index // self.cols
        c = index % self.cols
        
        # Resize frame to fit the tile
        resized = cv2.resize(frame, (self.tile_w, self.tile_h))
        
        # Place on canvas
        y_start = r * self.tile_h
        y_end = y_start + self.tile_h
        x_start = c * self.tile_w
        x_end = x_start + self.tile_w
        
        self.canvas[y_start:y_end, x_start:x_end] = resized

    def show(self):
        # Convert RGB to BGR for OpenCV display
        bgr_canvas = cv2.cvtColor(self.canvas, cv2.COLOR_RGB2BGR)
        cv2.imshow(self.window_name, bgr_canvas)
        # Need waitKey to process events
        return cv2.waitKey(1) & 0xFF
