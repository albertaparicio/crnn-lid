import numpy as np
from imageio import imread

from .csv_loader import CSVLoader


class ImageLoader(CSVLoader):
    def process_file(self, file_path):

        # image = imread(file_path, pilmode=self.config["color_mode"])
        image = imread(file_path)

        # Image shape should be (cols, rows, channels)
        if len(image.shape) == 2:
            image = np.expand_dims(image, -1)

        assert len(image.shape) == 3

        return np.divide(image, 255.0)  # Normalize images to 0-1.0
