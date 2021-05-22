from pathlib import Path
import cv2
import numpy as np


class Colorizer:
    _input_image_width = 224
    _input_image_height = 224

    def __init__(self, path_to_proto_file: Path, path_to_weights_file: Path, path_to_bin_centers_file: Path):
        self.net = cv2.dnn.readNetFromCaffe(str(path_to_proto_file), str(path_to_weights_file))

        points_in_hull = np.load(path_to_bin_centers_file)
        points_in_hull = points_in_hull.transpose().reshape(2, 313, 1, 1)

        self.net.getLayer(self.net.getLayerId("class8_ab")).blobs = [points_in_hull.astype(np.float32)]
        self.net.getLayer(self.net.getLayerId("conv8_313_rh")).blobs = [np.full([1, 313], 2.606, np.float32)]

    def get_colorized(self, image):
        original_height, original_width = image.shape[:2]

        image_in_rgb = (image[:, :, [2, 1, 0]] * 1.0 / 255).astype(np.float32)
        image_in_lab = cv2.cvtColor(image_in_rgb, cv2.COLOR_RGB2Lab)
        lightness_channel = image_in_lab[:, :, 0]

        lightness_channel_resized = cv2.resize(lightness_channel, (self._input_image_width, self._input_image_height))
        lightness_channel_resized -= 50

        self.net.setInput(cv2.dnn.blobFromImage(lightness_channel_resized))
        predicted_channels = self.net.forward()[0, :, :, :].transpose((1, 2, 0))

        predicted_channels_with_original_size = cv2.resize(predicted_channels, (original_width, original_height))
        predicted_image_in_lab = np.concatenate(
            (lightness_channel[:, :, np.newaxis], predicted_channels_with_original_size), axis=2
        )
        return np.clip(cv2.cvtColor(predicted_image_in_lab, cv2.COLOR_Lab2BGR), 0, 1)
