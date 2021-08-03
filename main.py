from pathlib import Path
from typing import Dict
import cv2
import streamlit as st
import numpy as np

from colorizer import Colorizer


def get_image(image_options_to_paths: Dict[str, Path]):
    mode = st.radio(
        "Choose whether you will upload image yourself or see an example ('Upload image' by default):",
        ("Upload image", "See example"),
    )

    if mode == "See example":
        image_option = st.selectbox("Choose what you want to colorize:", tuple(image_options_to_paths))
        path = str(image_options_to_paths[image_option])
        return cv2.imread(path)

    uploader_label = (
        "Select the file you want to colorize. You can select a color image and then it will be shown "
        "how the colorized image would look if the original was black and white:"
    )

    uploaded_image = st.file_uploader(uploader_label, type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        image_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        return cv2.imdecode(image_bytes, 1)


def main():
    path_to_examples = "./examples"
    image_options_to_paths = {
        "Town": Path(f"{path_to_examples}/town.jpg"),
        "Human": Path(f"{path_to_examples}/human.jpg"),
        "Dog": Path(f"{path_to_examples}/dog.jpg"),
    }

    path_to_net_settings = "./network_settings"
    colorizer = Colorizer(
        Path(f"{path_to_net_settings}/colorization_deploy_v2.prototxt"),
        Path(f"{path_to_net_settings}/colorization_release_v2.caffemodel"),
        Path(f"{path_to_net_settings}/pts_in_hull.npy"),
    )

    st.title("Image colorization")

    image = get_image(image_options_to_paths)
    if st.button("Colorize"):
        st.header("Result")
        st.image(
            [image, colorizer.get_colorized(image)],
            caption=["An original image", "An image after colorization"],
            channels="BGR",
        )


if __name__ == "__main__":
    main()
