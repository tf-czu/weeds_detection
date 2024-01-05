from gettext import npgettext
import json
import numpy as np
import os
import cv2
from image_processing import get_ndvi_im, noise_reduction
from typing import List, Dict

class AutoLabelWeeds:
    def __init__(self, path: str, out_json: str, naming_convention: str):
        """
        Initializes AutoLabelWeeds class.

        Args:
        - path: Path to the directory containing NDVI images.
        - out_json: Path to the output JSON file for annotations.
        - naming_convention: Naming convention for annotated objects.
        """
        self.path = path
        assert out_json.endswith(".json"), "Output file must be a .json file."
        self.out_json = out_json
        self.naming_convention = naming_convention

    def get_ndvi_images(self) -> List[str]:
        """
        Retrieves all NDVI images in the specified directory.

        Returns:
        - ndvi_images: List of paths to NDVI images.
        """
        ndvi_images = [os.path.join(self.path, img) for img in os.listdir(self.path) if img.endswith(".png")]
        return ndvi_images

    def get_visible_elements(self, ndvi_image: npgettext.ndarray) -> List[Dict[str, int]]:
        """
        Identifies visible elements in the NDVI image.

        Args:
        - ndvi_image: NDVI image in a numpy array format.

        Returns:
        - visible_elements: List of dictionaries containing annotations for visible elements.
        """
        _, binary_im = cv2.threshold(ndvi_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_im = noise_reduction(binary_im)
        contours, _ = cv2.findContours(binary_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        visible_elements = []

        for index, cnt in enumerate(contours):
            if cv2.contourArea(cnt) > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                kind = f"{self.naming_convention}_{index}"  
                visible_elements.append({"x": x, "y": y, "w": w, "h": h, "kind": kind})

        return visible_elements

    def annotate_images(self):
        """
        Annotates NDVI images with visible elements and saves annotations to a JSON file.
        """
        ndvi_images = self.get_ndvi_images()
        annotations = {}

        for ndvi_path in ndvi_images:
            ndvi_image = cv2.imread(ndvi_path, cv2.IMREAD_GRAYSCALE)
            visible_elements = self.get_visible_elements(ndvi_image)
            annotations[ndvi_path] = {"annotations": visible_elements}

        with open(self.out_json, "w") as outfile:
            json.dump(annotations, outfile, indent=4)

def perform_ndvi(image_dir: str, output_dir: str):
    """
    Perform NDVI operation on the images in the specified directory.

    Args:
    - image_dir: Directory containing the images.
    - output_dir: Output directory to save NDVI images.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_list = os.listdir(image_dir)
    for item in sorted(file_list):
        if item.endswith("C.tif"):  # Assuming color images end with "C.tif"
            color_path = os.path.join(image_dir, item)
            nir_path = os.path.join(image_dir, item[:-5] + "N.tif")  # Corresponding NIR image
            if os.path.exists(nir_path):
                color_im = cv2.imread(color_path)
                nir_im = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)

                # Perform NDVI
                ndvi_im = get_ndvi_im(color_im, nir_im)

                # Save NDVI image
                output_path = os.path.join(output_dir, f"NDVI_{item[:-5]}.png")
                cv2.imwrite(output_path, ndvi_im)
