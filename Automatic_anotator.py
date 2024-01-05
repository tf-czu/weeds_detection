import os
import cv2
import numpy as np
import json

class AnnotateWhiteObjects:
    def __init__(self, input_dir, output_json, object_name):
        self.input_dir = input_dir
        self.output_json = output_json
        self.object_name = object_name
        self.data = {}
        self.annotate_objects()

    def annotate_objects(self):
        if not os.path.exists(self.input_dir):
            print(f"Input directory '{self.input_dir}' does not exist.")
            return

        if not os.path.exists(os.path.dirname(self.output_json)):
            os.makedirs(os.path.dirname(self.output_json))

        file_list = os.listdir(self.input_dir)
        for filename in file_list:
            if filename.endswith(".png"):
                image_path = os.path.join(self.input_dir, filename)
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

                # Otsu's thresholding
                _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # Morphological operations for noise reduction
                kernel = np.ones((5, 5), np.uint8)
                binary_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

                # Find contours
                contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Annotate white objects in JSON format
                object_annotations = []
                for i, contour in enumerate(contours):
                    x, y, w, h = cv2.boundingRect(contour)
                    object_annotations.append({
                        "object_id": i,
                        "object_name": self.object_name,
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h)
                    })

                # Store annotations in data dictionary
                self.data[filename] = object_annotations

        # Write annotations to JSON file
        with open(self.output_json, 'w') as json_file:
            json.dump(self.data, json_file, indent=4)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Annotate white objects in binary images.')
    parser.add_argument('input_dir', help='Path to directory containing grayscale images.')
    parser.add_argument('output_json', help='Output JSON file to store annotations.')
    parser.add_argument('--object_name', help='Name for annotated objects', default='white_object')

    args = parser.parse_args()
    annotator = AnnotateWhiteObjects(args.input_dir, args.output_json, args.object_name)
