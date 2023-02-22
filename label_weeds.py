"""
  Tool for weeds labeling in imagesTool for weeds labeling in images
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from image_processing import transform_nir, get_ndvi_im


class LabelWeeds:
    def __init__(self, path):
        self.path = path
        self.data = {}
        self.add_images()

    def add_images(self):
        assert os.path.isdir(self.path)
        file_list = os.listdir(self.path)
        for item in sorted(file_list):
            if item.endswith("C.tif"):
                color_path = os.path.join(self.path, item)
                nir_path = os.path.join(self.path, item[:-5]+"N.tif")
                if os.path.exists(nir_path):
                    if not color_path in self.data:
                        self.data[color_path] = {"nir": nir_path}

    def get_contours(self, color_path):
        nir_path = self.data[color_path]["nir"]
        im_color = cv2.imread(color_path)

        im_nir = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
        nir = transform_nir(im_nir)
        ndvi = get_ndvi_im(im_color, nir)

        cv2.namedWindow("win", cv2.WINDOW_NORMAL)
        n, m = nir.shape
        cv2.resizeWindow("win", m // 2, n // 2)
        cv2.imshow("win", ndvi)
        cv2.waitKey(0)

    def run(self):
        print(self.data)
        item_list = [d for d in self.data]
        ii = 0
        while True:
            self.get_contours(item_list[ii])
            assert False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='__doc__')
    parser.add_argument('path', help='Path to image directory.')

    args = parser.parse_args()
    label = LabelWeeds(path = args.path)
    label.run()
