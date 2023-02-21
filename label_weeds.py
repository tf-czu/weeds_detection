"""
  Tool for weeds labeling in imagesTool for weeds labeling in images
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

g_threshold = 160


def transform_nir(nir):
    n, m = nir.shape
    nir_resized = cv2.resize(nir, (m - 6, n - 6))
    ret_nir = np.zeros((n, m), np.uint8)
    ret_nir[3:-3, 3:-3] = nir_resized

    return ret_nir


def get_ndvi_im(im_color, nir):
    b, g, r = cv2.split(im_color)
    r = r.astype(float)
    nir = nir.astype(float)
    ndvi = (nir - r) / (nir + r)
    ndvi_im = (ndvi + 1)/2 * 255

    return ndvi_im.astype(np.uint8)


def show_im(im):
    cv2.namedWindow("win", cv2.WINDOW_NORMAL)
    n, m = im.shape
    cv2.resizeWindow("win", m // 2, n // 2)
    cv2.imshow("win", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_hist(im):
    plt.hist(im.ravel(), 256, [0, 256])
    plt.show()


def test_ndvi(path):
    assert os.path.isfile(path)
    assert path.endswith("C.tif")
    color_im = cv2.imread(path)

    nir_path = path[:-5]+"N.tif"
    assert os.path.exists(nir_path)
    nir_im = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)

    ndvi = get_ndvi_im(color_im, nir_im)
    show_im(ndvi)
    show_hist(ndvi)
    ret, binary_im = cv2.threshold(ndvi, g_threshold, 255, cv2.THRESH_BINARY)
    show_im(binary_im)


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
    parser.add_argument('--ndvi', action="store_true")

    args = parser.parse_args()
    if args.ndvi:
        test_ndvi(args.path)
    else:
        label = LabelWeeds(path = args.path)
        label.run()
