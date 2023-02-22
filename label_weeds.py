"""
  Tool for weeds labeling in imagesTool for weeds labeling in images
"""
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from image_processing import transform_nir, get_ndvi_im, get_excess_green, get_com_im, noise_reduction


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

    def get_bbox(self, color_path):
        nir_path = self.data[color_path]["nir"]
        im_color = cv2.imread(color_path)

        im_nir = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
        nir = transform_nir(im_nir)
        ndvi = get_ndvi_im(im_color, nir)
        exg = get_excess_green(im_color)
        com_im = get_com_im(exg, ndvi)

        com_im = cv2.medianBlur(com_im, 5)
        ret, binary_im = cv2.threshold(com_im, 75, 255, cv2.THRESH_BINARY)
        binary_im = noise_reduction(binary_im)
        contours, hierarchy = cv2.findContours(binary_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bbox_list = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                bbox_list.append((x, y, w, h, None))

        return bbox_list

    def run(self):
        print(self.data)
        item_list = [d for d in self.data]
        ii = 0
        jj = 0
        while True:
            if ii < 0:
                ii = 0
            elif ii >= len(item_list):
                ii = len(item_list)-1

            im_name = item_list[ii]
            im_dic = self.data[im_name]
            bbox_list = im_dic.get("bbox_list")
            im_color = cv2.imread(im_name)
            if bbox_list is None:
                bbox_list = self.get_bbox(im_name)

            if jj < 0:
                jj = 0
            elif jj >= len(bbox_list):
                jj = len(bbox_list)-1

            for x, y, w, h, kind in bbox_list:
                if kind is None:
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)
                cv2.rectangle(im_color, (x, y), (x+w, y+h), color, 1)

            # current bbox
            x, y, w, h, kind = bbox_list[jj]

            # prepare sub_img
            n, m, __ = im_color.shape
            print(n, m)
            x0 = max(0, x - 100)
            y0 = max(0, y - 100)
            x1 = min(m, x + w + 100)
            y1 = min(n, y + h + 100)
            sub_im = im_color[y0:y1, x0:x1, :]
            cv2.rectangle(sub_im, (x - x0, y - y0), (x + w - x0, y + h - y0), (255, 0, 0), 1)

            cv2.namedWindow("win", cv2.WINDOW_NORMAL)
            cv2.namedWindow("sub_win", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("win", m // 2, n // 2)
            cv2.imshow("win", im_color)
            cv2.imshow("sub_win", sub_im)

            cv2.waitKey(0)
            cv2.destroyAllWindows()

            assert False


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='__doc__')
    parser.add_argument('path', help='Path to image directory.')

    args = parser.parse_args()
    label = LabelWeeds(path = args.path)
    label.run()
