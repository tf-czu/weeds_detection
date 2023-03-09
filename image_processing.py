"""
    Basic tools for image processing
"""
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


def noise_reduction(bin_im):
    element = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    bin_im = cv2.morphologyEx(bin_im, cv2.MORPH_OPEN, element, iterations=1)
    return bin_im

def transform_nir(nir):
    n, m = nir.shape
    nir_resized = cv2.resize(nir, (m - 6, n - 6))
    ret_nir = np.zeros((n, m), np.uint8)
    ret_nir[3:-3, 3:-3] = nir_resized
    return ret_nir


def get_norm_colors(im):
    b, g, r = cv2.split(im)
    b = b.astype(float)
    g = g.astype(float)
    r = r.astype(float)
    sum_color = b+g+r
    return b/sum_color, g/sum_color, r/sum_color


def get_excess_green(im_color):
    b, g, r = get_norm_colors(im_color)
    exg = 2*g-r-b
    return ((exg + 2)/4 * 255).astype(np.uint8)


def get_ndvi_im(im_color, nir):
    b, g, r = cv2.split(im_color)
    r = r.astype(float)
    nir = nir.astype(float)
    ndvi = (nir - r) / (nir + r)
    ndvi_im = (ndvi + 1)/2 * 255
    return ndvi_im.astype(np.uint8)


def get_com_im(exg, ndvi):
    com_im = exg.astype(np.uint16) * ndvi.astype(np.uint16) / 255
    return com_im.astype(np.uint8)


def show_im(im):
    cv2.namedWindow("win", cv2.WINDOW_NORMAL)
    n = im.shape[0]
    m = im.shape[1]
    cv2.resizeWindow("win", m // 2, n // 2)
    cv2.imshow("win", im)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def show_hist(im):
    plt.hist(im.ravel(), 256, [0, 256])
    plt.show()


def draw_contours(binary_im, color_im):
    contours, hierarchy = cv2.findContours(binary_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(color_im, contours, -1, (255, 0, 0), thickness=1)
    for cnt in contours:
        if cv2.contourArea(cnt) > 100:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(color_im, (x, y), (x + w, y + h), (0, 0, 255), 1)

    return color_im


def test_ndvi(path):
    assert os.path.isfile(path)
    assert path.endswith("C.tif")
    color_im = cv2.imread(path)

    nir_path = path[:-5]+"N.tif"
    assert os.path.exists(nir_path)
    nir_im = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)

    ndvi = get_ndvi_im(color_im, nir_im)
    show_im(ndvi)
    # ndvi = cv2.GaussianBlur(ndvi, (5, 5), 0)
    # show_im(ndvi)
    show_hist(ndvi)

    ndvi = cv2.equalizeHist(ndvi)
    show_im(ndvi)
    show_hist(ndvi)

    ret, binary_im = cv2.threshold(ndvi, 130, 255, cv2.THRESH_BINARY)
    # binary_im = cv2.adaptiveThreshold(ndvi, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,21,2)
    show_im(binary_im)

    binary_im = noise_reduction(binary_im)
    show_im(binary_im)
    color_im = draw_contours(binary_im, color_im)
    show_im(color_im)


def test_excess_green(path):
    assert os.path.isfile(path)
    assert path.endswith("C.tif")
    color_im = cv2.imread(path)
    exg = get_excess_green(color_im)

    show_im(exg)
    show_hist(exg)
    ret, binary_im = cv2.threshold(exg, 130, 255, cv2.THRESH_BINARY)
    # binary_im = cv2.adaptiveThreshold(exg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
    show_im(binary_im)

    binary_im = noise_reduction(binary_im)
    show_im(binary_im)
    color_im = draw_contours(binary_im, color_im)
    show_im(color_im)


def test_combination(path):
    assert os.path.isfile(path)
    assert path.endswith("C.tif")
    color_im = cv2.imread(path)
    exg = get_excess_green(color_im)
    nir_path = path[:-5] + "N.tif"
    assert os.path.exists(nir_path)
    nir_im = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
    ndvi = get_ndvi_im(color_im, nir_im)

    com_im = get_com_im(exg, ndvi)
    show_im(com_im)
    # com_im = cv2.GaussianBlur(com_im,(5,5),0)
    com_im = cv2.medianBlur(com_im, 5)
    show_im(com_im)
    show_hist(com_im)

    # com_im = cv2.equalizeHist(com_im)
    # show_im(com_im)
    # show_hist(com_im)

    ret, binary_im = cv2.threshold(com_im, 75, 255, cv2.THRESH_BINARY)
    # binary_im = cv2.adaptiveThreshold(com_im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 2)
    show_im(binary_im)

    binary_im = noise_reduction(binary_im)
    show_im(binary_im)
    color_im = draw_contours(binary_im, color_im)
    show_im(color_im)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='__doc__')
    parser.add_argument('path', help='Path to color image.')
    parser.add_argument('--ndvi', action="store_true")
    parser.add_argument('--exg', action="store_true")
    parser.add_argument('--com', action="store_true")

    args = parser.parse_args()
    if args.ndvi:
        test_ndvi(args.path)
    if args.exg:
        test_excess_green(args.path)
    if args.com:
        test_combination(args.path)
