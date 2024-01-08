import os

import cv2
import numpy as np

from image_processing import show_hist


def transform_nir(nir):
    n, m = nir.shape
    nir_resized = cv2.resize(nir, (m - 6, n - 6))
    ret_nir = np.zeros((n, m), np.uint8)
    ret_nir[3:-3, 3:-3] = nir_resized
    return ret_nir

def process_images(image_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    file_list = os.listdir(image_dir)
    for item in file_list:
        if item.endswith("C.tif"):  # Assuming color images end with "C.tif"
            color_path = os.path.join(image_dir, item)
            nir_path = os.path.join(image_dir, item[:-5] + "N.tif")  # Corresponding NIR image
            if os.path.exists(nir_path):
                color_im = cv2.imread(color_path)
                b, g, r = cv2.split(color_im)
                b = b.astype(float)
                g = g.astype(float)
                r = r.astype(float)
                nir_im = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)

                # Resize NIR image to fit over RGB
                nir_im_resized = transform_nir(nir_im)
                n = nir_im_resized.astype(float)

                # Processing formula: (2xNIR + Green - Red - Blue + 510)/4
                processed_im = (1 * n + g - r - b +510) /4
                # processed_im = (((2 * n + g - r - b) / (2 * n + g + r + b + 510) - 0.05) * 1.8 + 1) * 128
                show_hist(processed_im)
                # processed_im = cv2.convertScaleAbs(processed_im)
                # processed_im = (processed_im - processed_im.min())/(processed_im.max() - processed_im.min()) * 255
                processed_im = (processed_im - 90) / (170 - 90) * 255

                show_hist(processed_im)
                processed_im = np.clip(processed_im, 0, 255).astype(np.uint8)  # Clip values to 0-255 range

                # Save processed image as grayscale
                output_path = os.path.join(output_dir, f"processed_{item[:-5]}.png")
                cv2.imwrite(output_path, processed_im)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Process NIR and color image pairs.')
    parser.add_argument('input_dir', help='Path to input directory containing NIR and color image pairs.')
    parser.add_argument('output_dir', help='Path to output directory to save processed images.')

    args = parser.parse_args()
    input_directory = args.input_dir
    output_directory = args.output_dir

    process_images(input_directory, output_directory)
