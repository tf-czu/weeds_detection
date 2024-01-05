import os
import cv2
from image_processing import get_ndvi_im

def perform_ndvi(image_dir, output_dir):
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

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Perform NDVI on image pairs.')
    parser.add_argument('input_dir', help='Directory containing image pairs.')
    parser.add_argument('output_dir', help='Output directory to save NDVI images.')

    args = parser.parse_args()
    perform_ndvi(args.input_dir, args.output_dir)
