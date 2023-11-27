"""
    Sort RGB and NIR images.
"""

import os
import numpy as np
import shutil

NIR_SIZE = 3145902
RGB_SIZE = 9437364
MIN_TIME_DIFF = 50  # ms


def stat_test(data1, data2):
    data2 = np.asarray(data2)
    for num in data1:
        diff_arr = abs(data2 - num)
        # print(diff_arr[diff_arr < 50])
        assert len(diff_arr[diff_arr < 50]) <= 1


def copy_twins(img_dir, twins, num_offset=0):
    assert not os.path.exists(os.path.join(img_dir, "twins"))  # "Twins directory already exists!"
    os.makedirs(os.path.join(img_dir, "twins"))
    for ii, (rgb_path, nir_path) in enumerate(twins):
        shutil.copy(rgb_path, os.path.join(img_dir, "twins", f"im_{num_offset + ii:06d}_rgb"))
        shutil.copy(nir_path, os.path.join(img_dir, "twins", f"im_{num_offset + ii:06d}_nir"))


def name_to_timestamp(img_list):
    timestamps_list = []
    for im_path in img_list:
        im_name = os.path.basename(im_path)
        time_id = int(im_name[:-5].split("_")[1], 16)
        timestamps_list.append(time_id)

    return timestamps_list


def sort_by_time(rgb_list, nir_list):
    img_twins = []
    rgb_timestamps = name_to_timestamp(rgb_list)
    nir_timestamps = name_to_timestamp(nir_list)
    stat_test(rgb_timestamps, nir_timestamps)

    nir_timestamps_arr = np.asarray(nir_timestamps)
    for rgb_path, rgb_time_id in zip (rgb_list, rgb_timestamps):
        time_diff = abs(nir_timestamps_arr-rgb_time_id)
        if min(time_diff) < MIN_TIME_DIFF:
            nir_path = nir_list[np.argmin(time_diff)]
            img_twins.append([rgb_path, nir_path])

    return img_twins


def sort_images(img_dir):
    rgb_list = []
    nir_list = []
    dir_content = os.listdir(img_dir)
    print(dir_content)
    for item in dir_content:
        item_path = os.path.join(img_dir, item)  # directory with rgb or nir images
        if os.path.isdir(item_path) and ("NIR" in item or "RGB" in item):
            img_list = os.listdir(item_path)
            for img_name in img_list:
                # print(img_name, len(img_name))
                if len(img_name) == 30 and img_name.endswith(".tiff"):
                    img_name_path = os.path.join(item_path, img_name)
                    img_size = os.path.getsize(img_name_path)
                    assert img_size in [NIR_SIZE, RGB_SIZE], img_size
                    if img_size == RGB_SIZE:
                        rgb_list.append(img_name_path)
                    elif img_size == NIR_SIZE:
                        nir_list.append(img_name_path)

    img_twins = sort_by_time(rgb_list, nir_list)
    copy_twins(img_dir, img_twins)


def all_directories(main_dir):
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('img_dir', help='Images directory.')
    parser.add_argument('--all', '-a', help='Apply on all subdirectories.', action="store_true")
    args = parser.parse_args()

    if args.all:
        all_directories(args.img_dir)
    else:
        sort_images(args.img_dir)
