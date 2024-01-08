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


def copy_twins(img_dir, twins, im_id=0):
    if not twins:
        return im_id
    log = open(os.path.join(img_dir, "rename_log.log"), "w")
    assert not os.path.exists(os.path.join(img_dir, "twins")), img_dir  # "Twins directory already exists!"
    os.makedirs(os.path.join(img_dir, "twins"))
    for rgb_path, nir_path in twins:
        im_id += 1
        new_rgb_name = f"im_{im_id:06d}_rgb.tiff"
        shutil.copy(rgb_path, os.path.join(img_dir, "twins", new_rgb_name))
        log.write(f"{rgb_path}, {new_rgb_name}\r\n")

        new_nir_name = f"im_{im_id:06d}_nir.tiff"

    log.close()

    return im_id


def name_to_timestamp(img_list):
    timestamps_list = []
    for im_path in img_list:
        im_name = os.path.basename(im_path)
        time_id = int(im_name[:-5].split("_")[1], 16)
        timestamps_list.append(time_id)

    # sort data
    time_im = sorted(zip(timestamps_list, img_list))
    img_list2 = [im for __, im, in time_im]
    timestamps_list2 = [t for t, __ in time_im]

    return timestamps_list2, img_list2


def sort_by_time(rgb_list, nir_list):
    if not rgb_list or not nir_list:
        return
    img_twins = []
    rgb_timestamps, rgb_list = name_to_timestamp(rgb_list)
    nir_timestamps, nir_list = name_to_timestamp(nir_list)
    stat_test(rgb_timestamps, nir_timestamps)

    nir_timestamps_arr = np.asarray(nir_timestamps)
    for rgb_path, rgb_time_id in zip(rgb_list, rgb_timestamps):
        time_diff = abs(nir_timestamps_arr-rgb_time_id)
        if min(time_diff) < MIN_TIME_DIFF:
            nir_path = nir_list[np.argmin(time_diff)]
            img_twins.append([rgb_path, nir_path])

    return img_twins


def sort_images(img_dir, last_im_id=0):
    rgb_list = []
    nir_list = []
    dir_content = os.listdir(img_dir)
    # print(dir_content)
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
    return copy_twins(img_dir, img_twins, last_im_id)


def all_directories(main_dir, num_offset):
    dir_content = os.listdir(main_dir)
    for im_dir in dir_content:
        im_dir_path = os.path.join(main_dir, im_dir)
        if os.path.isdir(im_dir_path):
            print(im_dir_path)
            print(f"Last im_id: {num_offset}")
            num_offset = sort_images(im_dir_path, num_offset)


def del_twins(main_dir):
    response = input(f"Delete all twins directories in {main_dir}?  ")
    if response != "yes":
        return
    dir_content = os.listdir(main_dir)
    for im_dir in dir_content:
        twin_path = os.path.join(main_dir, im_dir, "twins")
        if os.path.exists(twin_path):
            print(f"Removing: {twin_path}")
            shutil.rmtree(twin_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('img_dir', help='Images directory.')
    parser.add_argument("--num-offset", "-n", help="Initial image id", default=0, type=int)
    parser.add_argument('--all', '-a', help='Apply on all subdirectories.', action="store_true")
    parser.add_argument("--del-twins", help='Delete twins directories', action="store_true")
    args = parser.parse_args()

    if args.del_twins:
        del_twins(args.img_dir)
    elif args.all:
        all_directories(args.img_dir, args.num_offset)
    else:
        sort_images(args.img_dir, args.num_offset)
