"""
    Sort RGB and NIR images.
"""


def sort_images(img_dir):
    pass


def all_directories(main_dir):
    pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('main_dir', help='Images directory.')
    parser.add_argument('--all', '-a', help='Apply on all subdirectories.', action="store_true")
    args = parser.parse_args()

    main_dir = args.main_dir
    all_dir = args.all