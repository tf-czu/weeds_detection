"""
  Tool for weeds labeling in images.
"""
import os
import cv2
import numpy as np
import json

from image_processing import transform_nir, get_ndvi_im, get_excess_green, get_com_im, noise_reduction


class LabelWeeds:
    def __init__(self, path, out_json):
        self.path = path
        assert out_json.endswith(".json"), out_json
        self.out_json = out_json

        self.labels = [0x30, 0x31, 0x32, 0x33, 0x34, 0x35, 0x36, 0x37, 0x38, 0x39]
        self.current_label = "0"
        self.thrvalue = 140
        self.manual_xy = None
        self.manual_wh = None
        if os.path.exists(out_json):
            with open(out_json) as json_file:
                self.data = json.load(json_file)
            # make a backup
            with open(self.out_json+".backup", "w") as outfile:
                json.dump(self.data, outfile, indent=4)
        else:
            self.data = {}
        self.add_images()

    @staticmethod
    def make_im_to_show(im_color, bbox_list):
        im_to_show = im_color.copy()
        for x, y, w, h, kind in bbox_list:
            if kind is None:
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)
                cv2.putText(im_to_show, kind, (x + 3, y + 7), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3,
                            color=color, thickness=1)
            cv2.rectangle(im_to_show, (x, y), (x + w, y + h), color, 1)
        return im_to_show

    def make_hist_im(self, gray):
        bin_w = 2
        hist_h = 400
        hist = cv2.calcHist([gray], [0], None, [256], (0, 256), accumulate=False)
        cv2.normalize(hist, hist, alpha=0, beta=400, norm_type=cv2.NORM_MINMAX)
        hist_im = np.zeros((hist_h + 50, 512, 3), dtype=np.uint8)
        for ii in range(1, 256):
            cv2.line(hist_im, (bin_w * (ii - 1), hist_h - int(hist[ii - 1])),
                    (bin_w * (ii), hist_h - int(hist[ii])),
                    (255, 0, 0), thickness=2)
        cv2.line(hist_im, (self.thrvalue*2, 0), (self.thrvalue*2, 400), (0, 0, 255), thickness=2)
        cv2.putText(hist_im, str(self.thrvalue), (412, 448), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(0, 255, 0), thickness=2)
        return hist_im

    def make_gray_im(self, color_path, method="ndvi"):
        nir_path = self.data[color_path]["nir"]
        im_color = cv2.imread(color_path)

        im_nir = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
        nir = transform_nir(im_nir)
        if method == "ndvi":
            gray = get_ndvi_im(im_color, nir)
        elif method == "exg":
            gray = get_excess_green(im_color)
        elif method == "com":
            ndvi = get_ndvi_im(im_color, nir)
            exg = get_excess_green(im_color)
            gray = get_com_im(exg, ndvi)
        else:
            assert False, f"unknown method: {method}"
        gray = cv2.medianBlur(gray, 5)
        # gray = cv2.equalizeHist(gray)
        hist_im = self.make_hist_im(gray)
        return gray, hist_im

    def save_data(self):
        with open(self.out_json, "w") as outfile:
            json.dump(self.data, outfile, indent=4)

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

    def get_bbox(self, gray):
        assert gray is not None
        ret, binary_im = cv2.threshold(gray, self.thrvalue, 255, cv2.THRESH_BINARY)
        binary_im = noise_reduction(binary_im)
        contours, hierarchy = cv2.findContours(binary_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bbox_list = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 100:
                x, y, w, h = cv2.boundingRect(cnt)
                bbox_list.append([x, y, w, h, None])
        return bbox_list

    def make_sub_im(self, im_color, im_to_show, bbox):
        x, y, w, h, kind = bbox
        new_bbox = None
        n, m, __ = im_color.shape
        x0 = max(0, x - 100)
        y0 = max(0, y - 100)
        x1 = min(m, x + w + 100)
        y1 = min(n, y + h + 100)
        sub_im = im_color.copy()[y0:y1, x0:x1, :]
        cv2.rectangle(im_to_show, (x, y), (x + w, y + h), (0, 255, 255), 2)
        label_strip = np.ones((sub_im.shape[0], 30, 3), dtype=np.uint8) * 255
        cv2.putText(label_strip, self.current_label, (5, 28), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(255, 0, 0), thickness=2)

        if self.manual_xy and self.manual_wh is None:
            xm, ym = self.manual_xy
            cv2.circle(sub_im, (xm , ym), 4, (0, 255, 255), -1)
            return im_to_show, np.hstack((label_strip, sub_im)), new_bbox

        elif self.manual_wh:
            xm, ym = self.manual_xy
            w, h = self.manual_wh
            x = xm + x0
            y = ym + y0
            kind = None
            new_bbox = [x, y, w, h, None]

        cv2.rectangle(sub_im, (x - x0, y - y0), (x + w - x0, y + h - y0), (0, 255, 255), 1)
        if kind is not None:
            cv2.putText(sub_im, kind, (x - x0 + 3, y - y0 + 7), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.3,
                        color=(255, 0, 0), thickness=1)
        return im_to_show, np.hstack((label_strip, sub_im)), new_bbox

    def manual_label(self, event, x, y, flags, param):
        """
        The function is called during all mouse events.
        """
        if event == cv2.EVENT_LBUTTONDOWN:
            if x < 30:
                return
            x = x - 30
            if self.manual_xy is None:
                self.manual_xy = (x, y)
            else:
                x0, y0 = self.manual_xy
                if x0 >= x or y0 >= y:
                    self.manual_xy = None
                    return
                self.manual_wh = (x-x0, y-y0)

    def run(self):
        # print(self.data)
        item_list = [d for d in self.data]
        ii = 0
        jj = 0
        new_image = True
        bbox_list = None
        im_name = None
        im_color = None
        hist_im = None
        gray = None
        while True:
            if ii < 0:
                ii = 0
            elif ii >= len(item_list):
                ii = len(item_list)-1

            if new_image:
                im_name = item_list[ii]
                im_color = cv2.imread(im_name)
                gray, hist_im = self.make_gray_im(im_name)

                bbox_list = self.data[im_name].get("bbox_list")
                self.thrvalue = self.data[im_name].get("thrvalue", self.thrvalue)
                new_image = False

            if bbox_list is None:
                bbox_list = self.get_bbox(gray)

            if jj < 0:
                jj = 0
            elif jj >= len(bbox_list):
                jj = len(bbox_list)-1

            assert im_color is not None
            im_to_show = self.make_im_to_show(im_color, bbox_list)

            # current bbox
            bbox = bbox_list[jj]
            # prepare sub_img
            im_to_show, sub_im, new_bbox = self.make_sub_im(im_color, im_to_show, bbox)
            if new_bbox:
                bbox_list[jj] = new_bbox
                self.manual_xy = None
                self.manual_wh = None
            n, m, __ = im_color.shape

            cv2.namedWindow("win", cv2.WINDOW_NORMAL)
            cv2.namedWindow("sub_win", cv2.WINDOW_NORMAL)
            if hist_im is not None:
                cv2.namedWindow("hist", cv2.WINDOW_NORMAL)
                cv2.imshow("hist", hist_im)
            cv2.setMouseCallback('sub_win', self.manual_label)
            cv2.resizeWindow("win", m // 2, n // 2)
            cv2.imshow("win", im_to_show)
            cv2.imshow("sub_win", sub_im)

            k = cv2.waitKey(100) & 0xFF
            if k == ord(" "):  # label and take next sub_img
                bbox_list[jj][4] = self.current_label
                # jj += 1
            elif k == ord("n"):  # take next sub_img
                jj += 1
            elif k == ord("b"):  # take sub_img one back
                jj -= 1
            elif k == ord("r"):  # reset current sub_img
                bbox_list[jj][4] = None
            elif k in self.labels:  # set weed label
                self.current_label = chr(k)

            elif k in [0x53, ord("d")]:  # take next image
                ii += 1
                jj = 0
                new_image = True
            elif k in [0x51, ord("a")]:  # take one image back
                ii -= 1
                jj = 0
                new_image = True
            elif k == ord("+"):  # move thrvalue
                self.thrvalue += 2
                gray, hist_im = self.make_gray_im(im_name)
            elif k == ord("-"):
                self.thrvalue -= 2  # move thrvalue
                gray, hist_im = self.make_gray_im(im_name)
            elif k == ord("u"):  # update bbox_list
                bbox_list = self.get_bbox(gray)
                self.data[im_name]["thrvalue"] = self.thrvalue
            elif k == ord("s"):
                self.save_data()
            elif k == ord("q"):
                cv2.destroyAllWindows()
                break

            else:
                pass
                #print(k, chr(k))

            self.data[im_name]["bbox_list"] = bbox_list
            # print(k, chr(k), ii, jj)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='__doc__')
    parser.add_argument('path', help='Path to image directory.')
    parser.add_argument('--out', help='Specify an annotation file name (.json)', default="test.json")

    args = parser.parse_args()
    label = LabelWeeds(path=args.path, out_json=args.out)
    label.run()
