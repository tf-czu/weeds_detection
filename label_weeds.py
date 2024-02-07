import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET

from image_processing import transform_nir, get_ndvi_im, get_excess_green, get_com_im, noise_reduction, pavel_method

class LabelWeeds:
    def __init__(self, path, out, label_names):
        self.path = path
        assert os.path.isdir(out), f"Output directory '{out}' does not exist."
        self.out_dir = out
        self.labels = [str(label) for label in label_names]  # Convert labels to strings
        self.current_label = self.labels[0]  # Set the initial label
        self.thrvalue = 140
        self.manual_xy = None
        self.manual_wh = None
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

    def make_gray_im(self, color_path, method="pavel"):
        nir_path = self.data[color_path]["nir"]
        im_color = cv2.imread(color_path)

        im_nir = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
        nir = transform_nir(im_nir)
        if method == "pavel":
            gray = pavel_method(im_color, nir)
        elif method == "ndvi":
            gray = get_ndvi_im(im_color, nir)
        elif method == "exg":
            gray = get_excess_green(im_color)
        elif method == "com":
            ndvi = get_ndvi_im(im_color, nir)
            exg = get_excess_green(im_color)
            gray = get_com_im(exg, ndvi)
        else:
            assert False, f"unknown method: {method}"
        # gray = cv2.medianBlur(gray, 5)
        # gray = cv2.equalizeHist(gray)
        hist_im = self.make_hist_im(gray)
        return gray, hist_im

    def save_data_voc(self):
        for image_name, image_data in self.data.items():
            root_elem = ET.Element("annotation")

            filename_elem = ET.SubElement(root_elem, "filename")
            filename_elem.text = os.path.basename(image_name)

            bbox_list = image_data.get("bbox_list", [])
            for bbox in bbox_list:
                label = bbox[4]
                if label:
                    object_elem = self.make_object_element(bbox[:4], label)
                    root_elem.append(object_elem)

            tree = ET.ElementTree(root_elem)
            xml_path = os.path.join(self.out_dir, f"{os.path.splitext(os.path.basename(image_name))[0]}.xml")
            tree.write(xml_path, encoding="utf-8", xml_declaration=True)

    @staticmethod
    def make_object_element(bbox, label):
        x, y, w, h = bbox
        object_elem = ET.Element("object")
        name_elem = ET.SubElement(object_elem, "name")
        name_elem.text = label
        bndbox_elem = ET.SubElement(object_elem, "bndbox")
        xmin_elem = ET.SubElement(bndbox_elem, "xmin")
        xmin_elem.text = str(x)
        ymin_elem = ET.SubElement(bndbox_elem, "ymin")
        ymin_elem.text = str(y)
        xmax_elem = ET.SubElement(bndbox_elem, "xmax")
        xmax_elem.text = str(x + w)
        ymax_elem = ET.SubElement(bndbox_elem, "ymax")
        ymax_elem.text = str(y + h)
        return object_elem        

    def add_images(self):
        assert os.path.isdir(self.path)
        file_list = os.listdir(self.path)
        for item in sorted(file_list):
            if item.lower().endswith("_rgb.tiff"):
                color_path = os.path.join(self.path, item)
                nir_path = os.path.join(self.path, item.replace("_rgb.tiff", "_nir.tiff"))
                if os.path.exists(nir_path):
                    if not color_path in self.data:
                        self.data[color_path] = {"nir": nir_path}

    def get_bbox(self, gray):
        assert gray is not None
        ret, binary_im = cv2.threshold(gray, self.thrvalue, 255, cv2.THRESH_BINARY)
        # binary_im = noise_reduction(binary_im)
        contours, hierarchy = cv2.findContours(binary_im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bbox_list = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 80:
                x, y, w, h = cv2.boundingRect(cnt)
                bbox_list.append([x, y, w, h, None])
        return bbox_list

    def annotate_all_objects(self, bbox_list):
        # Annotate all objects in the image with the current label
        for bbox in bbox_list:
            if bbox[4] is None:
                bbox[4] = self.current_label

    def draw_pre_annotated_boxes_nir(self, im_nir, bbox_list):
        nir_im_with_boxes = im_nir.copy()

        for x, y, w, h, kind in bbox_list:
            cv2.rectangle(nir_im_with_boxes, (x, y), (x + w, y + h), (0, 255, 255), 2)

        return nir_im_with_boxes

    def make_sub_im(self, im_color, im_to_show, bbox):
        x, y, w, h, kind = bbox
        new_bbox = None
        n, m, __ = im_color.shape
        x0 = max(0, x - 100)
        y0 = max(0, y - 100)
        x1 = min(m, x + w + 100)
        y1 = min(n, y + h + 100)

        # Make bounding box larger by 2 pixels
        x0 = max(0, x0 - 2)
        y0 = max(0, y0 - 2)
        x1 = min(m, x1 + 2)
        y1 = min(n, y1 + 2)

        sub_im = im_color.copy()[y0:y1, x0:x1, :]
        cv2.rectangle(im_to_show, (x, y), (x + w, y + h), (0, 255, 255), 2)
        label_strip = np.ones((sub_im.shape[0], 30, 3), dtype=np.uint8) * 255
        cv2.putText(label_strip, self.current_label, (5, 28), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                    color=(255, 0, 0), thickness=2)

        if self.manual_xy and self.manual_wh is None:
            xm, ym = self.manual_xy
            cv2.circle(sub_im, (xm, ym), 4, (0, 255, 255), -1)
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

    def switch_label(self, direction):
        labels_count = len(self.labels)
        current_index = self.labels.index(self.current_label)
        new_index = (current_index + direction) % labels_count
        self.current_label = str(self.labels[new_index])

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
            if not item_list:
                print("No images found in the specified directory.")
                break

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
            cv2.namedWindow("nir_win", cv2.WINDOW_NORMAL)  # New NIR window
            if hist_im is not None:
                cv2.namedWindow("hist", cv2.WINDOW_NORMAL)
                cv2.imshow("hist", hist_im)
            cv2.setMouseCallback('sub_win', self.manual_label)
            cv2.resizeWindow("win", m // 2, n // 2)
            cv2.imshow("win", im_to_show)
            cv2.imshow("sub_win", sub_im)

            # Display the NIR image with pre-annotated bounding boxes in the "nir_win" window
            if im_name in self.data and "nir" in self.data[im_name]:
                nir_path = self.data[im_name]["nir"]
                im_nir = cv2.imread(nir_path, cv2.IMREAD_GRAYSCALE)
                nir_im_with_boxes = self.draw_pre_annotated_boxes_nir(gray, bbox_list)
                cv2.imshow("nir_win", nir_im_with_boxes)

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
            elif k == ord("o"):  # Annotate all objects with the current label
                self.annotate_all_objects(bbox_list)
            elif k == ord("d"):  # Delete current annotation
                bbox_list.pop(jj)
            elif k == ord("+"):  # move thrvalue
                self.thrvalue += 1
                gray, hist_im = self.make_gray_im(im_name)
            elif k in [0x53, ord("e")]:  # take next image
                ii += 1
                jj = 0
                new_image = True
            elif k in [0x51, ord("w")]:  # take one image back
                ii -= 1
                jj = 0
                new_image = True    
            elif k == ord("-"):
                self.thrvalue -= 1  # move thrvalue
                gray, hist_im = self.make_gray_im(im_name)
            elif k == ord("u"):  # update bbox_list
                bbox_list = self.get_bbox(gray)
                self.data[im_name]["thrvalue"] = self.thrvalue
            elif k == ord("s"):
                self.save_data_voc()
            elif k == ord("["):  # switch to the previous label
                self.switch_label(-1)
            elif k == ord("]"):  # switch to the next label
                self.switch_label(1)
            elif k == ord("q"):
                cv2.destroyAllWindows()
                break
            else:
                pass
                # print(k, chr(k))

            self.data[im_name]["bbox_list"] = bbox_list
            # print(k, chr(k), ii, jj)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='__doc__')
    parser.add_argument('path', help='Path to image directory.')
    parser.add_argument('--out', help='Specify an output directory for annotations.', required=True)
    parser.add_argument('--labels', nargs='+', help='List of label names.')

    args = parser.parse_args()
    label = LabelWeeds(path=args.path, out=args.out, label_names=args.labels)
    label.run()
