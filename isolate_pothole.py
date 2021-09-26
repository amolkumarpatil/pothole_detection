import glob
import os
import cv2


class Isolate:
    def __init__(self):
        pass

    def process(self, gt_path, rgb_path, pothole_out):
        for img_path in glob.glob(gt_path):
            print(os.path.basename(img_path))
            img = cv2.imread(img_path)
            rgb_img = cv2.imread(os.path.join(rgb_path, os.path.basename(img_path)))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
            _, contours, hierarchy= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            try:
                hierarchy = hierarchy[0]
            except:
                hierarchy = []

            height, width, _ = img.shape

            for contour, hier in zip(contours, hierarchy):
                (x, y, w, h) = cv2.boundingRect(contour)
                if w > 20 and h > 20:
                    cv2.rectangle(rgb_img, (x, y), (x + w, y + h), (255, 0, 255), 2)
                cv2.imshow('road', rgb_img)
                cv2.waitKey(0)
                pothole_filename = os.path.join(pothole_out,os.path.basename(img_path))
                cv2.imwrite(pothole_filename, rgb_img)


if __name__ == '__main__':
    gt_path = r"H:\workspace\pothole_detection\dataset\pothole600\training\GT\*.png"
    rgb_path = r"H:\workspace\pothole_detection\dataset\pothole600\training\RGB"
    pothole_out = r"H:\workspace\pothole_detection\dataset\pothole600\training"
    isolate_pothole = Isolate()
    isolate_pothole.process(gt_path, rgb_path, pothole_out)
