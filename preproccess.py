import glob
import os
import cv2


class GenerateData:
    def __init__(self):
        pass

    def process(self, gt_path, rgb_path, road_out, pothole_out):
        for img_path in glob.glob(gt_path):
            print(os.path.basename(img_path))
            img = cv2.imread(img_path)
            rgb_img = cv2.imread(os.path.join(rgb_path, os.path.basename(img_path)))
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)[1]
            _, contours, hierarchy= cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

            try:
                hierarchy = hierarchy[0]
            except:
                hierarchy = []

            height, width, _ = img.shape
            min_x, min_y = width, height
            max_x = max_y = 0

            for contour, hier in zip(contours, hierarchy):
                (x, y, w, h) = cv2.boundingRect(contour)
                # min_x, max_x = min(x, min_x), min(x + w, max_x)
                # min_y, max_y = min(y, min_y), max(y + h, max_y)
                if w < 20 and h < 20:
                    continue

                # cv2.imshow('img', img)
                # cv2.waitKey(0)
                # crop road
                # cv2.rectangle(rgb_img, (x - 20, y - 20), (x - w - 20, y - h - 20), (255, 0, 0), 2)
                # cv2.imshow('road', rgb_img)
                # cv2.waitKey(0)
                try:
                    if x > 20 and y > 20:
                        road_img = rgb_img[y-70:y , x-70:x]
                    else:
                        road_img = rgb_img[y + h:y + h + 50, x + w:x + w + 50]
                    road_filename = os.path.join(road_out, os.path.basename(img_path))
                    cv2.imwrite(road_filename, road_img)
                    # cv2.imshow('img', road_img)
                    # cv2.waitKey(0)
                except:
                    pass

                # pothole
                # cv2.rectangle(rgb_img, (x - 20, y - 20), (x + w + 20, y + h + 20), (255, 0, 0), 2)
                if x > 20 and y >20:
                    pothole_img = rgb_img[y - 20:y + h + 20, x-20:x + w + 20]
                else:
                    pothole_img = rgb_img[y:y + h + 20, x:x + w + 20]
                # cv2.imshow('img', pothole_img)
                # cv2.waitKey(0)

                pothole_filename = os.path.join(pothole_out,os.path.basename(img_path))
                cv2.imwrite(pothole_filename, pothole_img)


class Preprocess:
    def __init__(self):
        pass

    def depth2gray(self, save_directory, input_dir):
        for img in glob.glob(input_dir):
            inp = cv2.imread(img)
            gray = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY)
            outname = os.path.join(save_directory, os.path.basename(img))
            print(outname)
            cv2.imwrite(outname, gray)


if __name__ == '__main__':

    data_gen = GenerateData()
    gt_path = r"H:\workspace\pothole_detection\dataset\pothole600\validation\GT\*.png"
    rgb_path = r"H:\workspace\pothole_detection\dataset\pothole600\validation\RGB"
    road_out = r"H:\workspace\pothole_detection\dataset\pothole600\validation\classification\test\road"
    pothole_out = r"H:\workspace\pothole_detection\dataset\pothole600\validation\classification\test\pothole"
    data_gen.process(gt_path, rgb_path, road_out, pothole_out)
