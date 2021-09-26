from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import decode_predictions
import cv2
import numpy as np

classifier = load_model('weights/model.h5')

#################### Classification on Input ########################
class Classify:
    def __init__(self, img):
        self.img = img

    def detect(self, img):
        img1 = cv2.resize(self.img, (100, 100))
        img = image.img_to_array(img1)
        img = img / 255

        # expand image dimensions
        img = np.expand_dims(img, axis=0)
        prediction = classifier.predict(img, batch_size=None, steps=1)
        print(prediction)
        if prediction[0][0] > 0.4:
            print('pothole')
        else:
            print('road')

    def detect_n_draw_bb(self, result_d3net, in_path):
        opencv_image = result_d3net.reshape((result_d3net.shape[0], result_d3net.shape[1], 1))
        thresh = cv2.threshold(opencv_image, 170, 255, cv2.THRESH_BINARY)[1]

        result = opencv_image.copy()
        _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        try:
            hierarchy = hierarchy[0]
        except:
            hierarchy = []

        height, width, _ = opencv_image.shape
        in_img = cv2.imread(in_path)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        fontColor = (255, 255, 255)
        lineType = 1

        for contour, hier in zip(contours, hierarchy):
            (x, y, w, h) = cv2.boundingRect(contour)
            if w < 20 and h < 20:
                continue
            cv2.rectangle(in_img, (x, y), (x + w, y + h), (255, 0, 0), 1)
            cropped_img = in_img[y:y + h + 20, x:x + w + 20]
            img1 = cv2.resize(cropped_img, (100, 100))
            img = image.img_to_array(img1)
            img = img / 255

            # expand image dimensions
            img = np.expand_dims(img, axis=0)
            prediction = classifier.predict(img, batch_size=None, steps=1)
            pred_norm = np.linalg.norm(prediction[0])
            prediction_array = prediction/pred_norm
            # print(prediction_array)
            # print('decode',decode_predictions(prediction))
            if prediction_array[0][0] > 0.7:
                # print('pothole')
                cv2.rectangle(in_img, (x, y), (x + w, y + h), (255, 0, 0), 1)
                txt = 'pothole'

            else:
                # print('road')
                cv2.rectangle(in_img, (x, y), (x + w, y + h), (255, 0, 0), 1)
                txt = 'road'
            cv2.putText(in_img, txt,
                        (x, y),
                        font,
                        fontScale,
                        fontColor,
                        lineType)
        return in_img


if __name__ == '__main__':
    img1 = cv2.imread(
        r"H:\workspace\pothole_detection\dataset\pothole600\validation\classification\test\pothole\0088.png")
    classify = Classify(img1)
    classify.detect(img1)
