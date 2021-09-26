import cv2
import os
import glob


img_path = r"H:\workspace\pothole_detection\eval\result\D3Net\testing\0043.png"
img = cv2.imread(img_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)[1]
_, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.imshow('road', thresh)
cv2.waitKey(0)
# cv2.drawContours(img, contours, -1, (0, 0, 255), 3)

try:
    hierarchy = hierarchy[0]
except:
    hierarchy = []

height, width, _ = img.shape
min_x, min_y = width, height
max_x = max_y = 0

for contour, hier in zip(contours, hierarchy):
    (x, y, w, h) = cv2.boundingRect(contour)
    print(x, y, w, h)
    # min_x, max_x = min(x, min_x), min(x + w, max_x)
    # min_y, max_y = min(y, min_y), max(y + h, max_y)
    if w > 30 and h > 30:
        cv2.rectangle(img, (x, y), (x+ w, y+h), (255, 0, 0), 1)
cv2.imshow('road', img)
cv2.waitKey(0)
