import cv2 as cv
import numpy

cap = cv.VideoCapture(0)

while True:
    
    _, frame = cap.read()