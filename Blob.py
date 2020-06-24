import cv2
import math as mt
class Blob:

    contour = []
    bounding_rect = None
    db_diagonal_size = None
    h = None
    w = None
    x = None
    y = None
    db_aspect_ratio = None
    area = None
    
    def __init__(self,_contour):
        self.contour = _contour

    def calculate_metrics(self):
        (x,y,w,h) = cv2.boundingRect(self.contour)
        bounding_rect = cv2.boundingRect(self.contour)
        self.h = h
        self.w = w
        self.x = x
        self.y = y
        self.area = w * h
        self.center_position_x = (x + x  + w)/2
        self.center_position_y = (y + y  + h)/2

        self.db_diagonal_size = mt.sqrt(mt.pow(w,2) + mt.pow(h,2))

        self.db_aspect_ratio = w / h