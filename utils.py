import cv2
from skimage.transform import resize
import numpy as np
import pickle

EMPTY = True
NOT_EMPTY = False
MODEL = pickle.load(open("model.p", "rb"))

def empty_or_not(spot_bgr):
    flat_data = []
    resize_img = resize(spot_bgr , (15,15,3))
    flat_data.append(resize_img.flatten())
    flat_data = np.array(flat_data)
    predict  = MODEL.predict(flat_data)

    if predict==0:
        return EMPTY
    else:
        return NOT_EMPTY


def get_parking_spots_bboxes(connected_components):
    (totalLabels, label_ids, values, centroid) = connected_components
    slots = []
    coef = 1
    for i in range(totalLabels):
        x = int(values[i , cv2.CC_STAT_LEFT]*coef)
        y = int(values[i , cv2.CC_STAT_TOP]*coef)
        w  = int(values[i  , cv2.CC_STAT_WIDTH]*coef)
        h = int(values[i , cv2.CC_STAT_HEIGHT])
        slots.append([x,y,w,h])
    return slots

