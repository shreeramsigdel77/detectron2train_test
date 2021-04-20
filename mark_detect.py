import cv2
import numpy as np
import os


test_path = "/home/pasonatech/Desktop/7/7_27/infer_result/real_image30k/good_bb/41.jpg"

def viewImage(img):
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL) #custom resize, can use window_autosize -adjust automatically
    cv2.imshow('Frame',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#range of silver color
low_silver = np.array([128,128,128])
high_silver = np.array([211,211,211])

#range of pink color
low_pink = np.array([199,21,133])
high_pink = np.array([255,192,203])



#read image
img = cv2.imread(test_path)

viewImage(img)
#transform to HSV
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
viewImage(hsv)

#Threshold with color range for specific colors

mask_silver = cv2.inRange(hsv,low_silver,high_silver)
mask_pink = cv2.inRange(hsv,low_pink,high_pink)


#Bitwise operation with the masks and original image
output_pink = cv2.bitwise_and(img,img, mask=mask_pink)
output_silver = cv2.bitwise_and(img,img, mask=mask_silver)


#Output
viewImage(output_pink)
viewImage(output_silver)