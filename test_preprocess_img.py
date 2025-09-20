import cv2
import numpy as np
from os import listdir

clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))

img_list = ["/home/raph/ai-framework/temp_no_preprocessed/"+f for f in listdir("/home/raph/ai-framework/temp_no_preprocessed/") ]

for img_path in img_list:
    img = cv2.imread(img_path)
    ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb_img)

    # Apply CLAHE to the Y channel
    y_clahe = clahe.apply(y)
    clahe_ycrcb = cv2.merge([y_clahe, cr, cb])
    clahe_img = cv2.cvtColor(clahe_ycrcb, cv2.COLOR_YCrCb2BGR)

    cv2.imwrite(img_path.replace("temp_no_preprocessed", "temp"), clahe_img)

