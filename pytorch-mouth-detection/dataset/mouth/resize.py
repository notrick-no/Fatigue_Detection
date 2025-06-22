import cv2
import os

dir = "./closed_mouth"
for root,dirs,files in os.walk(dir):
    for file in files:
        filepath = os.path.join(root,file)
        try:
            image = cv2.imread(filepath)
            dim = (64,64)
            resized = cv2.resize(image,dim)
            path = "./resize_closed_mouth/" + file
            cv2.imwrite(path,resized)
        except:
            print(filepath)
            os.remove(filepath)