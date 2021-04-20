import cv2
import glob,os
import numpy as np

#img_path0 = "/home/pasonatech/Desktop/20_/datasize_2083/noaug_2083/test_img/*.jpg"
img_path1 = "/home/pasonatech/Desktop/5_22/5_21_output/*.jpg"
img_path2 = "/home/pasonatech/Desktop/5_22/5_22_output/*.jpg"


save_img_path = "/home/pasonatech/Desktop/5_22/combined/"


def file_name(img_path: str):
    filenames = glob.glob(img_path)
    filenames.sort()
    return filenames
    

def img_load(i:int,filenames:list,text:str): 
    images = cv2.imread(filenames[i])
    images_resize = cv2.resize(images,(512,512))
    cv2.putText(
        img = images_resize,
        text = text,
        org = (0,100),
        fontFace = font,
        fontScale = 3,
        color = (0, 255, 0),
        thickness = 2,
        lineType= cv2.LINE_AA
        )
    return images_resize

#filenames0 = file_name(img_path0)
filenames1 = file_name(img_path1)
filenames2 = file_name(img_path2)

font = cv2.FONT_HERSHEY_SIMPLEX

if (len(filenames1)!=len(filenames2)):
    print("Number of files in both directory doesn't match")
    
elif ((len(filenames1)or len(filenames2)) == 0):
    print("Folder is EMPTY!!!")
else:
    print(len(filenames1))
    print(len(filenames2))
    for i in range(len(filenames1)):
        #images0_resize = img_load(i,filenames0,'5/21')
        images1_resize = img_load(i,filenames1,'5/21')
        images2_resize = img_load(i,filenames2,'5/22')
        
        
        print("Complete ",i)
        c_img = cv2.hconcat([images1_resize, images2_resize])
        cv2.imwrite(os.path.join(save_img_path,str(i)+'.jpg'),c_img)
        cv2.imshow("Window", c_img)
        cv2.waitKey(1000)
        

cv2.destroyAllWindows()