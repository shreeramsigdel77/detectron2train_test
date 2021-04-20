import cv2
import glob,os
import numpy as np
#blender_model
# img_path1 = "/home/pasonatech/Desktop/10/10_8_blender_vs_3dflow/blender_model/inference_result/textured_sim_inference/*.jpg"

# img_path1 = "/home/pasonatech/Desktop/10/10_9/inference_result/net_data_model/*.jpg"

# img_path1 ="/home/pasonatech/Desktop/10/10_9/inference_result/blender_model2/*.jpg"


#from blend model1
img_path1 ="/home/pasonatech/Desktop/10/10_8_blender_vs_3dflow/blender_model/inference_result/blender2_sim_inference/*.jpg"

#3dflow_model
# img_path2 = "/home/pasonatech/Desktop/10/10_8_blender_vs_3dflow/photogrammetry_model/inference_result/inference_result/textured_sim_inference/*.jpg"

# img_path2 = "/home/pasonatech/Desktop/10/10_8_blender_vs_3dflow/photogrammetry_model/inference_result/inference_result/web_data/*.jpg"

# img_path2 = "/home/pasonatech/Desktop/10/10_7/cresecent_inference/blender_model2/*.jpg"


#from blend model 2
img_path2 = "/home/pasonatech/Desktop/10/10_9/inference_result/blender_model2/*.jpg"

save_img_path = "/home/pasonatech/Desktop/10/10_9/combined_inference_preview/blend2_data_blendmodels/"

filenames1 = glob.glob(img_path1)
filenames1.sort()
filenames2 = glob.glob(img_path2)
filenames2.sort()

images1_resize = []
images2_resize = []
font = cv2.FONT_HERSHEY_SIMPLEX
if (len(filenames1)!=len(filenames2)):
    print("Number of files in both directory doesn't match")
    
elif ((len(filenames1)or len(filenames2)) == 0):
    print("Folder is EMPTY!!!")
else:
    print(len(filenames1))
    print(len(filenames2))
    for i in range(len(filenames1)):
        images1 = cv2.imread(filenames1[i])
        # images1_resize = cv2.resize(images1,(1024,1024))
        cv2.putText(
            img = images1,
            text = 'Blend1',
            org = (10,50),
            fontFace = font,
            fontScale = 1,
            color = (0, 0, 255),
            thickness = 2,
            lineType= cv2.LINE_AA
            )


        images2 = cv2.imread(filenames2[i])
        # images2_resize = cv2.resize(images2,(1024,1024))
        cv2.putText(
            img = images2,
            # text = '3DFlow',
            text = 'Blend2',
            org = (10,50),
            fontFace = font,
            fontScale = 1,
            color = (0, 0, 255),
            thickness = 2,
            lineType= cv2.LINE_AA
            )
        print("start")
        c_img = cv2.hconcat([images1, images2])
        cv2.imwrite(os.path.join(save_img_path,str(i)+'.jpg'),c_img)
        cv2.imshow("Window", c_img)
        cv2.waitKey(1000)
        

cv2.destroyAllWindows()