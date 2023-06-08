import tensorflow as tf
import numpy as np
import cv2
import glob

INPUT_NAME:str="4Ktest.png"
OUTPUT_NAME:str="4Kresult.png"
MODEL_NAME:str="Real_SRGAN_36_1_2x"
CACHEFILE:str="cache.h5"

def inference(net,input_name:str,output_name:str,model_name:str,cachefile:str="cache.h5") -> None:
    img=cv2.imread(input_name)
    weight=tf.keras.models.load_model("saved_model/"+model_name)
    weight.save_weights(cachefile)

    if img is not None:
        h,w,_=img.shape
    else:
        raise FileExistsError("Failed to load image")

    net(tf.zeros((1,h,w,3)))
    #net.build((None,h,w,3))
    net.load_weights(cachefile)

    input_tensor=tf.convert_to_tensor(img[np.newaxis,:,:,:],dtype=np.float32)/255.0

    output_tensor=net.predict(input_tensor)
    SR_img=np.clip(output_tensor[0]*127.5+127.5,0.0,255.0).astype(np.uint8)

    cv2.imwrite(output_name,SR_img)

def auto_evaluation(net,test_img_dir:str,model_name:str,video:bool=False,cachefile:str="cache.h5") -> None:
    if video==True:
        # TODO: support video inference and TensorRT inference
        raise NotImplementedError
    
    filenames:list[str]=glob.glob("./"+test_img_dir+"/**")

    for filename in filenames:
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            result_name:str="saved_model/"+model_name+"/"+filename.split("\\")[-1]
            inference(net,filename,result_name,model_name,cachefile)
            print(filename,"'s result is saved as",result_name) #debug
    
if __name__=="__main__":
    from base_net import Real_SRNet
    #from tensorflow.keras import mixed_precision as mp

    #mp.set_global_policy("mixed_float16")

    physical_devices=tf.config.list_physical_devices("GPU")
    tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(physical_devices[0],"GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0],True)

    net=Real_SRNet(64)
    inference(net,INPUT_NAME,OUTPUT_NAME,MODEL_NAME,CACHEFILE)