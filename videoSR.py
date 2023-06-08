import numpy as np
import cv2
import multiprocessing
import ffmpeg

from numba import njit

INPUT_NAME:str=".mp4"
OUTPUT_NAME:str="result.mp4"
MODEL_NAME:str="Real_SRGAN_36_1_2x"
CACHE_FILE:str="cache.h5"
SCALE:int=2

@njit("f4[:,:,:,:](u1[:,:,:,:])")
def norm(img):
    a=np.array([255.0],dtype=np.float32)
    rtn=img/a
    return rtn

@njit("u1[:,:,:](f4[:,:,:])")
def clip(img):
    return np.clip(img*127.5+127.5,0.0,255.0).astype(np.uint8)

def sr(LRqueue,HRqueue,infoqueue):
    import tensorflow as tf
    import base_net as network

    from tensorflow.keras import mixed_precision as mp

    mp.set_global_policy("mixed_float16")

    physical_devices=tf.config.list_physical_devices("GPU")
    tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(physical_devices[0],"GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0],True)

    weight=tf.keras.models.load_model("saved_model/"+MODEL_NAME)
    weight.save_weights(CACHE_FILE)

    print("GPU Ready")

    info=infoqueue.get()
    h=int(info[1])
    w=int(info[0])
    frame=info[3]

    net=network.Real_SRNet(64)
    net.build((None,h,w,3))
    net.load_weights(CACHE_FILE)
    
    #net=tf.keras.models.load_model("trt_model/"+"Real_SRGAN_36_2_2x_TRT")

    print("inference start")

    for _ in range(int(frame)):
        img=LRqueue.get()
        input_tensor=tf.convert_to_tensor(norm(img),dtype=np.float32)
        output_tensor=net.predict(input_tensor)
        HRqueue.put(output_tensor)

    print("cooming soon...")

def videoread(LRqueue,infoqueue):
    cap=cv2.VideoCapture(INPUT_NAME)
    if cap.isOpened():
        print(INPUT_NAME+" accepted")

    w=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps=cap.get(cv2.CAP_PROP_FPS)
    frame=cap.get(cv2.CAP_PROP_FRAME_COUNT)

    info=[w,h,fps,frame]
    infoqueue.put(info)
    infoqueue.put(info)

    print("reading start")

    for _ in range(int(frame)):
        ret,img=cap.read()
        LRqueue.put(img[np.newaxis,:,:,:])

    print("reading complete")

def videowrite(HRqueue,infoqueue):
    info=infoqueue.get()
    w=info[0]
    h=info[1]
    fps=info[2]
    frame=info[3]

    codec=cv2.VideoWriter_fourcc(*"avc1")
    video=cv2.VideoWriter("cache.mp4",codec,fps,(int(w*SCALE),int(h*SCALE)))

    print("writing start")

    for i in range(int(frame)):
        tensor=HRqueue.get()
        img=clip(np.array(tensor[0],dtype=np.float32))
        video.write(img)
        print(i+1,"f complete",sep="")

    print("saving...")

    video.release()
    
    print("SuperResolution Complete")

def joinaudio(audio_name,video_name,output_name):
    tmp=ffmpeg.input(audio_name)
    video=ffmpeg.input(video_name)

    audio=tmp.audio

    if audio_name.endswith(".mp4") and video_name.endswith(".webm"):
        stream=ffmpeg.output(video,audio,output_name,vcodec="copy",acodec="libopus")
    else:
        stream=ffmpeg.output(video,audio,output_name,vcodec="copy",acodec="copy")

    ffmpeg.run(stream)

if __name__=='__main__':
    LRqueue=multiprocessing.Queue(5)
    HRqueue=multiprocessing.Queue(5)
    infoqueue=multiprocessing.Queue(2)

    readprocess=multiprocessing.Process(target=videoread,args=(LRqueue,infoqueue))
    srprocess=multiprocessing.Process(target=sr,args=(LRqueue,HRqueue,infoqueue))
    writeprocess=multiprocessing.Process(target=videowrite,args=(HRqueue,infoqueue))

    readprocess.start()
    srprocess.start()
    writeprocess.start()

    readprocess.join()
    srprocess.join()
    writeprocess.join()

    joinaudio(INPUT_NAME,"cache.mp4",OUTPUT_NAME)