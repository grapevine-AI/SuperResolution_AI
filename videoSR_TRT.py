import numpy as np
import cv2
import multiprocessing
import ffmpeg

INPUT_NAME:str=".mp4"
OUTPUT_NAME:str="result.mp4"
MODEL_NAME:str="Real_SRGAN_36_1_2x_TRT_INT8"
SCALE:int=2

def sr(LRqueue,HRqueue,infoqueue):
    import torch
    from torch2trt import TRTModule

    device=torch.device("cuda:0")

    print("GPU Ready")

    info=infoqueue.get()
    frame=info[3]

    trt_net=TRTModule()
    trt_net.load_state_dict(torch.load("trt_model/"+MODEL_NAME+".pth"))

    print("inference start")

    with torch.no_grad():
        for _ in range(int(frame)):
            img=LRqueue.get()
            input_tensor=torch.from_numpy(img)
            input_fp32=input_tensor.to(device).to(torch.float32)/255.0
            input_fp16=input_fp32.half()
            output_fp16=trt_net(input_fp16)
            output_fp32=output_fp16.float()*127.5+127.5
            output_clip=torch.clamp(output_fp32,0.0,255.0).to(torch.uint8).permute(0,2,3,1)[0]
            output_tensor=output_clip.cpu()
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
        LRqueue.put(img.transpose(2,0,1)[np.newaxis,:,:,:])

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
        img=np.array(tensor,dtype=np.uint8)
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