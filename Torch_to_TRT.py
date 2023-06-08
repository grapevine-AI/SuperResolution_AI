import torch
from torch2trt import torch2trt
from basenet_torch import Real_SRNet

MODEL_NAME:str="Real_SRGAN_36_1_2x"
TRT_NAME:str="Real_SRGAN_36_1_2x_TRT_INT8"

BATCH_SIZE:int=1
MAX_HEIGHT:int=1080
MAX_WIDTH:int=1920
NET_CHANNEL:int=64

FP16:bool=True
INT8:bool=True

with torch.no_grad():
    net=Real_SRNet(NET_CHANNEL)
    net.load_state_dict(torch.load("trt_model/"+MODEL_NAME+".pth"))
    net.cuda().eval().half()

    x=torch.rand((BATCH_SIZE,3,MAX_HEIGHT,MAX_WIDTH)).cuda().half()

    model_trt=torch2trt(net,[x],fp16_mode=FP16,int8_mode=INT8,max_batch_size=BATCH_SIZE)

torch.save(model_trt.state_dict(),"trt_model/"+TRT_NAME+".pth")