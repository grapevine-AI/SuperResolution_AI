import tensorflow as tf
import torch

from basenet_torch import Real_SRNet

#Reference: https://logmi.jp/tech/articles/325685

INPUT_NAME:str="Real_SRGAN_36_1_2x"
OUTPUT_NAME:str="Real_SRGAN_36_1_2x"
NET_CHANNEL:int=64

tf_model=tf.keras.models.load_model("saved_model/"+INPUT_NAME)
torch_model=Real_SRNet(NET_CHANNEL)

tf_params=[]

for layer in tf_model.layers:
    for var in layer.weights:
        tf_params.append(var.numpy())

torch_params=torch_model.state_dict()

for var,key in zip(tf_params,torch_params):
    print("tf",var.shape)
    print("torch",torch_params[key].detach().numpy().shape)

    if len(var.shape)==4: # 2d-Conv layer
        torch_params[key].data=torch.from_numpy(var.transpose(3,2,0,1))
    elif len(var.shape)==3: # 1d-Conv layer
        torch_params[key].data=torch.from_numpy(var.transpose(2,1,0))
    elif len(var.shape)==2: # Linear layer
        torch_params[key].data=torch.from_numpy(var.transpose(1,0))
    else:
        torch_params[key].data=torch.from_numpy(var)

torch_model.load_state_dict(torch_params)
torch.save(torch_model.state_dict(),"trt_model/"+OUTPUT_NAME+".pth")