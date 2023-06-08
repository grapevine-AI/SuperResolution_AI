import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow_addons.layers import SpectralNormalization as SN

def UpSampling2D_nearest(inputs,scale:int=2):
    x=tf.repeat(inputs,scale,axis=1)
    x=tf.repeat(x,scale,axis=2)
    return x

class EDSR_Block(keras.Model):
    def __init__(self,channel:int=64,kernel_size:tuple[int,int]=(3,3),activation=tf.nn.leaky_relu) -> None:
        super().__init__()
        self.conv1=layers.Conv2D(channel,kernel_size,padding="same",activation=None)
        self.act=layers.Activation(activation)
        self.conv2=layers.Conv2D(channel,kernel_size,padding="same",activation=None)
        self.add=layers.Add()
    def call(self,x):
        h=self.conv1(x)
        h=self.act(h)
        h=self.conv2(h)
        return self.add([x,h*0.1])

class UpSample(keras.Model):
    def __init__(self,out_channel:int=64,scale:int=2,kernel_size:tuple[int,int]=(3,3),activation=tf.nn.leaky_relu) -> None:
        super().__init__()
        self.scale=scale
        self.conv=layers.Conv2D(out_channel*scale**2,kernel_size,padding="same",activation=None,dtype="float32")
        self.act=layers.Activation(activation,dtype="float32")
    def call(self,x):
        x=self.conv(x)
        x=self.act(x)
        return tf.nn.depth_to_space(x,self.scale)

class Real_SRNet(keras.Model):
    def __init__(self,channel:int=64,scale:int=2,kernel_size:tuple[int,int]=(3,3),activation=tf.nn.leaky_relu) -> None:
        super().__init__()
        self.conv1=layers.Conv2D(channel,kernel_size,padding="same",activation=None,dtype="float32")
        self.block1=EDSR_Block(channel)
        self.block2=EDSR_Block(channel)
        self.block3=EDSR_Block(channel)
        self.block4=EDSR_Block(channel)
        self.block5=EDSR_Block(channel)
        self.block6=EDSR_Block(channel)
        self.block7=EDSR_Block(channel)
        self.block8=EDSR_Block(channel)
        self.block9=EDSR_Block(channel)
        self.block10=EDSR_Block(channel)
        self.block11=EDSR_Block(channel)
        self.block12=EDSR_Block(channel)
        self.block13=EDSR_Block(channel)
        self.block14=EDSR_Block(channel)
        self.block15=EDSR_Block(channel)
        self.block16=EDSR_Block(channel)
        self.conv2=layers.Conv2D(channel,kernel_size,padding="same",activation=None,dtype="float32")
        self.add=layers.Add(dtype="float32")
        self.upsample1=UpSample(channel,scale)
        #self.upsample2=UpSample(channel)
        self.drop=layers.SpatialDropout2D(0.1)
        self.conv3=layers.Conv2D(3,kernel_size,padding="same",activation=None,dtype="float32")
    def call(self,x):
        h=self.conv1(x)
        fx=self.block1(h)
        fx=self.block2(fx)
        fx=self.block3(fx)
        fx=self.block4(fx)
        fx=self.block5(fx)
        fx=self.block6(fx)
        fx=self.block7(fx)
        fx=self.block8(fx)
        fx=self.block9(fx)
        fx=self.block10(fx)
        fx=self.block11(fx)
        fx=self.block12(fx)
        fx=self.block13(fx)
        fx=self.block14(fx)
        fx=self.block15(fx)
        fx=self.block16(fx)
        fx=self.conv2(fx)
        h=self.add([h,fx])
        h=self.upsample1(h)
        #h=self.upsample2(h)
        h=self.drop(h)
        return self.conv3(h)

class U_Net(keras.Model):
    def __init__(self,channel:int=64,activation=tf.nn.leaky_relu,bias:bool=False) -> None:
        super().__init__()
        self.conv0=layers.Conv2D(channel,kernel_size=3,strides=1,padding="same",activation=None,dtype="float32")
        self.act0=layers.Activation(activation)
        #downsample
        self.conv1=SN(layers.Conv2D(channel*2,4,2,"same",activation=None,use_bias=bias))
        self.act1=layers.Activation(activation)
        self.conv2=SN(layers.Conv2D(channel*4,4,2,"same",activation=None,use_bias=bias))
        self.act2=layers.Activation(activation)
        self.conv3=SN(layers.Conv2D(channel*8,4,2,"same",activation=None,use_bias=bias))
        self.act3=layers.Activation(activation)
        #upsample
        self.unpool3=layers.Lambda(UpSampling2D_nearest,arguments={"scale":2})
        self.conv4=SN(layers.Conv2D(channel*4,3,1,"same",activation=None,use_bias=bias))
        self.act4=layers.Activation(activation)
        self.add4_2=layers.Add()
        self.unpool4=layers.Lambda(UpSampling2D_nearest,arguments={"scale":2})
        self.conv5=SN(layers.Conv2D(channel*2,3,1,"same",activation=None,use_bias=bias))
        self.act5=layers.Activation(activation)
        self.add5_1=layers.Add()
        self.unpool5=layers.Lambda(UpSampling2D_nearest,arguments={"scale":2})
        self.conv6=SN(layers.Conv2D(channel,3,1,"same",activation=None,use_bias=bias))
        self.act6=layers.Activation(activation)
        self.add6_0=layers.Add()
        #finishing
        self.conv7=SN(layers.Conv2D(channel,3,1,"same",activation=None,use_bias=bias))
        self.act7=layers.Activation(activation)
        self.conv8=SN(layers.Conv2D(channel,3,1,"same",activation=None,use_bias=bias))
        self.act8=layers.Activation(activation)
        self.conv9=layers.Conv2D(1,3,1,"same",activation=None,dtype="float32")
    def call(self,x):
        x0=self.act0(self.conv0(x))
        #downsample
        x1=self.act1(self.conv1(x0))
        x2=self.act2(self.conv2(x1))
        x3=self.act3(self.conv3(x2))
        #upsample
        x3=self.unpool3(x3)
        x4=self.act4(self.conv4(x3))
        x4=self.add4_2([x2,x4])
        x4=self.unpool4(x4)
        x5=self.act5(self.conv5(x4))
        x5=self.add5_1([x5,x1])
        x5=self.unpool5(x5)
        x6=self.act6(self.conv6(x5))
        x6=self.add6_0([x6,x0])
        #finishing
        x7=self.act7(self.conv7(x6))
        x8=self.act8(self.conv8(x7))
        return self.conv9(x8)