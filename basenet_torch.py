from torch import nn,Tensor
from torch.nn import functional as F
from torch.nn.utils import spectral_norm as SN

class EDSR_Block(nn.Module):
    def __init__(self,channel,kernel_size=3,padding=1,bias=True):
        super().__init__()
        self.conv1=nn.Conv2d(channel,channel,kernel_size,padding=padding,bias=bias)
        self.conv2=nn.Conv2d(channel,channel,kernel_size,padding=padding,bias=bias)
    def forward(self,x):
        h=F.leaky_relu(self.conv1(x),negative_slope=0.2)
        return x+0.1*self.conv2(h)

class Real_SRNet(nn.Module):
    def __init__(self,channel,kernel_size=3,padding=1,bias=True):
        super().__init__()
        self.conv1=nn.Conv2d(3,channel,kernel_size,padding=padding)
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
        self.conv2=nn.Conv2d(channel,channel,kernel_size,padding=padding,bias=bias)
        self.conv3=nn.Conv2d(channel,channel*4,kernel_size,padding=padding)
        #self.conv4=nn.Conv2d(channel,channel*4,kernel_size,padding=padding)
        self.drop=nn.Dropout2d(0.1)
        self.convf=nn.Conv2d(channel,3,kernel_size,padding=padding)
    def forward(self,x):
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
        h=h+self.conv2(fx)
        h=F.pixel_shuffle(F.leaky_relu(self.conv3(h),negative_slope=0.2),2)
        #h=F.pixel_shuffle(F.leaky_relu(self.conv4(h),negative_slope=0.2),2)
        h=self.drop(h)
        return self.convf(h)

class U_Net(nn.Module):
    def __init__(self,channel,bias=False):
        super().__init__()
        self.conv0=nn.Conv2d(3,channel,kernel_size=3,stride=1,padding=1)
        #downsample
        self.conv1=SN(nn.Conv2d(channel,channel*2,4,2,1,bias=bias))
        self.conv2=SN(nn.Conv2d(channel*2,channel*4,4,2,1,bias=bias))
        self.conv3=SN(nn.Conv2d(channel*4,channel*8,4,2,1,bias=bias))
        #upsample
        self.conv4=SN(nn.Conv2d(channel*8,channel*4,3,1,1,bias=bias))
        self.conv5=SN(nn.Conv2d(channel*4,channel*2,3,1,1,bias=bias))
        self.conv6=SN(nn.Conv2d(channel*2,channel,3,1,1,bias=bias))
        #finishing
        self.conv7=SN(nn.Conv2d(channel,channel,3,1,1,bias=bias))
        self.conv8=SN(nn.Conv2d(channel,channel,3,1,1,bias=bias))
        self.conv9=nn.Conv2d(channel,1,3,1,1)
    def forward(self,x):
        x0=F.leaky_relu(self.conv0(x),negative_slope=0.2,inplace=True)
        #downsample
        x1=F.leaky_relu(self.conv1(x0),negative_slope=0.2,inplace=True)
        x2=F.leaky_relu(self.conv2(x1),negative_slope=0.2,inplace=True)
        x3=F.leaky_relu(self.conv3(x2),negative_slope=0.2,inplace=True)
        #upsample
        x3=F.interpolate(x3,scale_factor=2,mode="bilinear",align_corners=False)
        x4=F.leaky_relu(self.conv4(x3),negative_slope=0.2,inplace=True)
        x4=x4+x2
        x4=F.interpolate(x4,scale_factor=2,mode="bilinear",align_corners=False)
        x5=F.leaky_relu(self.conv5(x4),negative_slope=0.2,inplace=True)
        x5=x5+x1
        x5=F.interpolate(x5,scale_factor=2,mode="bilinear",align_corners=False)
        x6=F.leaky_relu(self.conv6(x5),negative_slope=0.2,inplace=True)
        x6=x6+x0
        #finishing
        x7=F.leaky_relu(self.conv7(x6),negative_slope=0.2,inplace=True)
        x8=F.leaky_relu(self.conv8(x7),negative_slope=0.2,inplace=True)
        return self.conv9(x8)