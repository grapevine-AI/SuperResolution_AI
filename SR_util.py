import cv2
import numpy as np
from numpy.typing import NDArray
from scipy.special import j1
from numba import njit

@njit("f8(f8,f8,f8,f8,f8,f8)",cache=True)
def CsigmaC(i:float,j:float,sigma1:float,sigma2:float,theta:float,beta:float) -> float:
    C:NDArray[np.float64]=np.array([[i],
                                    [j]])
    R:NDArray[np.float64]=np.array([[np.cos(theta),-np.sin(theta)],
                                    [np.sin(theta),np.cos(theta)]])
    sigma:NDArray[np.float64]=np.array([[sigma1**2,0.0],
                                        [0.0,sigma2**2]])
    
    SIGMA:NDArray[np.float64]=R@sigma@R.T

    return np.power(C.T@np.linalg.inv(SIGMA)@C,beta)[0,0]

@njit("f8[:,:](i4,f8,f8,f8,f8)",cache=True)
def getGaussKernel(size:int,sigma1:float,sigma2:float,theta:float,beta:float=1.0) -> NDArray[np.float64]:
    if size%2==0:
        #raise ValueError("size must be odd-number")
        print("warning: size must be odd-number")

    kernel:NDArray[np.float64]=np.zeros((size,size))
    k:int=(size-1)//2

    for i in range(size):
        for j in range(size):
            kernel[i,j]=np.exp(-1/2*CsigmaC(i-k,j-k,sigma1,sigma2,theta,beta))

    N=np.sum(kernel)
    kernel=kernel/N

    return kernel

@njit("f8[:,:](i4,f8,f8,f8,f8)",cache=True)
def getPlateauKernel(size:int,sigma1:float,sigma2:float,theta:float,beta:float=1.0) -> NDArray[np.float64]:
    if size%2==0:
        #raise ValueError("size must be odd-number")
        print("warning: size must be odd-number")

    kernel:NDArray[np.float64]=np.zeros((size,size))
    k:int=(size-1)//2

    for i in range(size):
        for j in range(size):
            kernel[i,j]=1/(1+CsigmaC(i-k,j-k,sigma1,sigma2,theta,beta))

    N=np.sum(kernel)
    kernel=kernel/N

    return kernel

def sinc(i:int,j:int,omega:float,eps:float=1e-12) -> float:
    norm=np.linalg.norm(np.array([i,j]))

    if (i==0) and (j==0):
        norm+=eps

    return omega/(2*np.pi*norm)*j1(omega*norm)

def getSincKernel(size:int,omega:float) -> NDArray[np.float64]:
    if size%2==0:
        raise ValueError("size must be odd-number")

    kernel:NDArray[np.float64]=np.zeros((size,size))
    k:int=(size-1)//2

    for i in range(size):
        for j in range(size):
            kernel[i,j]=sinc(i-k,j-k,omega)
    
    return kernel

def USM(value:float=1/9) -> NDArray[np.float64]:
    kernel:NDArray[np.float64]=np.full((3,3),-value)
    kernel[1,1]=1+value*8
    return kernel

@njit("u1[:,:,:](f8[:,:,:],u1,u1)",cache=True)
def clip(img:NDArray[np.float64],min:int,max:int) -> NDArray[np.uint8]:
    return np.clip(img,min,max).astype(np.uint8)

@njit("u1[:,:,:](u1[:,:,:],f8,b1)",cache=True)
def GaussNoise(img:NDArray[np.uint8],sigma:float,gray:bool):
    mu:float=0.5

    if gray==True:
        noise:NDArray[np.float64]=np.random.normal(mu,sigma,(img.shape[0],img.shape[1],1))
    else:
        noise:NDArray[np.float64]=np.random.normal(mu,sigma,img.shape)

    return clip((img+noise),0,255)

#Reference:スパース表現によるポアソンノイズの除去（平成28年度電気情報関係学会）
#sigma(i,j)^2=I(i,j)/I_ALL*sigma^2
#in paper, sigma(i,j)=I(i,j)*sigma probably
@njit("u1[:,:,:](u1[:,:,:],f8,b1)",cache=True)
def PoissonNoise(img:NDArray[np.uint8],scale:float,gray:bool) -> NDArray[np.uint8]:
    BGR:tuple[float,float,float]=(0.114,0.587,0.299)
    x,y,_=img.shape
    #gray_scale=BGR[0]*img[:,:,0]+BGR[1]*img[:,:,1]+BGR[2]*img[:,:,2]
    noise:NDArray[np.float64]=np.zeros(img.shape)
    #I_ALL=np.average(gray_scale)

    if gray==True:
        gray_scale:NDArray[np.float64]=BGR[0]*img[:,:,0]+BGR[1]*img[:,:,1]+BGR[2]*img[:,:,2]

        for i in range(x):
            for j in range(y):
                #lamda=np.sqrt(gray_scale[i,j]/I_ALL)*scale
                lamda:float=gray_scale[i,j]*scale
                seed:float=np.random.poisson(lamda)-lamda+0.5
                noise[i,j]=seed
    else:
        for i in range(x):
            for j in range(y):
                for k in range(3):
                    #lamda=np.sqrt(img[i,j,k]/I_ALL)*scale
                    lamda:float=img[i,j,k]*scale
                    noise[i,j,k]=np.random.poisson(lamda)-lamda+0.5

    return clip((img+noise),0,255)

def JPEG(img:NDArray[np.uint8],quality:int) -> NDArray[np.uint8]:
    ret,encoded=cv2.imencode(".jpg",img,(cv2.IMWRITE_JPEG_QUALITY,quality))
    decoded:NDArray[np.uint8]=cv2.imdecode(encoded,flags=cv2.IMREAD_COLOR)
    return decoded

@njit("u1[:,:,:](u1[:,:,:])",cache=True)
def fliplr(img:NDArray[np.uint8]) -> NDArray[np.uint8]:
    return np.fliplr(img)
    #return img[:,::-1]

@njit("u1[:,:,:](u1[:,:,:],u1)",cache=True)
def rot(img:NDArray[np.uint8],k:int) -> NDArray[np.uint8]:
    return np.rot90(img,k)

@njit("u1[:,:,:](u1[:,:,:])",cache=True)
def randflip(img:NDArray[np.uint8]) -> NDArray[np.uint8]:
    if np.random.randint(0,2):
        return fliplr(img)
    else:
        return img

@njit("u1[:,:,:](u1[:,:,:])",cache=True)
def randrotate(img:NDArray[np.uint8]) -> NDArray[np.uint8]:
    return rot(img,np.random.randint(0,4))

@njit("u1[:,:,:](u1[:,:,:],UniTuple(i4,2))",cache=True)
def randcrop(img:NDArray[np.uint8],cropsize:tuple[int,int]) -> NDArray[np.uint8]:
    h,w,_=img.shape

    top:int=np.random.randint(0,h-cropsize[0]+1)
    left:int=np.random.randint(0,w-cropsize[1]+1)

    ret:NDArray[np.uint8]=img[top:top+cropsize[0],left:left+cropsize[1]]

    return ret

@njit("UniTuple(u1[:,:,:],2)(u1[:,:,:],u1[:,:,:],UniTuple(i4,2))",cache=True)
def pairedcrop(img_small:NDArray[np.uint8],img_large:NDArray[np.uint8],cropsize_small:tuple[int,int]) -> tuple[NDArray[np.uint8],NDArray[np.uint8]]:
    if img_large.shape[0]%img_small.shape[0]!=0:
        #raise ValueError("img_big.shape[0] must be multiple of img_small.shape[0]")
        print("Error: img_big.shape[0] must be multiple of img_small.shape[0]")

    scale:int=img_large.shape[0]//img_small.shape[0]
    h,w=img_small.shape[0],img_small.shape[1]

    top:int=np.random.randint(0,h-cropsize_small[0]+1)
    left:int=np.random.randint(0,w-cropsize_small[1]+1)

    ret_small:NDArray[np.uint8]=img_small[top:top+cropsize_small[0],left:left+cropsize_small[1]]
    ret_large:NDArray[np.uint8]=img_large[scale*top:scale*(top+cropsize_small[0]),scale*left:scale*(left+cropsize_small[1])]

    return ret_small,ret_large