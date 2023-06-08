from __future__ import annotations
from typing import Callable

from SR_util import *

from numpy.random import uniform,randint
from multiprocessing import Pool
from functools import lru_cache

import cv2
import glob

class VanillaLoader:
    def __init__(self,scale:int=2,cropsize:int=256,N:int=10,plus:bool=True) -> None:
        self.scale = scale
        self.cropsize = cropsize
        self.N = N
        self.plus = plus

    def __call__(self,filename:str) -> tuple[NDArray[np.uint8],NDArray[np.uint8]]:
        PI:float=3.141592653589793
        tmpsize:int=400

        if (tmpsize%self.scale!=0) or (self.cropsize%self.scale!=0):
            raise ValueError('tmpsize and cropsize must be divisible by scale')
        
        whole_img:NDArray[np.uint8]=cv2.imread(filename) #type: ignore
        h,w,_=whole_img.shape

        #in paper, small image is mirror-padded bottom and right
        if (h<tmpsize) or (w<tmpsize):
            pad_h:int=max(0,tmpsize-h)
            pad_w:int=max(0,tmpsize-w)
            whole_img:NDArray[np.uint8]=cv2.copyMakeBorder(whole_img,0,pad_h,0,pad_w,cv2.BORDER_REFLECT_101)
            h,w,_=whole_img.shape

        x_list:list[NDArray[np.uint8]|None]=[None]*self.N
        y_list:list[NDArray[np.uint8]|None]=[None]*self.N

        for i in range(self.N):
            #in paper, image is cropped 400*400 at first
            if (h>tmpsize) or (w>tmpsize):
                img:NDArray[np.uint8]=randcrop(whole_img,(tmpsize,tmpsize))
            else:
                img:NDArray[np.uint8]=whole_img

            #flip&rotate
            img:NDArray[np.uint8]=randrotate(randflip(img))

            #----------1st degradation----------

            #kernel
            ksize:int=2*randint(3,11)+1
            if uniform()<0.9: #bokeh
                sigma1,sigma2=uniform(0.2,self.scale,2)
                theta:float=uniform(-PI,PI)
                rand:float=uniform()
                if rand<0.7: #Gauss
                    x:NDArray[np.uint8]=cv2.filter2D(img,-1,getGaussKernel(ksize,sigma1,sigma2,theta,1))
                elif rand<0.85: #generalized-Gauss
                    x:NDArray[np.uint8]=cv2.filter2D(img,-1,getGaussKernel(ksize,sigma1,sigma2,theta,uniform(0.5,4)))
                else: #plateau
                    x:NDArray[np.uint8]=cv2.filter2D(img,-1,getPlateauKernel(ksize,sigma1,sigma2,theta,uniform(1,2)))
            else: #sinc
                if ksize<13:
                    x:NDArray[np.uint8]=cv2.filter2D(img,-1,getSincKernel(ksize,uniform(PI/3,PI)))
                else:
                    x:NDArray[np.uint8]=cv2.filter2D(img,-1,getSincKernel(ksize,uniform(PI/5,PI)))

            #resize
            seed:float=uniform()
            if seed<0.2: #up-sample
                size:int=int(tmpsize*uniform(1,1.5))
            elif seed<0.9: #down-sample
                size:int=int(tmpsize*uniform(0.15,1))
            else: #skip
                size:int=tmpsize
            x:NDArray[np.uint8]=cv2.resize(x,(size,size),interpolation=randint(1,4)) #type: ignore [cv2.INTER_LINEAR,cv2.INTER_CUBIC,cv2.INTER_AREA]

            #noise
            if uniform()<0.4:
                gray:bool=True
            else:
                gray:bool=False
            if uniform()<0.5:
                x:NDArray[np.uint8]=GaussNoise(x,uniform(1,15),gray)
            else:
                x:NDArray[np.uint8]=PoissonNoise(x,uniform(0.05,1.5),gray)

            #jpeg
            x:NDArray[np.uint8]=JPEG(x,randint(50,96))

            #----------2nd degradation----------

            #kernel
            if uniform()<0.8: #skip or not
                ksize:int=2*randint(3,11)+1
                if uniform()<0.9: #bokeh
                    sigma1,sigma2=uniform(0.2,self.scale/2,2)
                    theta:float=uniform(-PI,PI)
                    rand:float=uniform()
                    if rand<0.7: #Gauss
                        x:NDArray[np.uint8]=cv2.filter2D(x,-1,getGaussKernel(ksize,sigma1,sigma2,theta,1))
                    elif rand<0.85: #generalized-Gauss
                        x:NDArray[np.uint8]=cv2.filter2D(x,-1,getGaussKernel(ksize,sigma1,sigma2,theta,uniform(0.5,4)))
                    else: #plateau
                        x:NDArray[np.uint8]=cv2.filter2D(x,-1,getPlateauKernel(ksize,sigma1,sigma2,theta,uniform(1,2)))
                else: #sinc
                    if ksize<13:
                        x:NDArray[np.uint8]=cv2.filter2D(x,-1,getSincKernel(ksize,uniform(PI/3,PI)))
                    else:
                        x:NDArray[np.uint8]=cv2.filter2D(x,-1,getSincKernel(ksize,uniform(PI/5,PI)))

                #resize
                h,w,_=x.shape
                seed:float=uniform()
                if seed<0.3: #up-sample
                    size:int=int(h*uniform(1,1.2))
                elif seed<0.7: #down-sample
                    size:int=int(h*uniform(0.3,1))
                else: #skip
                    size:int=h
                x:NDArray[np.uint8]=cv2.resize(x,(size,size),interpolation=randint(1,4)) #type: ignore [cv2.INTER_LINEAR,cv2.INTER_CUBIC,cv2.INTER_AREA]

                #noise
                if uniform()<0.4:
                    gray:bool=True
                else:
                    gray:bool=False
                if uniform()<0.5:
                    x:NDArray[np.uint8]=GaussNoise(x,uniform(1,15),gray)
                else:
                    x:NDArray[np.uint8]=PoissonNoise(x,uniform(0.05,1.5),gray)

            #JPEG&finish
            if uniform()<0.5: #resize->sinc->jpeg
                x:NDArray[np.uint8]=cv2.resize(x,(tmpsize//self.scale,tmpsize//self.scale),interpolation=randint(1,4)) #type: ignore [cv2.INTER_LINEAR,cv2.INTER_CUBIC,cv2.INTER_AREA]
                if uniform()<0.8: #sinc or skip
                    ksize:int=2*randint(3,10)+1
                    if ksize<13:
                        x:NDArray[np.uint8]=cv2.filter2D(x,-1,getSincKernel(ksize,uniform(PI/3,PI)))
                    else:
                        x:NDArray[np.uint8]=cv2.filter2D(x,-1,getSincKernel(ksize,uniform(PI/5,PI)))
                x:NDArray[np.uint8]=JPEG(x,randint(30,96))
            else: #jpeg->resize->sinc
                x:NDArray[np.uint8]=JPEG(x,randint(30,96))
                x:NDArray[np.uint8]=cv2.resize(x,(tmpsize//self.scale,tmpsize//self.scale),interpolation=randint(1,4)) #type: ignore [cv2.INTER_LINEAR,cv2.INTER_CUBIC,cv2.INTER_AREA]
                if uniform()<0.8: #sinc or skip
                    ksize:int=2*randint(3,10)+1
                    if ksize<13:
                        x:NDArray[np.uint8]=cv2.filter2D(x,-1,getSincKernel(ksize,uniform(PI/3,PI)))
                    else:
                        x:NDArray[np.uint8]=cv2.filter2D(x,-1,getSincKernel(ksize,uniform(PI/5,PI)))

            #paired crop
            x,img=pairedcrop(x,img,(self.cropsize//self.scale,self.cropsize//self.scale))

            #USM
            if self.plus==True:
                y:NDArray[np.uint8]=cv2.filter2D(img,-1,USM())
            else:
                y:NDArray[np.uint8]=img

            x_list[i]=x
            y_list[i]=y

        return np.array(x_list,dtype=np.uint8),np.array(y_list,dtype=np.uint8)
    
@lru_cache(1)
def image_searcher(data_dir:str="DIV2K_train_HR") -> list[str]:
    filenames:list[str]=glob.glob("./"+data_dir+"/**")
    image_list:list[str]=[]
    cnt:int=0

    for filename in filenames:
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png"):
            image_list.append(filename)
            cnt+=1

    print(str(cnt)+"-images detected")

    return image_list

# align the generated data
def align(loader_output:list[tuple[NDArray[np.uint8],NDArray[np.uint8]]]) -> tuple[NDArray[np.uint8],NDArray[np.uint8]]:
    x_output:list[NDArray[np.uint8]]=[]
    y_output:list[NDArray[np.uint8]]=[]
    
    for x_list,y_list in loader_output:
        for x,y in zip(x_list,y_list):
            x_output.append(x)
            y_output.append(y)

    return np.array(x_output,dtype=np.uint8),np.array(y_output,dtype=np.uint8)

# data-generating with multi-thread
def dataloader(data_dir:str,loadfunc:Callable[[str],tuple[NDArray[np.uint8],NDArray[np.uint8]]],thread:int=8) -> tuple[NDArray[np.uint8],NDArray[np.uint8]]:
    LIMITER:int=15

    if thread<=1:
        outputs:list[tuple[NDArray[np.uint8],NDArray[np.uint8]]]=list(map(loadfunc,image_searcher(data_dir)))
    else:
        with Pool(min(thread,LIMITER)) as p:
            outputs:list[tuple[NDArray[np.uint8],NDArray[np.uint8]]]=p.map(func=loadfunc,iterable=image_searcher(data_dir))

    return align(outputs)