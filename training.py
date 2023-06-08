from my_loader import MyLoader,dataloader

from time import time
from multiprocessing import Process,JoinableQueue as Queue
from numpy.typing import NDArray

import numpy as np

N:int=10
DATA_NUM:int=800
PATCH_PER_IMG:int=40
QUEUE_SIZE:int=1
THREAD_NUM:int=10

SCALE:int=2
EPOCHS:int=30
BATCH_SIZE:int=8
G_CHANNEL:int=64
D_CHANNEL:int=48
G_LRATE:float=1e-4
D_LRATE:float=1e-4
DATA_DIR:str="DIV2K_train_HR"
PRETRAINED:str="Real_SRNet_36_1_2x"
MODEL_NAME:str="Real_SRGAN_36_1_2x_48_SGD"
NOTE:str="D's chnnel is 48, D's optimizer is SGD(lr=1e-3)"

def CPU_Process(queue:"Queue[tuple[NDArray[np.uint8],NDArray[np.uint8]]]") -> None:
    loader:MyLoader=MyLoader(SCALE)
    print("Data-Generator ready")

    for i in range(EPOCHS*PATCH_PER_IMG//N):
        queue.join()
        data:tuple[NDArray[np.uint8],NDArray[np.uint8]]=dataloader(DATA_DIR,loader,thread=THREAD_NUM)
        queue.put(data)
        print(str(i+1)+"-set images synthesized")

    print("Image-Synthesis Complete")

def GPU_Process(queue:"Queue[tuple[NDArray[np.uint8],NDArray[np.uint8]]]") -> None:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import mixed_precision as mp
    from base_net import Real_SRNet,U_Net
    from inference import auto_evaluation

    physical_devices=tf.config.list_physical_devices("GPU")
    tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(physical_devices[0],"GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0],True)

    mp.set_global_policy("mixed_float16")

    # weight=tf.keras.models.load_model("saved_model/"+PRETRAINED)
    # weight.save_weights("cache.h5")
    # generator=Real_SRNet(channel=G_CHANNEL)
    # generator.build((None,128,128,3))
    # generator.load_weights("cache.h5")
    generator=tf.keras.models.load_model("saved_model/"+PRETRAINED)
    discriminator=U_Net(channel=D_CHANNEL)

    @tf.function(jit_compile=True)
    def train_step(lr,hr) -> None:
        with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
            sr=generator(lr,training=True)
            
            real_output=discriminator(hr,training=True)
            fake_output=discriminator(sr,training=True)

            g_loss=g_MAE_loss(sr,hr)+0.02*g_GAN_loss(tf.ones_like(fake_output),fake_output)
            d_loss=0.5*d_real_loss(tf.ones_like(real_output),real_output)+0.5*d_fake_loss(tf.zeros_like(fake_output),fake_output)

            g_loss*=128
            d_loss*=128

        g_grad=g_tape.gradient(g_loss,generator.trainable_variables)
        d_grad=d_tape.gradient(d_loss,discriminator.trainable_variables)

        g_grad=[(grad/128) for grad in g_grad]
        d_grad=[(grad/128) for grad in d_grad]

        g_optimizer.apply_gradients(zip(g_grad,generator.trainable_variables))
        d_optimizer.apply_gradients(zip(d_grad,discriminator.trainable_variables))

        g_train_loss(g_loss/128)
        d_train_loss(d_loss/128)

    g_MAE_loss=keras.losses.MeanAbsoluteError()
    g_GAN_loss=keras.losses.Hinge()

    d_real_loss=keras.losses.Hinge()
    d_fake_loss=keras.losses.Hinge()

    g_optimizer=keras.optimizers.Adam(learning_rate=G_LRATE)
    #d_optimizer=keras.optimizers.Adam(learning_rate=D_LRATE)
    d_optimizer=keras.optimizers.SGD(learning_rate=D_LRATE*10)

    g_train_loss=keras.metrics.Mean(name="g_train_loss")
    d_train_loss=keras.metrics.Mean(name="d_train_loss")

    print("GPU-Training ready")

    for epoch in range(EPOCHS*PATCH_PER_IMG//N):
        data_x,data_y=queue.get()
        queue.task_done()

        x=tf.convert_to_tensor(data_x,np.uint8)
        y=tf.convert_to_tensor(data_y,np.uint8)
        train_x=tf.data.Dataset.from_tensor_slices(x)
        train_y=tf.data.Dataset.from_tensor_slices(y)
        train_ds=tf.data.Dataset.zip((train_x,train_y))
        #train_ds=train_ds.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        train_ds=train_ds.shuffle(DATA_NUM*PATCH_PER_IMG).batch(BATCH_SIZE)

        for lr,hr, in train_ds:
            train_step(tf.cast(lr,dtype=tf.float32)/255.0,tf.cast(hr,dtype=tf.float32)/127.5-1.0)

        template="CheckPoint {}, G_Loss: {:.4f}, D_Loss: {:.4f}"
        print(template.format(epoch+1,g_train_loss.result(),d_train_loss.result()))

    generator.save("saved_model/"+MODEL_NAME)
    discriminator.save("saved_model/"+MODEL_NAME+"_d")

    with open("saved_model/"+MODEL_NAME+"/parameters.txt","w") as f:
        for symbol, value in globals().items():
            if symbol.isupper():
                f.write(symbol+":"+str(value)+"\n")

    auto_evaluation(Real_SRNet(G_CHANNEL),"test_images",MODEL_NAME)

if __name__=="__main__":
    start=time()

    queue:"Queue[tuple[NDArray[np.uint8],NDArray[np.uint8]]]"=Queue(QUEUE_SIZE)

    CPU=Process(target=CPU_Process,args=(queue,))
    GPU=Process(target=GPU_Process,args=(queue,))

    CPU.start()
    GPU.start()

    CPU.join()
    GPU.join()

    print("Processing Complete")
    print("Time:"+str(time()-start))