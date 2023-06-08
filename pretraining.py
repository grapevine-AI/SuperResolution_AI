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
EPOCHS:int=60
BATCH_SIZE:int=8
CHANNEL:int=64
L_RATE:float=2e-4
DATA_DIR:str="DIV2K_train_HR"
MODEL_NAME:str="Real_SRNet_36_1_2x"
NOTE:str="This is test run."

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
    from base_net import Real_SRNet

    physical_devices=tf.config.list_physical_devices("GPU")
    tf.config.list_physical_devices("GPU")
    tf.config.set_visible_devices(physical_devices[0],"GPU")
    tf.config.experimental.set_memory_growth(physical_devices[0],True)

    mp.set_global_policy("mixed_float16")

    net=Real_SRNet(channel=CHANNEL)

    @tf.function(jit_compile=True)
    def train_step(x,t) -> None:
        with tf.GradientTape() as tape:
            predictions=net(x,training=True)
            loss=loss_object(t,predictions)*128
            #scaled_loss=optimizer.get_scaled_loss(loss)

        #scaled_gradient=tape.gradient(scaled_loss,net.trainable_variables)
        #gradient=optimizer.get_unscaled_gradients(scaled_gradient)
        gradient=tape.gradient(loss,net.trainable_variables)
        gradient=[(grad/128) for grad in gradient]
        optimizer.apply_gradients(zip(gradient,net.trainable_variables))

        train_loss(loss/128)
        train_accuracy(t,predictions)

    loss_object=keras.losses.MeanAbsoluteError()
    optimizer=keras.optimizers.Adam(learning_rate=L_RATE)
    #optimizer=mp.LossScaleOptimizer(optimizer)

    train_loss=keras.metrics.Mean(name="train_loss")
    train_accuracy=keras.metrics.MeanSquaredError(name="train_accuracy")

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

        for lr,hr in train_ds:
            train_step(tf.cast(lr,dtype=tf.float32)/255.0,tf.cast(hr,dtype=tf.float32)/127.5-1.0)

        templete="CheckPoint {}, Loss: {:.4f}, MSE: {:.4f}"
        print(templete.format(epoch+1,train_loss.result(),train_accuracy.result()))

    net.save("saved_model/"+MODEL_NAME)

    with open("saved_model/"+MODEL_NAME+"/parameters.txt","w") as f:
        for symbol, value in globals().items():
            if symbol.isupper():
                f.write(symbol+":"+str(value)+"\n")

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