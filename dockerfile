# usage: docker build -t ngc:tf .
FROM nvcr.io/nvidia/tensorflow:22.12-tf2-py3
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y libgl1-mesa-dev
RUN pip install opencv-python
RUN apt-get install -y ffmpeg
RUN pip install ffmpeg-python
RUN cp /usr/share/zoneinfo/Asia/Tokyo /etc/localtime