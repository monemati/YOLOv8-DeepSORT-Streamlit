
<div align="center">

# YOLOv8 DeepSORT Streamlit

  <p>
    <a align="center" href="https://ultralytics.com/yolov8" target="_blank">
      <img width="50%" src="images/banner-yolov8.png"></a>
  </p>

<br>

<div>
    <a href="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml"><img src="https://github.com/ultralytics/ultralytics/actions/workflows/ci.yaml/badge.svg" alt="Ultralytics CI"></a>
    <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv8 Citation"></a>
    <a href="https://hub.docker.com/r/ultralytics/ultralytics"><img src="https://img.shields.io/docker/pulls/ultralytics/ultralytics?logo=docker" alt="Docker Pulls"></a>
    <br>
    <a href="https://console.paperspace.com/github/ultralytics/ultralytics"><img src="https://assets.paperspace.io/img/gradient-badge.svg" alt="Run on Gradient"/></a>
    <a href="https://colab.research.google.com/github/ultralytics/ultralytics/blob/main/examples/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
    <a href="https://www.kaggle.com/ultralytics/yolov8"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
  </div>
  <br>
</div>

## Introduction
This repository supply a user-friendly interactive interface for [YOLOv8](https://github.com/ultralytics/ultralytics) with Object Tracking and Counting capability. The interface is powered by [Streamlit](https://github.com/streamlit/streamlit).

## Features
- Feature1: Object detection task.
- Feature2: Multiple detection models. `yolov8n`, `yolov8s`, `yolov8m`, `yolov8l`, `yolov8x`
- Feature3: Multiple input formats. `Image`, `Video`, `Webcam`
- Feature4: Multiple Object Tracking and Counting.

## Run online
You can use [This](https://monemati-yolov8-deepsort-streamlit-app-et5bli.streamlit.app/) link to try an online version on Streamlit.   

## Installation
### Create a virtual environment
```commandline
# create
python -m venv yolov8-mot-streamlit

# activate
source yolov8-mot-streamlit/bin/activate
```

### Clone repository
```commandline
git clone https://github.com/monemati/YOLOv8-DeepSORT-Streamlit.git
cd YOLOv8-DeepSORT-Streamlit
```

### Install packages
```commandline
# Streamlit dependencies
pip install streamlit

# YOLOv8 dependecies
pip install -e '.[dev]'
```
### Download Pre-trained YOLOv8 Detection Weights
Create a directory named `weights` and create a subdirectory named `detection` and save the downloaded YOLOv8 object detection weights inside this directory. The weight files can be downloaded from the table below.

| Model                                                                                | size<br><sup>(pixels) | mAP<sup>val<br>50-95 | Speed<br><sup>CPU ONNX<br>(ms) | Speed<br><sup>A100 TensorRT<br>(ms) | params<br><sup>(M) | FLOPs<br><sup>(B) |
| ------------------------------------------------------------------------------------ | --------------------- | -------------------- | ------------------------------ | ----------------------------------- | ------------------ | ----------------- |
| [YOLOv8n](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt) | 640                   | 37.3                 | 80.4                           | 0.99                                | 3.2                | 8.7               |
| [YOLOv8s](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt) | 640                   | 44.9                 | 128.4                          | 1.20                                | 11.2               | 28.6              |
| [YOLOv8m](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt) | 640                   | 50.2                 | 234.7                          | 1.83                                | 25.9               | 78.9              |
| [YOLOv8l](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt) | 640                   | 52.9                 | 375.2                          | 2.39                                | 43.7               | 165.2             |
| [YOLOv8x](https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt) | 640                   | 53.9                 | 479.1                          | 3.53                                | 68.2               | 257.8             |


## Run
```commandline
streamlit run app.py
```
Then will start the Streamlit server and open your web browser to the default Streamlit page automatically.
For Object Counting, you can choose "Video" from "Select Source" combo box and use "test3.mp4" inside videos folder as an example.

## Result

![alt text](images/YOLOv8-DeepSort-Streamlit-Counting.jpg "YOLOv8 DeepSort Streamlit Counting")
  
## Acknowledgement
- https://github.com/ultralytics/ultralytics
- https://github.com/streamlit/streamlit
- https://github.com/ZQPei/deep_sort_pytorch
- https://github.com/JackDance/YOLOv8-streamlit-app
- https://github.com/MuhammadMoinFaisal/YOLOv8-DeepSORT-Object-Tracking

