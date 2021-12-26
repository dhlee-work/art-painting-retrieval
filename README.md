# Art-Painting Retrieval



![Pipeline](fig/pipeline.jpg)


## Requirements
Pillow == 8.4.8, numpy==1.21.4, torch==1.10.1, torchvision==0.11.2, torchaudio==0.10.1, imaug==0.4.0



### Detection Model
We use **yolo-v3** model for detect paining object. Weights of model is released by official yolo Group
at **https://pjreddie.com/darknet/yolo/**. Run below shell code to download the weights of the model
```shell
wget https://pjreddie.com/media/files/yolov3-openimages.weights
```

### Retrieval Model
We trained several art-painting retrieval models. if you want to get our weights of models, 
please email dhlee.ie@yonsei.ac.kr

## Quick Start

ln -s /home/dhlee/DATA/wikiart wikiart