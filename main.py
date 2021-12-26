import os
import numpy as np

import matplotlib.pyplot as plt
from detection.yolov3 import predict
from retrieval.AE import autoencoder
from retrieval.utils import load_db, vis_retrieval
from sklearn.metrics.pairwise import cosine_similarity


## Detect Art-Image
detected_dict = predict()
detected_dict['./data/detect_input/110.jpg'].keys()
detected_img = detected_dict['./data/detect_input/110.jpg'][0]['img']

plt.imshow(detected_img)
plt.show()
plt.close()

## Detected Image Encoding
retrieval_model = autoencoder('./weights/ae_model_50.pth')
_feature = retrieval_model.predict(detected_img)

## Database Retrieval
data_db, db_feature = load_db(db_path='./data/DB/autoencoder')
data_q_sim = cosine_similarity(_feature, db_feature[:, 1:])

## visualization
vis_retrieval(detected_img, data_db, data_q_sim)
