import os
import numpy as np

import matplotlib.pyplot as plt
from detection.yolov3 import predict
from retrieval.AE import autoencoder
from retrieval.utils import load_db
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
data_q_sim.shape


plt.figure(figsize=(30, 15))
dummy_i = 1
i = 0
intr_sim = data_q_sim[i, :]
sort_val = np.argsort(1 - intr_sim)
for _iter in range(11):
    rank = _iter
    if _iter == 0:
        plt.subplot(2, 6, rank + dummy_i)
        img = detected_img

        plt.imshow(img)
        plt.xlabel(f'query',  # \n size:{_size}
                   fontsize=15)
        plt.xticks([], [])
        plt.yticks([], [])

    if _iter == 6:
        dummy_i = + dummy_i + 1

    if _iter != 0:
        _sim = round(intr_sim[sort_val[rank - 1]], 3)
        _name = os.path.splitext(data_db[sort_val[rank - 1]]['name'])[0]
        _path = data_db[sort_val[rank - 1]]['path']

        plt.subplot(2, 6, rank + dummy_i)
        img = plt.imread(_path)
        _size = img.shape[:2]
        _size = ','.join([str(i) for i in _size])
        _name2 = _name[:len(_name) // 2] + '\n ' + _name[len(_name) // 2:]
        plt.imshow(img)
        plt.xlabel(f'{_name2}\n similarity:{_sim}',
                   fontsize=15)
        plt.xticks([], [])
        plt.yticks([], [])

    if _iter == 0:
        plt.title('Query')
    else:
        plt.title(f'Top {rank}')
plt.show()
plt.savefig(f'./result.jpg')
plt.close()





