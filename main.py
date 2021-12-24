import matplotlib.pyplot as plt
from detection.yolov3 import predict
from retrieval.AE import autoencoder

detected_dict = predict()
detected_dict['./data/detect_input/110.jpg'].keys()
detected_img = detected_dict['./data/detect_input/110.jpg'][0]['img']

plt.imshow(detected_img)
plt.show()
plt.close()

retrieval_model = autoencoder('./weights/ae_model_50.pth')
retrieval_model.predict(detected_img)
