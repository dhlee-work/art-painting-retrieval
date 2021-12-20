import matplotlib.pyplot as plt
from detection.yolov3 import predict
detected_dict = predict()
detected_dict.keys()
detected_dict['./data/detect_input/110.jpg'].keys()
detected_img = detected_dict['./data/detect_input/110.jpg'][0]['img']

plt.imshow(detected_img)
plt.show()
plt.close()