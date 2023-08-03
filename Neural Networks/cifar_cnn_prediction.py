import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# from cifar_cnn_training import class_names

# use the same classes as per the training file
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# open the image
image_path = '/Users/korvo/Documents/SoSE2022/M8/lab/'  #  Path to Image, change to the current/needed path
img = cv2.imread(str(image_path)+'inka.jpg', 1)  # image reading


cap = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
cv2.imshow("Original Image", cap)

# proper image dimensions for the model
width = 32
height = 32
dim = (width, height)

# resize image
image = cv2.resize(cap, dim, interpolation=cv2.INTER_AREA)
cv2.imshow("resized", image)

# Path to model File
model_path = '/Users/korvo/Documents/SoSE2022/M8/lab/' # Path to Model

# Load the Previously trained model
model = tf.keras.models.load_model(str(model_path)+'cifar10.hdf5', compile=True)

# Classification and prediction based on loaded model

test_x = np.expand_dims(image, axis=0)
test_y = model.predict(test_x)
class_label = np.argmax(test_y[0])
print(image, ' shows a ', class_names[class_label], ' with confidence ', test_y[0][class_label] )





# exit

cv2.waitKey(0)

cv2.destroyAllWindows()