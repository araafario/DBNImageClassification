import numpy as np
import os
import cv2

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from dbn.tensorflow import SupervisedDBNClassification

np.random.seed(1337)  # for reproducibility
IMG_SIZE = 32

dataX = []
dataY = []
label = []
img = []
flagged = []
path = 0
labeling = 0
path_normal = "D:\\PycharmProjects\Workspace\\DBNPneunomia\\chest_xray\\train\\NORMAL\\"
path_pneunomia = "D:\\PycharmProjects\Workspace\\DBNPneunomia\\chest_xray\\train\\PNEUMONIA\\"


for i in range(0, 2):
    if i == 0:
        path = path_normal
        labeling = "NORMAL"
    elif i == 1:
        path = path_pneunomia
        labeling = "PNEUMONIA"

    for file in os.listdir(path):
        img = cv2.imread(path + file)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = img.astype('float32')
        dataX.append(img)
        label.append(labeling)
        if labeling == "NORMAL":
            dataY.append(0)
        else:
            dataY.append(1)

dataX = np.array(dataX)
dataX = dataX/255.0
dataX = dataX.reshape(len(dataX), -1)
dataY = np.array(dataY)

X_train, X_test, Y_train, Y_test = train_test_split(dataX, dataY, test_size=0.3, random_state=0)
# Training
classifier = SupervisedDBNClassification(hidden_layers_structure=[256, 256],
                                         learning_rate_rbm=0.05,
                                         learning_rate=0.1,
                                         n_epochs_rbm=10,
                                         n_iter_backprop=100,
                                         batch_size=10,
                                         activation_function='relu',
                                         dropout_p=0.2)

classifier.fit(X_train, Y_train)

# Save the model
classifier.save('model.pkl')

# Restore
classifier = SupervisedDBNClassification.load('model.pkl')

# Test
Y_pred = classifier.predict(X_test)

if __name__ == '__main__':
    print('Done.\nAccuracy: %f' % accuracy_score(Y_test, Y_pred))
