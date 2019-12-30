from data_generator import trainGenerator, testGenerator
import numpy as np
from unet_model import set_model, second_model
from keras.callbacks import TensorBoard, ModelCheckpoint
import os
import cv2
import matplotlib.pyplot as plt
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, confusion_matrix


test_image_path = "data/test/image/"
test_label_path = "data/test/label/"
img_size = (512,512)
img_shape = (512,512,1)

test_data = testGenerator(test_image_path,test_label_path,img_size,img_shape)

# get the test data
test_image = np.asarray(test_data[0])
test_label = np.asarray(test_data[1])


#############################################################
##### This is the first model
#############################################################
# model = set_model()
# # identify the training result and load it
# if os.path.isfile('vessel_unet.hdf5'):
#     print("[TEST INFO]: Loading the training result")
#     model.load_weights('vessel_unet.hdf5')
# model_checkpoint = ModelCheckpoint('vessel_unet.hdf5', monitor='val_acc', verbose=1, save_best_only=True)
# print("[TEST INFO]: Finished Loading")
######################################################################################################################

#############################################################
##### This is the second model
#############################################################
model = second_model()
# identify the training result and load it
if os.path.isfile('vessel_unet_2.hdf5'):
    print("[TEST INFO]: Loading the training result")
    model.load_weights('vessel_unet_2.hdf5')
model_checkpoint = ModelCheckpoint('vessel_unet_2.hdf5', monitor='val_acc', verbose=1, save_best_only=True)
print("[TEST INFO]: Finished Loading second model")
######################################################################################################################



y_pred = model.predict(test_image)

y_pred_threshold = []

i = 0
result_path = "data/test/result/"
for y in y_pred:
    _, temp = cv2.threshold(y, 0.5, 1, cv2.THRESH_BINARY)
    y_pred_threshold.append(temp)
    filename = f"{result_path}{i}.jpg"
    cv2.imwrite(filename, temp) 
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    # axes[0].imshow(img)

    axes[0].imshow(np.reshape(y*255, img_size))

    axes[1].imshow(np.reshape(temp , img_size))
    i += 1

tn, fp, fn, tp = confusion_matrix(test_label, y_pred_threshold).ravel()

print('Accuracy:', accuracy_score(test_label, y_pred_threshold))

print('Sensitivity:', recall_score(test_label, y_pred_threshold))

print('Specificity', tn / (tn + fp))

print('NPV', tn / (tn + fn))

print('PPV', tp / (tp + fp))
print('AUC:', roc_auc_score(test_label, list(np.ravel(y_pred))))

print('Precision:', precision_score(test_label, y_pred_threshold))