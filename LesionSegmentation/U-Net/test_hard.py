from padding_data import padding_all
from data_generator import trainGenerator, testGenerator
import numpy as np
from unet_model import set_model
from keras.callbacks import TensorBoard, ModelCheckpoint
import os
import cv2
import matplotlib.pyplot as plt
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score, confusion_matrix

orig_path = "data/test/original_retinal_images/"
new_orig_path = "data/test/new_original_retinal_images/"

hard_path = "data/test/masks_Hard_Exudates/"
new_hard_path = "data/test/new_masks_Hard_Exudates/"

#padding original img
padding_all(orig_path,new_orig_path)
# padding hard img
padding_all(hard_path,new_hard_path)


test_image_path = "data/test/new_original_retinal_images/"
test_label_path = "data/test/new_masks_Hard_Exudates/"
img_size = (512,512)
img_shape = (512,512,1)

test_data = testGenerator(test_image_path,test_label_path,img_size,img_shape)
print("[TEST INFO]: Finished generating test data")
# get the test data
test_image = np.asarray(test_data[0])
test_label = np.asarray(test_data[1])


#############################################################
##### This is the first model
#############################################################
model = set_model()
# identify the training result and load it
if os.path.isfile('vessel_unet.hdf5'):
    print("[TEST INFO]: Loading the training result")
    model.load_weights('vessel_unet.hdf5')
model_checkpoint = ModelCheckpoint('vessel_unet.hdf5', monitor='val_acc', verbose=1, save_best_only=True)
print("[TEST INFO]: Finished Loading")
###################################################################################################

y_pred = model.predict(test_image)

y_pred_threshold = []

i = 0
#result_path = "data/test/result_hard/"
for y in y_pred:
    _, temp = cv2.threshold(y, 0.5, 1, cv2.THRESH_BINARY)
    y_pred_threshold.append(temp)
    #filename = f"{result_path}{i}.jpg"
    #cv2.imwrite(filename, temp) 
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    # axes[0].imshow(img)

    axes[0].imshow(np.reshape(y*255, img_size))

    axes[1].imshow(np.reshape(temp , img_size))
    i += 1