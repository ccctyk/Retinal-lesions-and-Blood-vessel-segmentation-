from padding_data import padding_all

from unet_model import set_model
from data_generator import trainGenerator, testGenerator

from keras.callbacks import TensorBoard, ModelCheckpoint

orig_path = "data/train/original_retinal_images/"
new_orig_path = "data/train/new_original_retinal_images/"

hard_path = "data/train/masks_Hard_Exudates/"
new_hard_path = "data/train/new_masks_Hard_Exudates/"

#padding original img
padding_all(orig_path,new_orig_path)
# padding hard img
padding_all(hard_path,new_hard_path)


batch_size =2
train_path = "data/train"
image_folder ="new_original_retinal_images"
mask_folder = "new_masks_Hard_Exudates"
aug_dict = dict(rotation_range=0.2,
                    width_shift_range=0.05,
                    height_shift_range=0.05,
                    shear_range=0.05,
                    zoom_range=0.05,
                    horizontal_flip=True,
                    fill_mode='nearest')

train_data = trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict)


#############################################################
##### This is the first model
#############################################################
model = set_model((512,512,1))
model_checkpoint = ModelCheckpoint('vessel_unet.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min')
model.fit_generator(train_data, steps_per_epoch=500, epochs=5, # shuffle=True,
                    callbacks=[TensorBoard(log_dir='./unetres'), model_checkpoint])
######################################################################################################################
