from data_generator import trainGenerator, testGenerator
from unet_model import set_model,second_model
from keras.callbacks import TensorBoard, ModelCheckpoint


batch_size =2
train_path = "data/train"
image_folder ="image"
mask_folder = "label"
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
#model = set_model((512,512,1))
#model_checkpoint = ModelCheckpoint('vessel_unet.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min')
# model.fit_generator(train_data, steps_per_epoch=100, epochs=3, # shuffle=True,
#                     callbacks=[TensorBoard(log_dir='./unetres'), model_checkpoint])
######################################################################################################################




#############################################################
##### This is the second model
#############################################################
model = second_model((512,512,1))
model_checkpoint = ModelCheckpoint('vessel_unet_2.hdf5', monitor='loss', verbose=1, save_best_only=True, mode='min')

model.fit_generator(train_data, steps_per_epoch=600, epochs=5, # shuffle=True,
                    callbacks=[TensorBoard(log_dir='./unetres_2'), model_checkpoint])
######################################################################################################################
