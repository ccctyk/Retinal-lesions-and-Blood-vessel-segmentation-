#!/usr/bin/env python
# coding: utf-8

# In[66]:


# Xiaowei Zhu
# This file is used to generate training data
from keras.preprocessing.image import ImageDataGenerator
import os
import cv2
#import gif2numpy
import numpy as np


# In[ ]:





# In[38]:


generator_config = {
    "image_color_mode":"grayscale",
    "mask_color_mode": "grayscale",
    "image_save_prefix": "image",
    "mask_save_prefix": "mask",
    "flag_multi_class": False,
    "num_class": 2,
    "save_to_dir":None,
    "target_size":(512,512),
    "seed":1
}


# In[52]:


def normalization(image,mask):
    # convert to 0-1
    image = image/255.
    mask  = mask/255.
    
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    
    return (image,mask)


# In[58]:

# datat generator cited from https://github.com/zhixuhao/unet/blob/master/main.py
def trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict):

    image_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    
    image_generator = image_datagen.flow_from_directory(
        train_path,
        classes = [image_folder],
        class_mode = None,
        color_mode = generator_config["image_color_mode"],
        target_size = generator_config["target_size"],
        batch_size = batch_size,
        save_to_dir = generator_config["save_to_dir"],
        save_prefix  = generator_config["image_save_prefix"],
        seed = generator_config["seed"])
    
    mask_generator = mask_datagen.flow_from_directory(
        train_path,
        classes = [mask_folder],
        class_mode = None,
        color_mode = generator_config["mask_color_mode"],
        target_size = generator_config["target_size"],
        batch_size = batch_size,
        save_to_dir = generator_config["save_to_dir"],
        save_prefix  = generator_config["mask_save_prefix"],
        seed = generator_config["seed"])
    
    train_generator = zip(image_generator, mask_generator)
    
    for (image,mask) in train_generator:
        i,m = normalization(image,mask)
        yield (i,m)
    print("Finsihed Loading Train Data")

# In[81]:


#img_size = (512,512)
#img_shape = (512,512,1)
def testGenerator(test_image_path,test_label_path,img_size,img_shape):
    # read the list of test image in dir
    image_list=os.listdir(test_image_path)
    image_list = sorted(image_list)
    print(image_list)
    test_test_data= []
    test_label_data = []
    for img_name in image_list:
        #print(test_image_path+img_name)
        img = cv2.imread(test_image_path+img_name,1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # resize to 512, 512
        img = cv2.resize(img,img_size)
        img = img.astype('float32')
        img = img/255.
        img = np.reshape(img,img_shape)
        test_test_data.append(img)
        
    
    label_list=os.listdir(test_label_path)
    label_list = sorted(label_list)
    #print("HHHH")
    print(label_list)
    
    for img_name in label_list:
        # read gif image
        #np_frames,extensions,img_spec =gif2numpy.convert(test_label_path+img_name)
        #print("TTTTT")
        img = cv2.imread(test_label_path+img_name,1)
        #print(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        #print(img.shape)
        # resize to 512, 512
        img = cv2.resize(img,img_size)
        #img = cv2.resize(np_frames[0],img_size)
        img = img.astype('float32')
        img = img/255.
        #img = img[:,:,0]
        img = np.reshape(img,img_shape)
        test_label_data.append(img)
    print("Finsihed Loading Test Data")

    return ([test_test_data,test_label_data])


# In[ ]:




# In[82]:


# batch_size =2
# train_path = "data/train"
# image_folder ="images"
# mask_folder = "label"
# aug_dict = dict(rotation_range=0.2,
#                     width_shift_range=0.05,
#                     height_shift_range=0.05,
#                     shear_range=0.05,
#                     zoom_range=0.05,
#                     horizontal_flip=True,
#                     fill_mode='nearest')


# In[83]:


# myGene = trainGenerator(batch_size,train_path,image_folder,mask_folder,aug_dict)


# In[84]:


# In[85]:


# test_image_path = "data/test/image/"
# test_label_path = "data/test/label/"
# img_size = (512,512)
# img_shape = (512,512,1)


# # In[86]:


# testGenerator(test_image_path,test_label_path,img_size,img_shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




