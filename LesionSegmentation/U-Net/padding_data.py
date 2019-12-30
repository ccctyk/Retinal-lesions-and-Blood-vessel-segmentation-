import cv2
import os
import matplotlib.pyplot as plt

def padding_square(img_path,img_name,new_path):
    desired_size = 512
    im_pth = img_path +img_name
    #print(im_pth)
    im = plt.imread(im_pth)
    #print(im.shape)
    old_size = im.shape[:2] # old_size is in (height, width) format

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])

    # new_size should be in (width, height) format

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,value=color)
    
    new_path_name = new_path + img_name
    #print(new_path_name)
    cv2.imwrite(new_path_name,new_im)


def padding_all(img_dir_path,new_dir_path):
    print("Start padding path=",img_dir_path)
    
    image_list=os.listdir(img_dir_path)
    image_list = sorted(image_list)
    
    new_path = new_dir_path
    
    # create the new path
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    
    for img_name in image_list:
        padding_square(img_dir_path,img_name,new_path)
    
    print("Finished padding path=",img_dir_path)