import os, sys
import glob
import cPickle as pkl
import numpy as np
import PIL.Image as Image
from skimage.transform import resize


data_path="/Users/pulkit/Desktop/test_image.png"

print data_path 
imgs = glob.glob(data_path + "/*.jpg")

print type(imgs)

img = Image.open(data_path)
img_array = np.array(img)
print np.shape(img_array)

input = np.copy(img_array)
im_input = Image.fromarray(input)

center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))
print center
target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16]

input = np.copy(target)
im_input = Image.fromarray(input)

center = (int(np.floor(target.shape[0] / 2.)), int(np.floor(target.shape[1] / 2.)))
print center
target = target[center[0]-16:center[0]+16, center[1] - 16:center[1]+16]

input_3 = np.copy(target)
im_input = Image.fromarray(input_3)
print np.shape(im_input)

im_input.save("target.jpeg")


"""
    center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))

    if len(img_array.shape) == 3:
        input = np.copy(img_array)
        #input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
          
        im_input = Image.fromarray(input)
           
        im_input.save(str(i)+"input_original"+".jpeg") 

batch_idx = 0
batch_size = 60000

batch_imgs = imgs[batch_idx*batch_size:(batch_idx+1)*batch_size]

img_array = np.zeros((len(batch_imgs),64,64,3))
print np.shape(img_array)

mscoco="/Users/pulkit/Desktop/Deep Learning/Project Code/inpainting/training_pulkit/input_original/"


"""

"""


for i, img_path in enumerate(batch_imgs):

    img = Image.open(img_path)
    img_array = np.array(img)
    #im_input = Image.fromarray(img_array)
    #im_input.save(str(i)+"input"+".jpeg")
    print np.shape(img_array)


    center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))

    if len(img_array.shape) == 3:
        input = np.copy(img_array)
        #input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = 0
          
        im_input = Image.fromarray(input)
           
        im_input.save(str(i)+"input_original"+".jpeg") 
            #target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16, :]
            #im_target = Image.fromarray(target)
            #im_target.save(str(i)+"target"+".jpeg") 
    else:
        input = np.copy(img_array)
        #input[center[0]-16:center[0]+16, center[1]-16:center[1]+16] = 0
        im_input = Image.fromarray(input)
        im_input.save(str(i)+"input_original"+".jpeg") 
            #target = img_array[center[0]-16:center[0]+16, center[1] - 16:center[1]+16]
            #im_target = Image.fromarray(target)
            #im_target.save(str(i)+"target"+".jpeg") 

"""

data_path="/Users/pulkit/Desktop/yo/2input_original.jpeg"

img = Image.open(data_path)
img_array = np.array(img)
input = np.copy(img_array)
im_input = Image.fromarray(input)
print np.shape(im_input)

im_input = np.array(im_input)
center = (int(np.floor(img_array.shape[0] / 2.)), int(np.floor(img_array.shape[1] / 2.)))

data_path_2="/Users/pulkit/Desktop/tester_3.png"

img = Image.open(data_path_2)
img_array_1 = np.array(img)
input_1 = np.copy(img_array)
im_input_1 = Image.fromarray(input_1)
print np.shape(img_array_1)

print np.shape(target)
print np.shape(input_3)
im_input[center[0]-16:center[0]+16, center[1]-16:center[1]+16] = img_array_1

print np.shape(im_input)
input_3 = np.copy(im_input)
im_input = Image.fromarray(input_3)

im_input.save("target_3.jpeg")

