import os, sys
import glob
import cPickle as pkl
import numpy as np
import PIL.Image as Image
from skimage.transform import resize
import scipy.misc
from keras import backend as K
K.set_image_dim_ordering('tf')
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras.utils.visualize_util import plot
from keras.models import load_model
#data path for input and target folders


data_path_input="/Users/pulkit/Desktop/Deep Learning/Project Code/inpainting/training_pulkit/input"
print data_path_input
imgs_input = glob.glob(data_path_input + "/*.jpeg")


data_path_target="/Users/pulkit/Desktop/Deep Learning/Project Code/inpainting/training_pulkit/target"
print data_path_target
imgs_target = glob.glob(data_path_target + "/*.jpeg")

# select some images from the input and target dataset

batch_idx = 0
batch_size = 20000

batch_imgs_input = imgs_input[batch_idx*batch_size:(batch_idx+1)*batch_size]

batch_imgs_target = imgs_target[batch_idx*batch_size:(batch_idx+1)*batch_size]

#dataset variables below

img_input_x = np.zeros((len(batch_imgs_input),64,64,3)) #batch_imgs_input

img_target_y = np.zeros((len(batch_imgs_target),32,32,3)) #batch_imgs_target


print "creating nd array for input data"
#loop over the input images
for i, img_path in enumerate(batch_imgs_input): #batch_imgs_input

    img1 = Image.open(img_path)
    if len(np.array(img1).shape) == 3:
    	img_input_x[i,:,:,:] = np.array(img1)
    else:
    	img_input_x[i,:,:,0] = np.array(img1)
    	img_input_x[i,:,:,1] = np.array(img1)
    	img_input_x[i,:,:,2] = np.array(img1)


print(np.shape(img_input_x))

print "creating nd array for target data"
#loop over the target images
for i, img_path in enumerate(batch_imgs_target): #batch_imgs_target

    img2 = Image.open(img_path)
    if len(np.array(img2).shape) == 3:
    	img_target_y[i,:,:,:] = np.array(img2)
    else:
    	img_target_y[i,:,:,0] = np.array(img2)
    	img_target_y[i,:,:,1] = np.array(img2)
    	img_target_y[i,:,:,2] = np.array(img2)

print(np.shape(img_target_y))




'''

# 1. Basic autoencoder as given in Keras blog

encoding_dim = 32  

input_img = Input(shape=(12288,))
encoded = Dense(encoding_dim, activation='relu')(input_img)
decoded = Dense(3072, activation='sigmoid')(encoded)


'''

'''

# 2. Deep autoencoder

input_img = Input(shape=(12288,))
encoded = Dense(128, activation='relu')(input_img)
encoded = Dense(64, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(64, activation='relu')(encoded)
decoded = Dense(128, activation='relu')(decoded)
decoded = Dense(3072, activation='sigmoid')(decoded)

'''

# 3. Deep Convolutional autoencoder

input_img = Input(shape=(64, 64, 3))

x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(input_img)
x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)

# at this point the representation is (8, 4, 4) i.e. 128-dimensional

x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = Convolution2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)
#x = Convolution2D(16, 3, 3, activation='relu')(x)
#x = UpSampling2D((2, 2))(x)
decoded = Convolution2D(3, 3, 3, activation='sigmoid', border_mode='same')(x)





# this model maps an input to its reconstruction
autoencoder = Model(input=input_img, output=decoded)
print "this"
print autoencoder.output_shape

plot(autoencoder, to_file='./model.png',show_shapes=True)
autoencoder.save('my_model.h5')
#Now, no need for this code fragment
'''
encoder = Model(input=input_img, output=encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer = autoencoder.layers[-1]
decoder = Model(input=encoded_input, output=decoder_layer(encoded_input))
'''

# Compile

autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

#prepare the data for train


a = img_input_x.astype('float32') / 255.
b = img_target_y.astype('float32') / 255.

#for 1. and 2.
'''
x_train = a.reshape((len(a), np.prod(a.shape[1:])))
y_train = b.reshape((len(b), np.prod(b.shape[1:])))
'''

#for 3.

x_train = a
y_train = b

print x_train.shape
print y_train.shape

#prepare the data for test

batch_idx_test = 41
batch_size_test = 2000

batch_imgs_input_test = imgs_input[batch_idx_test*batch_size_test:(batch_idx_test+1)*batch_size_test]
batch_imgs_target_test = imgs_target[batch_idx_test*batch_size_test:(batch_idx_test+1)*batch_size_test]

img_input_x_test = np.zeros((len(batch_imgs_input_test),64,64,3))

img_target_y_test = np.zeros((len(batch_imgs_target_test),32,32,3))


print "creating nd array for input data"
#loop over the input images
for i, img_path in enumerate(batch_imgs_input_test):

    img1 = Image.open(img_path)
    if len(np.array(img1).shape) == 3:
    	img_input_x_test[i,:,:,:] = np.array(img1)
    else:
    	img_input_x_test[i,:,:,0] = np.array(img1)
    	img_input_x_test[i,:,:,1] = np.array(img1)
    	img_input_x_test[i,:,:,2] = np.array(img1)


print "creating nd array for target data"
#loop over the target images
for i, img_path in enumerate(batch_imgs_target_test):

    img2 = Image.open(img_path)
    if len(np.array(img2).shape) == 3:
    	img_target_y_test[i,:,:,:] = np.array(img2)
    else:
    	img_target_y_test[i,:,:,0] = np.array(img2)
    	img_target_y_test[i,:,:,1] = np.array(img2)
    	img_target_y_test[i,:,:,2] = np.array(img2)



c = img_input_x_test.astype('float32') / 255.
d = img_target_y_test.astype('float32') / 255.

#for 1. and 2.
'''
x_test = c.reshape((len(c), np.prod(c.shape[1:])))
y_test = d.reshape((len(d), np.prod(d.shape[1:])))
'''
x_test = c
y_test = d


print x_test.shape
print y_test.shape


autoencoder.fit(x_train, y_train,
                nb_epoch=10,
                batch_size=256,
                shuffle=True,
                validation_data=(x_test, y_test))

# encode and decode some digits

'''
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)
'''

decoded_imgs = autoencoder.predict(x_test)
decoded_imgs = decoded_imgs*255

print np.shape(decoded_imgs)



# save the generated centre images in a folder

print "Save the decoded centre image of size 32X32"

outpath_decoded_centre = "/Users/pulkit/Desktop/Deep Learning/Project Code/inpainting/training_pulkit/decoded_centre"

for i, img_path in enumerate(decoded_imgs):
    
    im_input = Image.fromarray(decoded_imgs[i].astype('uint8'))

    yes_1 = os.path.join(outpath_decoded_centre, str(i+(batch_idx_test*batch_size_test))+"decoded_centre"+".jpeg")

    im_input.save(yes_1)


   # im_input.save(str(i+(batch_idx_test*batch_size_test))+"decoded_centre"+".jpeg")




# save the generated centre images overlayed on the orginal image in a folder

print "Save the decoded overlayed image of size 64X64"

outpath_decoded_overlayed = "/Users/pulkit/Desktop/Deep Learning/Project Code/inpainting/training_pulkit/decoded_overlayed"

for i, img_path in enumerate(x_test):

    center = (int(np.floor(x_test[i].shape[0] / 2.)), int(np.floor(x_test[i].shape[1] / 2.)))

    input = np.copy(x_test[i]*255)
    
    input[center[0]-16:center[0]+16, center[1]-16:center[1]+16, :] = decoded_imgs[i].astype('uint8')
    
    im_input = Image.fromarray(input.astype('uint8'))

    yes_2 = os.path.join(outpath_decoded_overlayed, str(i+(batch_idx_test*batch_size_test))+"decoded_centre"+".jpeg")

    im_input.save(yes_2)


    #im_input.save(str(i+(batch_idx_test*batch_size_test))+"decoded_full"+".jpeg") 




# save the actual central region of the orginal image in a folder

print "Save the original central image of size 64X64"

outpath_original_centre = "/Users/pulkit/Desktop/Deep Learning/Project Code/inpainting/training_pulkit/original_centre"

for i, img_path in enumerate(y_test):

    
    im_input = Image.fromarray((y_test[i]*255).astype('uint8'))

    yes_3 = os.path.join(outpath_original_centre, str(i+(batch_idx_test*batch_size_test))+"decoded_centre"+".jpeg")

    im_input.save(yes_3)

    #im_input.save(str(i+(batch_idx_test*batch_size_test))+"original_centre"+".jpeg")




# use Matplotlib (don't ask)
import matplotlib.pyplot as plt

n = 6  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display actual central region
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(y_test[i].reshape(32, 32,3))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display decoded central region
    ax = plt.subplot(3, n, i + 1 + n)
    #plt.imshow(decoded_imgs[i].reshape(32, 32,3))
    plt.imshow(decoded_imgs[i].astype('uint8'))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    #display balcked out original image

    ax = plt.subplot(3, n, i + 1 + n + n)
    plt.imshow(x_test[i].reshape(64, 64,3))
    #plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)


plt.show()




