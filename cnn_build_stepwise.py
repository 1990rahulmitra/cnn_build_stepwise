#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import matplotlib.pyplot as plt

# we will visualize the below mentioned image later on
# change the path and provide the path from your file system 
img = cv2.imread('C:\\Users\\rmitra\\Desktop\\photograph\\fruits\\fruits-360\\Training\\Mango\\13_100.jpg')


# In[2]:


import numpy as np
from sklearn.datasets import load_files

# change the path and provide the path from your file system  
training_folder = 'C:\\Users\\rmitra\\Desktop\\photograph\\fruits\\fruits-360\\Training'
test_folder = 'C:\\Users\\rmitra\\Desktop\\photograph\\fruits\\fruits-360\\Test'


# ## Preprocessing starts here

# ### loading dataset

# In[3]:


def load_mydata(folder_path):
    dataset = load_files(folder_path)
    output_labels = np.array(dataset['target_names'])
    file_names = np.array(dataset['filenames'])
    output_class = np.array(dataset['target'])
    return file_names,output_class,output_labels
    

# Loading training and test dataset
x_training, y_training, output_labels = load_mydata(training_folder)
x_test, y_test,_ = load_mydata(test_folder)
#Loading finished

print('Training image datset size : ' , x_training.shape[0])
print('Test image dataset size : ', x_test.shape[0])


# In[4]:


count_output_classes = len(np.unique(y_training))
print("Number of ouput classes : ",count_output_classes)


# ### output classes are converted to one-hot vector

# In[5]:


from keras.utils import np_utils
y_training = np_utils.to_categorical(y_training,count_output_classes)
y_test = np_utils.to_categorical(y_test,count_output_classes)


# ### testset splitted into validation and test dataset 
# Generally for creating validaion set we should use some standard technique 
# like k-way cross validation on training set.
# In competitions where you don't know the output label of testset, 
# there anyway you don't have a way to create validation set from testset. 

# For the sake of simplicity ,during demo I created validation set from testset, since 
# for my case I know the output label for testset.
# But genarally, you should try to create validation set from training set.

# In[6]:


x_test,x_validation = x_test[8000:],x_test[:8000]
y_test,y_vaildation = y_test[8000:],y_test[:8000]
print('Vaildation set size : ',x_validation.shape)
print('Test set size : ',x_test.shape)


# ### all the images are converted to arrays

# In[7]:



from keras.preprocessing.image import load_img,img_to_array

def image_to_array_conversion(filenames):
    img_array=[]
    for f in filenames:
        # Image to array conversion
        img_array.append(img_to_array(load_img(f)))
    return img_array


x_training = np.array(image_to_array_conversion(x_training))
x_validation = np.array(image_to_array_conversion(x_validation))
x_test = np.array(image_to_array_conversion(x_test))


# ### normalization of training,validation and testset

# In[8]:



x_training = x_training.astype('float32')/255
x_validation = x_validation.astype('float32')/255
x_test = x_test.astype('float32')/255


# ## Preprocessing ends here

# ## CNN building starts here
# 
# #### Recommendation is to use GPU system for larger Dataset
# #### For GPU you need to use GPU version of tensorflow and keras.

# In[13]:


from keras.models import Sequential
from keras.layers import Conv2D,MaxPooling2D
from keras.layers import Activation, Dense, Flatten, Dropout


model = Sequential()

#Addition of Convolution layer
model.add(Conv2D(filters = 8, kernel_size = 2,activation= 'relu',input_shape=(100,100,3)))
model.add(MaxPooling2D(pool_size=2))


# In[14]:

# function for printing the original image of a mango
def print_image(model1,img1) : 
    img_batch = np.expand_dims(img1,axis=0)
    print(img_batch.shape)
    img_conv = model1.predict(img_batch)
    print(img_conv.shape)
    #img2 = np.squeeze(img_conv,axis=0)
    #print(img2.shape)
    plt.imshow(img2)
	

# function for visualizing the image after pooling operation.
# In the last line of this function the 7th activation map has been printed.
# You can check other activation maps by just changing the value of last index.
 
def print_pooled_image(model1,img1):
    img_batch = np.expand_dims(img1,axis=0)
    print(img_batch.shape)
    img_conv = model1.predict(img_batch,verbose=1)
    print(img_conv.shape)
    plt.matshow(img_conv[0, :, :, 6], cmap='viridis')


# ## Print image of a Mango

# In[15]:


plt.imshow(img)


# ## Print the same image after applying Convolution and pooling operation

# In[16]:


print_pooled_image(model,img)


# ## Adding other layers of CNN

# In[17]:


#Addition of Pooing layer
#model.add(MaxPooling2D(pool_size=2))

#Addition of Convolution Layer and Pooling Layer for 3 more times
model.add(Conv2D(filters = 16,kernel_size = 2,activation= 'relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 32,kernel_size = 2,activation= 'relu'))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(filters = 64,kernel_size = 2,activation= 'relu'))
model.add(MaxPooling2D(pool_size=2))

#Flattening the pooled images
model.add(Flatten())

#Adding hidden layer to Neural Network
model.add(Dense(200))
model.add(Activation('relu'))
model.add(Dropout(0.4))

#Adding output Layer
model.add(Dense(95,activation = 'softmax'))
#model.summary()


# ## CNN building ends here

# ## Model evaluation

# In[18]:


#compiling the model
model.compile(metrics=['accuracy'],
              loss='categorical_crossentropy',
              optimizer='adam'
              )


# In[19]:


batch_size = 20


# ### Training

# In[20]:


history = model.fit(x_training,y_training,
        epochs=30,
        batch_size = 20,        
        validation_data=(x_validation, y_vaildation),
        verbose=2, shuffle=True)


# In[21]:


# evaluate and print test accuracy
score = model.evaluate(x_test, y_test, verbose=0)
print('\n', 'Test accuracy:', score[1])


## Plotting accuracy and loss

import matplotlib.pyplot as plt 
   
 # plot for accuracy  
   
plt.subplot(211)  
plt.plot(history.history['acc'])  
plt.plot(history.history['val_acc'])  
plt.title('Accuracy', fontsize='large')  
plt.xlabel('Epoch')
plt.ylabel('Accuracy')  
plt.legend(['training', 'validation'], loc='upper left')
plt.show()

# plot for loss  
   
plt.subplot(212)  
plt.plot(history.history['loss'])  
plt.plot(history.history['val_loss'])  
plt.title('Loss', fontsize='large')  
plt.xlabel('Epoch')
plt.ylabel('Loss')  
plt.legend(['training', 'validation'], loc='upper left')  
plt.show()

# ## Predicting the the fruit classes

# Here randomly 8 images have been printed. Among that 4 have been classified correctly
# and 4 have been wrongly classified. 

# In[22]:


y_pred = model.predict(x_test)

fig = plt.figure(figsize=(16, 5))

r_count = 0
for i, idx in enumerate(np.random.choice(x_test.shape[0], size=8000, replace=False)):
    pred_idx = np.argmax(y_pred[idx])
    true_idx = np.argmax(y_test[idx])
    
    if pred_idx == true_idx :
        r_count += 1
        ax = fig.add_subplot(1, 4, r_count, xticks=[], yticks=[])
        ax.imshow(np.squeeze(x_test[idx]))
        ax.set_title("Predicted: {} \n Actual: {}".format(output_labels[pred_idx], output_labels[true_idx]),
                 color=("green"))
        if r_count == 4 :
            break
            
fig1 = plt.figure(figsize=(16, 5))

w_count = 0
for i, idx in enumerate(np.random.choice(x_test.shape[0], size=8000, replace=False)):
    pred_idx = np.argmax(y_pred[idx])
    true_idx = np.argmax(y_test[idx])
    
    if pred_idx != true_idx :
        w_count += 1
        ax1 = fig1.add_subplot(1, 4, w_count, xticks=[], yticks=[])
        ax1.imshow(np.squeeze(x_test[idx]))
        ax1.set_title("Predicted: {} \n Actual: {}".format(output_labels[pred_idx], output_labels[true_idx]),
                 color=("red"))
        if w_count == 4 :
            break


# In[ ]:




