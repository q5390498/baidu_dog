
# coding: utf-8

# In[1]:

import os
os.environ['CUDA_VISIBLE_DEVICES']='0'
os.system('echo $CUDA_VISIBLE_DEVICES')

import numpy as np

from keras.preprocessing import image
from keras.models import Model, load_model
from keras import backend as K
from keras.layers.normalization import BatchNormalization
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.xception import Xception

from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras.layers import Input
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D, GlobalMaxPooling2D, AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.callbacks import ModelCheckpoint, CSVLogger, LearningRateScheduler, ReduceLROnPlateau, EarlyStopping
from keras.optimizers import SGD, Adam
from keras.models import load_model
from keras.utils import to_categorical


import os
import h5py
import time


# In[8]:

# dimensions of our images.
img_width, img_height = 299, 299

classification_train_dir = './training_data_classification_train100 (copy)//train/'
classification_val_dir = './data2/validation/'
epochs = 50
batch_size = 32
model_input_shape = (img_width,img_height,3)
n_classes = 100

nb_train_samples = 18760
nb_val_samoles = 3713


# In[9]:

gen = ImageDataGenerator()

#xception, 带preprocee_input
base_model = load_model('./new_pretrained_model/Xception.h5')
base_model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))


train_generator = gen.flow_from_directory(
    classification_train_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = None,
    shuffle = False
)

train_feature = base_model.predict_generator(train_generator, train_generator.n // batch_size + 1)

'''
val_generator = gen.flow_from_directory(
    classification_val_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = None,
    shuffle = False
)

val_feature = base_model.predict_generator(val_generator, nb_val_samoles // batch_size + 1)
'''


# In[10]:

train_label = train_generator.classes
#val_label = val_generator.classes

train_label = to_categorical(train_label, num_classes = 100)
#val_label = to_categorical(val_label, num_classes = 100)

#随机打乱训练集
from sklearn.utils import shuffle
train_feature, train_label = shuffle(train_feature, train_label)


# In[11]:

input_tensor = Input(train_feature.shape[1:])
#flatten = Flatten()(input_tensor)
x = Dense(2048, activation='relu')(input_tensor)
x = Dropout(0.8)(x)
preditions = Dense(100, activation='softmax')(x)
model = Model(input_tensor, preditions)


# In[12]:

model.compile(optimizer='sgd', loss='categorical_crossentropy',
             metrics = ['accuracy'])


# In[13]:

early_stopping = EarlyStopping(monitor = 'val_loss', patience = 10, mode = 'min')
model_save_path = "./my_model/" + "weights.{epoch:02d}-{val_loss:.4f}.hdf5"
checkpoints = ModelCheckpoint(model_save_path, monitor = 'val_loss', verbose=0, 
                           save_best_only=True, save_weights_only=False, mode='min', period=1)
model.fit(train_feature, train_label, epochs=300, validation_split = 0.2, callbacks=[early_stopping, checkpoints])


# In[14]:

#加载测试集
x_test = []
with h5py.File('mul_xception_test.h5', 'r') as h:
    X_test = np.array(h['train'])


# In[15]:

test_data_dir = './test-01/'
train_classes = []
for subdir in sorted(os.listdir(classification_train_dir)):
    if os.path.isdir(os.path.join(classification_train_dir, subdir)):
        train_classes.append(subdir)
num_class = len(train_classes)
train_class_indices = dict(zip(train_classes, range(len(train_classes))))
test_file_classification = sorted(os.listdir(test_data_dir+'image/')) 
test_name = []
test_num = len(test_file_classification)
if test_num % batch_size ==0:
    iter_test = test_num//batch_size
else:
    iter_test = test_num//batch_size +1
for j in test_file_classification:
    test_name.append(j[:-4])
    
label_rec = {value:key for key, value in train_class_indices.items()}


# In[16]:

best_model = load_model('./my_model/weights.78-0.6716.hdf5')
predict = best_model.predict(X_test, verbose=1, batch_size=64)
preds_cal = np.argmax(predict,axis = -1)
preds_rec = [label_rec[i] for i in preds_cal]
fi = open('xiwang.txt','w')
for i in range(len(preds_rec)):
    fi.write(str(preds_rec[i])+'\t' + str(test_name[i])+'\n')
fi.close()


# In[ ]:



