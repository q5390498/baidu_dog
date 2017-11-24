import h5py
import numpy as np
from sklearn.utils import shuffle
from keras.utils import plot_model
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import SGD
from keras import regularizers
np.random.seed(2017)

X_train = []
X_val = []
X_test = []
#y_val = []
'''"gap_ResNet50.h5", "gap_VGG16.h5","'''
for filename in [ "gap_Xception.h5", "gap_ResNet50.h5"]:
    with h5py.File(filename, 'r') as h:
        X_train.append(np.array(h['train']))
        X_val.append(np.array(h['val']))
        X_test.append(np.array(h['test']))
        y_train = np.array(h['train_label'])
        y_val = np.array([h['val_label']])
        test_name = np.array(h['test_name'])

X_train = np.concatenate(X_train, axis=1)
X_val = np.concatenate(X_val, axis=1)
X_test = np.concatenate(X_test, axis=1)

y_train = y_train.reshape((-1, 1))
y_val = y_val.reshape((-1, 1))
X_train = np.vstack((X_train,X_val))
print np.shape(y_train),np.shape(y_val)
y_train = np.vstack((y_train, y_val))
X_train, y_train = shuffle(X_train, y_train)
print np.shape(X_val), np.shape(y_val)

print np.shape(X_val), np.shape(y_val)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
cls = LogisticRegression()
#cls.fit(X_train, y_train)

#print cls.score(X_val, y_val)
'''

print np.shape(X_train)
from keras.models import *
from keras.layers import *
print np.shape(X_train)
input_tensor = Input(X_train.shape[1:])
x = input_tensor
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(100, activation='sigmoid',kernel_regularizer=regularizers.l2(0.0001))(x)
model = Model(input_tensor, x)
plot_model(model, 'combine.png')

auto_lr = ReduceLROnPlateau()
model.compile(optimizer=SGD(lr=0.001,momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

from keras.utils.np_utils import to_categorical

#y_train = to_categorical(y_train, nb_classes=None)
#y_train = y_train.reshape((-1, 1))
y = np.zeros((len(X_train),100))
for i in xrange(len(y_train)):
    y[i, y_train[i]] = 1

y_v = np.zeros((len(X_val),100))
for i in xrange(len(y_val)):
    y_v[i, y_val[i]] = 1

model.fit(X_train, y, batch_size=200, epochs=500, validation_split=0.2,callbacks=[auto_lr])
model.save('model.h5')
y_pred = model.predict(X_val, verbose=1)
acc = np.sum(y_pred == y_val)
print acc

#from sklearn.tests

print y_pred
y_pred = y_pred.clip(min=0.005, max=0.995)
'''
from keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255,)
valid_generator = test_datagen.flow_from_directory(
    '/home/zyh/PycharmProjects/baidu_dog/crop_val',
    target_size=(299, 299),
    batch_size=48,
    shuffle=False,
    class_mode='categorical'
)
import operator
print(valid_generator.class_indices)
label_idxs = sorted(valid_generator.class_indices.items(), key=operator.itemgetter(1))
from keras.models import load_model
model = load_model('model.h5')
pred = model.predict(X_test)
print (np.argmax(pred, axis=1))
with open('pred.txt', 'w') as f:
    for i, p in enumerate(pred):
        f.write(str(label_idxs[np.argmax(pred[i])][0]).split('_')[-1]+'\t'+str(test_name[i]).split('/')[-1].split('.')[0]+'\n')