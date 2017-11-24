import keras
import h5py
import numpy as np

X_train = []
X_val = []
X_test = []

pre_train_feature_file = [
    # './feature/googlenet_pet_breed.h5',

    # './feature_yolo/resnet18.h5',
    # './feature_yolo/densenet161.h5',
    # './feature_yolo/densenet169.h5',
    # './feature_yolo/densenet201.h5',
    # './feature_yolo/densenet121.h5',
    # './feature_yolo/densenet201.h5',

    # 'dpn92.h5'

    # './feature/vgg11.h5',
    # './feature/vgg13.h5',
    # './feature/vgg16.h5',
    # './feature/vgg19.h5',

    # './feature/resnet18.h5',
    # './feature/resnet34.h5',
    # './feature/resnet50.h5',
    './feature/resnet101.h5',
    # './feature/resnet152.h5',

    './feature/densenet121.h5',
    './feature/densenet161.h5',
    './feature/densenet169.h5',
    './feature/densenet201.h5',

    './feature/inceptionv3.h5',
]

for filename in pre_train_feature_file:
    with h5py.File(filename, 'r') as f:
        X_train.append(np.array(f['train_feature'][:]))
        X_val.append(f['val_feature'][:])
        X_test.append(f['test_feature'][:])

X_train = np.concatenate(X_train, axis=1)
X_val = np.concatenate(X_val, axis=1)
X_test = np.concatenate(X_test, axis=1)

train_txt = open('./train_aug_eq.txt').readlines()
val_txt = open('./val_eq.txt').readlines()
y_train = [int(line.strip('\n').split(' ')[-1]) for line in train_txt]
y_val = [int(line.strip('\n').split(' ')[-1]) for line in val_txt]

test_txt = open('./test_eq.txt').readlines()
test_name = [line.strip('\n') for line in test_txt]

from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train, num_classes=100)
y_val = to_categorical(y_val, num_classes=100)

from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras.optimizers import SGD
def get_model():
    input_tensor = Input(X_train.shape[1:])
    x = Dense(1024, activation='relu')
    x = Dropout(0.5)
    x = Dense(100, activation='sigmoid')
    model = Model(input_tensor, x)
    return model

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)

model = get_model()
model.compile(optimizer=SGD(lr=0.001,momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

from keras.callbacks import ReduceLROnPlateau
au_lr = ReduceLROnPlateau()
model.fit(X_train, y_train, batch_size=200, epochs=200, validation_data=[X_val, y_val], callbacks=[au_lr])
model.save('blend_model.h5')

from keras.preprocessing.image import ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)
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

pred = model.predict(X_test)
print (np.argmax(pred, axis=1))
with open('pred.txt', 'w') as f:
    for i, p in enumerate(pred):
        f.write(str(label_idxs[np.argmax(pred[i])][0]).split('_')[-1]+'\t'+str(test_name[i]).split('/')[-1].split('.')[0]+'\n')


