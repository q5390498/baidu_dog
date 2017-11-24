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

    # './feature/resnet18_mu.h5',
    # './feature/resnet34.h5',
     './feature/resnet50_mu.h5',
    './feature/resnet101_mu.h5',
     './feature/resnet152_mu.h5',

    './feature/densenet121_mu.h5',
    './feature/densenet161_mu.h5',
    './feature/densenet169_mu.h5',
    './feature/densenet201_mu.h5',

    './feature/inceptionv3_mu.h5',
    './feature/xception_mu.h5',
    './feature/inceptionv4_mu.h5',
]
pre_val_feature_file = [
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

    # './feature/resnet18_mu.h5',
    # './feature/resnet34.h5',
     './feature/resnet50_mu_val.h5',
    './feature/resnet101_mu_val.h5',
     './feature/resnet152_mu_val.h5',

    './feature/densenet121_mu_val.h5',
    './feature/densenet161_mu_val.h5',
    './feature/densenet169_mu_val.h5',
    './feature/densenet201_mu_val.h5',

    './feature/inceptionv3_mu_val.h5',
    './feature/xception_mu_val.h5',
    './feature/inceptionv4_mu_val.h5',
]

for filename in pre_train_feature_file:
    with h5py.File(filename, 'r') as f:
        X_train.append(np.array(f['train_feature']))
        X_val.append(np.array(f['val_feature']))
        X_test.append(f['test_feature'][:])
v = []
for filename in pre_val_feature_file:
    with h5py.File(filename, 'r') as f:
        v.append(np.array(f['val_feature']))

X_train = np.concatenate(X_train, axis=1)
X_val = np.concatenate(X_val, axis=1)
X_test = np.concatenate(X_test, axis=1)
v = np.concatenate(v, axis=1)
print np.shape(v)
train_txt = open('./train_mu.txt').readlines()
y_train = [int(line.strip('\n').split(' ')[-1]) for line in train_txt]

val_txt = open('./val_mu.txt').readlines()
y_val = [int(line.strip('\n').split(' ')[-1]) for line in val_txt]

test_txt = open('./test.txt').readlines()
test_name = [line.strip('\n') for line in test_txt]

v_label = open('./mu_val_aug.txt').readlines()
y_v = [int(line.strip('\n').split(' ')[-1]) for line in v_label]

X_train = np.vstack((X_train, v))
y_train = np.hstack((y_train, y_v))
print np.shape(X_train)

from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train, num_classes=100)
y_val = to_categorical(y_val, num_classes=100)

from keras.layers import Dense, Dropout, Input
from keras.models import Model
from keras.optimizers import SGD
from keras.regularizers import l2
def get_model():
    input_tensor = Input(X_train.shape[1:])
    x = Dense(1024, activation='relu')(input_tensor)
    x = Dropout(0.89)(x)
    x = Dense(100, activation='sigmoid')(x)
    model = Model(input_tensor, x)
    return model

from sklearn.utils import shuffle
X_train, y_train = shuffle(X_train, y_train)

model = get_model()
model.compile(optimizer=SGD(lr=0.001,momentum=0.9),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

from keras.callbacks import ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
au_lr = ReduceLROnPlateau()
save_model = ModelCheckpoint('blend_{val_acc:.3f}.h5')
#model.fit(X_train, y_train, batch_size=200, epochs=80, validation_split=0.2, callbacks=[au_lr, save_model])
#model.save('blend_model.h5')

# from keras.preprocessing.image import ImageDataGenerator
# test_datagen = ImageDataGenerator(rescale=1./255)
# valid_generator = test_datagen.flow_from_directory(
#     '/home/zyh/PycharmProjects/baidu_dog/crop_val',
#     target_size=(299, 299),
#     batch_size=48,
#     shuffle=False,
#     class_mode='categorical'
# )
# import operator
# print(valid_generator.class_indices)
# label_idxs = sorted(valid_generator.class_indices.items(), key=operator.itemgetter(1))
from keras.models import load_model
model = load_model('./blend_0.905.h5')
label_map_txt = open('/home/zyh/PycharmProjects/baidu_dog/newLabel_to_label.txt').readlines()
label_map = {}
for line in label_map_txt:
    label_map[int(line.strip('\n').split(' ')[0])] = line.strip('\n').split(' ')[1]


pred = model.predict(X_test)
pred = np.argmax(pred, axis=1)
#print (np.argmax(pred, axis=1))
with open('pred.txt', 'w') as f:
    for i, p in enumerate(pred):
        f.write(label_map[pred[i]]+'\t'+str(test_name[i]).split('/')[-1].split('.')[0]+'\n')


