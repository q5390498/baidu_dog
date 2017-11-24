from keras.preprocessing.image import ImageDataGenerator
import os
from keras.models import load_model, Model, Sequential
from keras.applications import VGG16, InceptionV3
from keras.layers import Dense, Dropout, Input, Flatten
from keras.optimizers import Adam, SGD
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


train_datagen = ImageDataGenerator(rescale=1./255,
        shear_range=0.2,
        rotation_range=45,
        zoom_range=0.2,
        horizontal_flip=True)
val_datagen  = ImageDataGenerator(rescale=1./255)

batch_size = 128
train_generator = train_datagen.flow_from_directory(
    '/home/zyh/PycharmProjects/baidu_dog/crop_train',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)
val_generator = val_datagen.flow_from_directory(
    '/home/zyh/PycharmProjects/baidu_dog/crop_val',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True
)

if os.path.exists('dog_single_vgg16.h5'):
    model = load_model('dog_single_vgg16.h5')
else:
    input_tensor = Input(shape=(224, 224, 3))
    base_model = VGG16(include_top=True, weights='imagenet')
    base_model.layers.pop()
    base_model.outputs = [base_model.layers[-1].output]
    base_model.layers[-1].outbound_nodes = []
    base_model.output_layers = [base_model.layers[-1]]

    input = Input(shape=(224, 224, 3), name='img')
    feature = base_model(input)
    pred_layer = Dense(units=100, activation='softmax', name='prob')(
        Dropout(0.5)(
            Dense(4096, activation='relu')(feature)
        )
    )
    model = Model(inputs=input, outputs=pred_layer)
    plot_model(model, to_file='vgg-16--.png')
    for i, layer in enumerate(base_model.layers):
        layer.trainable = False
    model.compile(optimizer='adam', loss={'prob':'categorical_crossentropy'}, metrics=['accuracy'])
    early_stopping = EarlyStopping()
    model.fit_generator(generator=train_generator, steps_per_epoch=200, epochs=10,validation_data=val_generator,validation_steps=20, callbacks=[early_stopping])
    model.save('dog_single_vgg16.h5')

base_model = model.layers[1]
for i, layer in enumerate(base_model.layers):
    print(i, layer.name)
for layer in base_model.layers[:12]:
    layer.trainable = False
for layer in base_model.layers[12:]:
    layer.trainable = True
model.compile(optimizer=SGD(lr= 0.005, momentum=0.9), loss={'prob':'categorical_crossentropy'}, metrics=['accuracy'])
save_model = ModelCheckpoint('vgg16-weights-{epoch:02d}-{val_acc:.2f}.h5')
au_lr = ReduceLROnPlateau(factor=0.1, patience=3)
early_stopping = EarlyStopping()
model.fit_generator(generator=train_generator, validation_steps=20, epochs=100, steps_per_epoch=200, validation_data=val_generator, callbacks=[au_lr, save_model])

