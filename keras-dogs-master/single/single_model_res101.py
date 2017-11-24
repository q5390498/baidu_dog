from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import os
from keras.applications import ResNet50
from keras.models import load_model, Model
from keras.layers import Input, Dense, Dropout
from keras.utils import plot_model
from keras.optimizers import Adam, SGD
from keras.models import Sequential

batch_size = 8
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    rotation_range=45,
    zoom_range=0.2,
    horizontal_flip=True
)
val_datagen = ImageDataGenerator(
    rescale=1./255
)

train_generator = train_datagen.flow_from_directory(
    '/home/zyh/PycharmProjects/baidu_dog/new_crop_train_augment',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    '/home/zyh/PycharmProjects/baidu_dog/crop_val',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode = 'categorical'
)

early_stopping = EarlyStopping(monitor='val_loss', patience=3)

if os.path.exists('dog_single_dense161.h5'):
    model = load_model('dog_single_dense161.h5')
else:
    #input_tensor = Input(shape=(299, 299, 3))
    #base_model = ResNet50(include_top=True, weights='imagenet')
    import sys
    sys.path.append('.')
    from resnet101_keras import resnet101_model
    from densenet161 import DenseNet


    base_model = DenseNet(classes=100)
    base_model.layers.pop()
    base_model.outputs = [base_model.layers[-1].output]
    base_model.layers[-1].outbound_nodes = []
    base_model.output_layers = [base_model.layers[-1]]

    input = Input(shape = (224, 224, 3), name = 'img')
    feature = base_model(input)
    pred_layer = Dense(units=100, activation='softmax', name='prob')(
        Dropout(0.5)(
            Dense(2048, activation='relu')(feature)
        )
    )
    model = Model(inputs=input, outputs=pred_layer)
    plot_model(model=model, to_file='dense161_model.png')

    for layer in base_model.layers:
        layer.trainable = True
    model.compile(optimizer=Adam(lr=0.01),
                  loss={'prob': 'categorical_crossentropy'},
                  metrics=['accuracy'])

    model.fit_generator(generator=train_generator, steps_per_epoch=200, epochs=100, validation_steps=20, validation_data=val_generator, callbacks=[early_stopping])
    model.save('dog_single_dense161.h5')

for i, layer in enumerate(model.layers[1].layers):
    print(i, layer.name)

for layer in model.layers[1].layers[:88]:
    layer.trainable = False
for layer in model.layers[1].layers[88:]:
    layer.trainable = True

model.compile(optimizer=SGD(lr = 0.0001, momentum=0.9), loss={'prob': 'categorical_crossentropy'}, metrics=['accuracy'])
save_model = ModelCheckpoint('dense161-weights-{epoch:02d}-{val_acc:.3f}.h5')
au_lr = ReduceLROnPlateau()
model.fit_generator(generator=train_generator, validation_data=val_generator, validation_steps=20, steps_per_epoch=200, epochs=100, callbacks=[early_stopping, save_model, au_lr])
model.save('dense161-weights-tune.h5')