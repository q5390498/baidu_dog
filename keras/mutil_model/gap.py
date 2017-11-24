from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *

import h5py

def write_gap(MODEL, image_size, weights,lambda_func=None):
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)
    #base_model = MODEL(input_tensor=x, weights='imagenet', include_top=False)
    #model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))
    model = load_model(weights)
    fc_layer = Model(inputs=model.input, outputs=model.get_layer(model.layers[-3].name).output)
    train_gen = ImageDataGenerator(rescale=1./255)
    test_gen = ImageDataGenerator(rescale=1./255)
    train_generator = train_gen.flow_from_directory('/home/zyh/PycharmProjects/baidu_dog/crop_train', image_size, shuffle=False,
                                              batch_size=9)
    val_generator = test_gen.flow_from_directory('/home/zyh/PycharmProjects/baidu_dog/crop_val', image_size, shuffle=False,
                                             batch_size=9, class_mode=None)
    test_generator = test_gen.flow_from_directory('/home/zyh/PycharmProjects/baidu_dog/crop_test_img', image_size, shuffle=False,
                                             batch_size=1, class_mode=None)
    #train = fc_layer.predict_generator(train_generator, train_generator.n / train_generator.batch_size)
    #val = fc_layer.predict_generator(val_generator, val_generator.n / val_generator.batch_size)
    test = fc_layer.predict_generator(test_generator, test_generator.n / test_generator.batch_size)
    with h5py.File("gap_test%s.h5"%MODEL.func_name) as h:
        #h.create_dataset("train", data=train)
        #h.create_dataset("val", data=val)
        h.create_dataset("test", data=test)
        h.create_dataset("train_label", data=train_generator.classes)
        h.create_dataset("val_label", data=val_generator.classes)
        h.create_dataset("test_name", data=test_generator.filenames)

write_gap(ResNet50, (224, 224), '/home/zyh/PycharmProjects/baidu_dog/keras-dogs-master/single/resnet50_model/frozen_88/resnet50-weights-26-0.742.h5')
write_gap(Xception, (299, 299), '/home/zyh/PycharmProjects/baidu_dog/keras-dogs-master/single/xception_model/frozen_66/xception-tuned02-0.78.h5')
write_gap(VGG16, (224, 224), '/home/zyh/PycharmProjects/baidu_dog/keras-dogs-master/single/vgg16_model/frozen_16/vgg16-weights-21-0.70.h5')