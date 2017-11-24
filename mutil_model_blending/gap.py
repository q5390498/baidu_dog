from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *
from keras.preprocessing.image import load_img
import argparse
import h5py

parser = argparse.ArgumentParser()
parser.add_argument('--imgtxt', required=True, help='img_file')
parser.add_argument('--ffpath', required=True, help='ffpath')
opt = parser.parse_args()

def my_generator(image_txt, batchsize):
    idx = 0
    for idx in range(0, len(image_txt), batchsize):
        imgs = []
        if idx + batchsize < len(image_txt):
            for i in xrange(idx, idx + batchsize):
                img = load_img(image_txt[i].strip('\n').split(' ')[0])
                imgs.append(img)
            yield imgs
        else:
            for i in xrange(idx, len(image_txt)):
                img = load_img(image_txt[i].strip('\n').split(' ')[0])
                imgs.append(img)
            yield imgs

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
    # train_gen = ImageDataGenerator(rescale=1./255)
    # test_gen = ImageDataGenerator(rescale=1./255)
    # train_generator = train_gen.flow_from_directory('/home/zyh/PycharmProjects/baidu_dog/new_crop_train_augment', image_size, shuffle=False,
    #                                           batch_size=100)
    # val_generator = test_gen.flow_from_directory('/home/zyh/PycharmProjects/baidu_dog/crop_val', image_size, shuffle=False,
    #                                          batch_size=100, class_mode=None)
    # test_generator = test_gen.flow_from_directory('/home/zyh/PycharmProjects/baidu_dog/crop_test_img', image_size, shuffle=False,
    #                                          batch_size=100, class_mode=None)
    # train = fc_layer.predict_generator(train_generator, train_generator.n / train_generator.batch_size)
    #val = fc_layer.predict_generator(val_generator, val_generator.n / val_generator.batch_size)
    #test = fc_layer.predict_generator(test_generator, test_generator.n / test_generator.batch_size)
    # train = []
    # train_image_txt = open('/home/zyh/PycharmProjects/baidu_dog/Baidu-Dog2017-master/train_mu.txt').readlines()
    # for line in train_image_txt:
    #     #print 1
    #     image_file = line.strip('\n').split(' ')[0]
    #     img = load_img(image_file)
    #     img = img.resize([299, 299])
    #     img = img_to_array(img)
    #     img = 1. / 255 * img
    #     img = np.expand_dims(img, axis=0)
    #     feat = fc_layer.predict(img, batch_size=1)
    #     #print np.shape(feat)
    #     train.append(feat)

    val = []
    img_txt_path = opt.imgtxt
    val_image_txt = open(img_txt_path).readlines()
    for line in val_image_txt:
        image_file = line.strip('\n').split(' ')[0]
        img = load_img(image_file)
        img = img.resize([299, 299])
        img = img_to_array(img)
        img = 1. / 255 * img
        img = np.expand_dims(img, axis=0)
        feat = fc_layer.predict(img)
        val.append(feat)
    val = np.concatenate(val, axis=0)
    print np.shape(val)
    # test = []
    # test_image_txt = open('/home/zyh/PycharmProjects/baidu_dog/Baidu-Dog2017-master/test.txt').readlines()
    # for line in test_image_txt:
    #     image_file = line.strip('\n').split(' ')[0]
    #     img = load_img(image_file)
    #     img = img.resize([299, 299])
    #     img = img_to_array(img)
    #     img = 1. / 255 * img
    #     img = np.expand_dims(img, axis=0)
    #     feat = fc_layer.predict(img)
    #     test.append(feat)
    file = opt.ffpath
    with h5py.File(file) as h:
        #h.create_dataset("train_feature", data=train)
        h.create_dataset('test_feature', data=val)
        #h.create_dataset("val", data=val)
        #h.create_dataset("test", data=test)
        #h.create_dataset("test_feature", data=test)
        #h.create_dataset("val_label", data=val_generator.classes)
        #h.create_dataset("test_name", data=test_generator.filenames)

#write_gap(ResNet50, (224, 224), '/home/zyh/PycharmProjects/baidu_dog/keras-dogs-master/single/resnet50_model/frozen_88/resnet50-weights-26-0.742.h5')
write_gap(Xception, (299, 299), '/home/zyh/PycharmProjects/baidu_dog/keras-dogs-master/single/xception_model/frozen_66/xception-tuned02-0.78.h5')
#write_gap(VGG16, (224, 224), '/home/zyh/PycharmProjects/baidu_dog/keras-dogs-master/single/vgg16_model/frozen_16/vgg16-weights-21-0.70.h5')