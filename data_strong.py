#coding=utf8
from keras.preprocessing.image import ImageDataGenerator,array_to_img,img_to_array,load_img
from PIL import Image, ImageEnhance, ImageOps, ImageFile
import numpy as np
import random
import numpy as np
import skimage
import cv2

origin_data_file = open('./traindata/data_train_image.txt').readlines()
origin_image_path = '/home/zyh/PycharmProjects/baidu_dog/traindata/train/'

class DataAugmentation:
    def __init__(self):
        image = ''
    def load_image(self, image_name):
        return Image.open(image_name)

    def randomRotation(self, image_name):
        image = load_img(image_name)
        image = image.resize((256, 256))
        random_angle = np.random.randint(1,360)
        return image.rotate(random_angle, Image.BICUBIC)

    def randomCrop(self,image_name):
        image = load_img(image_name)
        image = image.resize((256, 256))
        w = image.size[0]
        h = image.size[1]

        randomw = random.randint(0,32)
        randomh = random.randint(0,32)

        random_region = (randomw, randomh, randomw+224, randomh+224)
        return image.crop(random_region)

    def randomColor(self, image_name):
        """
        对图像进行颜色抖动
        :param image: PIL的图像image
        :return: 有颜色色差的图像image
         """
        image = load_img(image_name)
        image = image.resize((256, 256))
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因1子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        return ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度

    def randomGaussian(self,image_name, mean=0.2, sigma=0.3):
        def gaussianNoisy(im, mean=0.2, sigma=0.3):
            """
            对图像做高斯噪音处理
            :param im: 单通道图像
            :param mean: 偏移量
           :param sigma: 标准差
           :return:
           """
            for _i in range(len(im)):
                im[_i] += random.gauss(mean, sigma)
            return im

        image = load_img(image_name)
        image = image.resize((256, 256))
        width, height = image.size[0], image.size[1]
        img = np.asarray(image)
        img.flags.writeable = True  # 将数组改为读写模式
        img_r = gaussianNoisy(img[:, :, 0].flatten(), mean, sigma)
        img_g = gaussianNoisy(img[:, :, 1].flatten(), mean, sigma)
        img_b = gaussianNoisy(img[:, :, 2].flatten(), mean, sigma)
        img[:, :, 0] = img_r.reshape([width, height])
        img[:, :, 1] = img_g.reshape([width, height])
        img[:, :, 2] = img_b.reshape([width, height])
        return Image.fromarray(np.uint8(img))

    def h_flip(self, image_name):
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (256,256))
        image = cv2.flip(image, 1)
        #return Image.fromarray(image)
        return image

    def v_flip(self, image_name):
        image = cv2.imread(image_name, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (256, 256))
        image = cv2.flip(image, 0)
        #return Image.fromarray(image)
        return image

def precessing(img):
    pass


import os
for root,sub,files in os.walk('/home/zyh/PycharmProjects/baidu_dog/crop_train'):
    for file in files:
        num = 0
        image_name = root + '/' + file
        #lable = line.split(' ')[1]
        print image_name
        image = DataAugmentation().load_image(image_name)
        image = image.resize((224,224))
        #image.show()

        image = DataAugmentation().randomRotation(image_name)
        new_file_name = file.split('.')[0] + '_' + str(num) + '.jpg'
        image.save(root +'/'+file.split('.')[0]+'_'+str(num)+'.jpg')
        num+=1

        image = DataAugmentation().randomCrop(image_name)
        new_file_name = file.split('.')[0] + '_' + str(num) + '.jpg'
        image.save(root +'/' +file.split('.')[0]+ str(num) + '.jpg')
        num += 1

        image = DataAugmentation().randomColor(image_name)
        new_file_name = file.split('.')[0] + '_' + str(num) + '.jpg'
        image.save(root +'/'+file.split('.')[0] + str(num) + '.jpg')
        num += 1

        image = DataAugmentation().randomGaussian(image_name)
        new_file_name = file.split('.')[0] + '_' + str(num) + '.jpg'
        image.save(root +'/'+file.split('.')[0] + str(num) + '.jpg')
        num += 1
        for i in xrange(3):
            image = DataAugmentation().randomCrop(image_name)
            image.save(root +'/'+file.split('.')[0] + str(num) + '.jpg')
            num += 1

        image = DataAugmentation().h_flip(image_name)
        cv2.imwrite(root +'/' +file.split('.')[0]+ str(num) + '.jpg', image)
        num+=1

        image = DataAugmentation().v_flip(image_name)
        cv2.imwrite(root +'/'+file.split('.')[0] + str(num) + '.jpg', image)
        num += 1

        #image.show()
        #break

    #break
