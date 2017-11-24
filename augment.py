import os
from data_banlance import DataAugmentation
import cv2

for r, s, files in os.walk('/home/zyh/PycharmProjects/baidu_dog/mutil_crop_val_aug'):
    for f in files:
        filename = r + '/' + f
        num = 0
        new_name = r + '/' +f.split('.')[0] + '_' + str(num) + '.jpg'
        im = DataAugmentation().randomRotation(filename)
        im.save(new_name)

        num = 1
        new_name = r + '/' + f.split('.')[0] + '_' + str(num) + '.jpg'
        im = DataAugmentation().randomColor(filename)
        im.save(new_name)

        num = 2
        new_name = r + '/' + f.split('.')[0] + '_' + str(num) + '.jpg'
        im = DataAugmentation().randomGaussian(filename)
        im.save(new_name)

        num = 3
        new_name = r + '/' + f.split('.')[0] + '_' + str(num) + '.jpg'
        im = DataAugmentation().h_flip(filename)
        cv2.imwrite(new_name, im)

# for r, s, files in os.walk('/home/zyh/PycharmProjects/baidu_dog/crop_test_img/image'):
#     for f in files:
#         filename = r + '/' + f
#         new_dir = '/home/zyh/PycharmProjects/baidu_dog/test_img_aug/origin/'
#         new_name = new_dir + f
#         import shutil
#         shutil.copy(filename, new_dir + f)
#
#         new_dir = '/home/zyh/PycharmProjects/baidu_dog/test_img_aug/random_rotation/'
#         im = DataAugmentation().randomRotation(filename)
#         im.save(new_dir + f)
#
#         num = 1
#         new_dir = '/home/zyh/PycharmProjects/baidu_dog/test_img_aug/color/'
#         im = DataAugmentation().randomColor(filename)
#         im.save(new_dir + f)
#
#         num = 2
#         new_dir = '/home/zyh/PycharmProjects/baidu_dog/test_img_aug/gassian/'
#         im = DataAugmentation().randomGaussian(filename)
#         im.save(new_dir + f)
#
#         num = 3
#         new_dir = '/home/zyh/PycharmProjects/baidu_dog/test_img_aug/hflip/'
#         im = DataAugmentation().h_flip(filename)
#         cv2.imwrite(new_dir + f, im)