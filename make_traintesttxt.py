import os
import shutil

for r, s, files in os.walk('./crop_val'):
    for file in files:
        src = r + '/' + file
        sub_dir = r.split('/')[-1]
        #print src, './crop_train/' + sub_dir + '/' + file
        #shutil.copy(src, './crop_train/' + sub_dir + '/' + file)

train_txt = open('./train_mu.txt', 'w')
test_txt = open('./val_mu.txt', 'w')
# for r, s, files in os.walk('/home/zyh/PycharmProjects/baidu_dog/mutil_crop_train'):
#     for f in files:
#         file_name = r + '/' + f
#         label = r.split('/')[-1].split('_')[0]
#         train_txt.write(file_name + ' ' + label + '\n')
#
# for r, s, files in os.walk('/home/zyh/PycharmProjects/baidu_dog/mutil_crop_val'):
#     for f in files:
#         file_name = r + '/' + f
#         label = r.split('/')[-1].split('_')[0]
#         test_txt.write(file_name + ' ' + label + '\n')

val_aug = open('random_ro_mu_val_aug.txt', 'w')
for r, s, files in os.walk('/home/zyh/PycharmProjects/baidu_dog/test_img_aug/random_rotation'):
    for f in files:
        file_name = r + '/' + f
        label = r.split('/')[-1].split('_')[0]
        val_aug.write(file_name + ' ' + label + '\n')
