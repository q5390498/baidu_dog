import  os

train_aug_txt = open("train_aug_eq.txt", 'w')
val_aug_txt = open("val_eq.txt", 'w')
test_eq_txt = open('test_eq.txt', 'w')

for r, s, files in os.walk('/home/zyh/PycharmProjects/baidu_dog/crop_equal/crop_train_aug_2'):
    for f in files:
        file_name = r + '/' + f
        label = r.split('/')[-1].split('_')[0]
        train_aug_txt.write(file_name + ' ' + label + '\n')

for r, s, files in os.walk('/home/zyh/PycharmProjects/baidu_dog/crop_equal/crop_val'):
    for f in files:
        file_name = r + '/' + f
        label = r.split('/')[-1].split('_')[0]
        val_aug_txt.write(file_name + ' ' + label + '\n')

for r, s, files in os.walk('/home/zyh/PycharmProjects/baidu_dog/crop_equal/crop_test_img/image'):
    for f in files:
        file_name = r + '/' + f
        label = r.split('/')[-1].split('_')[0]
        test_eq_txt.write(file_name+'\n')