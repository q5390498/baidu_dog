import numpy as np
import os
import shutil

train_file = open('train.txt', 'w')
val_file = open('val.txt', 'w')

label_train_dict = {}
label_val_dict = {}
label_train_file = open('./traindata/data_train_image.txt').readlines()
label_val_file = open('./traindata/val.txt').readlines()

labels = []
roots = '/home/zyh/PycharmProjects/baidu_dog/traindata/train/'
for line in label_train_file:
    filename = line.split(' ')[0].split('.')[0]
    label = line.split(' ')[1]
    labels.append(label)
    label_train_dict[filename] = label

for line in label_val_file:
    filename = line.split(' ')[0].split('.')[0]
    label = line.split(' ')[1]
    label_val_dict[filename] = label

labels = set(labels)
labels_map = {}
labels_map_file = open('label_to_newLabel.txt', 'w')
newLabels_map_file = open('newLabel_to_label.txt', 'w')
k=0
for i in labels:
    labels_map[i] = k
    labels_map_file.write(i + ' ' + str(k) + '\n')
    newLabels_map_file.write(str(k) + ' ' + i + '\n')
    k+=1

print labels_map
for root, sub, files in os.walk('./traindata/train'):
    for file in files:
        filename = file.split('.')[0]
        #print filename
        label = labels_map[label_train_dict[filename]]
        if os.path.exists('/home/zyh/PycharmProjects/baidu_dog/all_data/'+str(label) + '_' + str(label_train_dict[filename])) == False:
            os.mkdir('/home/zyh/PycharmProjects/baidu_dog/all_data/'+str(label) + '_' + str(label_train_dict[filename]))
            #print 111
        new_filename = roots + file
        #shutil.copyfile(new_filename, '/home/zyh/PycharmProjects/baidu_dog/all_data/'+str(label) + '_' + str(label_train_dict[filename])+'/'+filename+'.jpg')
        #break
        #train_file.write(new_filename + ' ' +str(label) + '\n')

error = 0
roots = '/home/zyh/PycharmProjects/baidu_dog/traindata/test1/'
print len(label_val_dict)
for root, sub, files in os.walk('./traindata/test1'):
    for file in files:
        filename = file.split('.')[0]
        try:
            new_filename = roots + file
            if os.path.exists(new_filename) == False:
                continue
            label = labels_map[label_val_dict[filename]]

            if os.path.exists('/home/zyh/PycharmProjects/baidu_dog/val_data/' + str(label) + '_' + str(label_val_dict[filename])) == False:
                os.mkdir('/home/zyh/PycharmProjects/baidu_dog/val_data/' + str(label) + '_' + str(label_val_dict[filename]))
            shutil.copyfile(new_filename, '/home/zyh/PycharmProjects/baidu_dog/val_data/' + str(label) + '_' + str(label_val_dict[filename])+'/'+filename+'.jpg')
            # import random
            # r = random.randint(0,10)
            # if()
            # val_file.write(new_filename + ' ' + str(label) + '\n')
        except:
            print filename
            error += 1

print error

all_data = []
for root,sub,files in os.walk('/home/zyh/PycharmProjects/baidu_dog/all_data'):
    for file in files:
        newfilename = root + '/' + file
        label = newfilename.split('/')[-2].split('_')[0]
        data = newfilename + ' ' + label
        all_data.append(data)

