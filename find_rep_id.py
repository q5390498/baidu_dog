import os

train_val_id = []
train_label = []
test_id = []
test_label = []

for r, s, files in os.walk('/home/zyh/PycharmProjects/baidu_dog/all_data'):
    for file in files:
        id = file.split('/')[-1].split('.')[0]
        label = r.split('/')[-1].split('_')[-1]
        train_val_id.append(id)
        train_label.append(label)
for r, s, files in os.walk('/home/zyh/PycharmProjects/baidu_dog/val_data'):
    for file in files:
        id = file.split('/')[-1].split('.')[0]
        label = r.split('/')[-1].split('_')[-1]
        train_val_id.append(label)
# train_txt = open('/home/zyh/PycharmProjects/baidu_dog/traindata/data_train_image.txt').readlines()
# val_txt = open('/home/zyh/PycharmProjects/baidu_dog/traindata/val.txt').readlines()
# for line in train_txt:
#     id = line.split(' ')[0]
#     label = line.split(' ')[1]
#     train_val_id.append(id)
#     train_label.append(label)
# for line in val_txt:
#     id = line.split(' ')[0]
#     label = line.split(' ')[1]
#     train_val_id.append(id)
#     train_label.append(label)


for r, s, files in os.walk('/home/zyh/PycharmProjects/baidu_dog/test2/image/'):
    for file in files:
        id = file.split('/')[-1].split('.')[0]
        test_id.append(id)
#print test_id
print len(train_val_id)
print len(test_id)
intersct = [t for t in test_id if t in train_val_id]
print len(intersct)
inters_id = []
inters_label = []
for i, tst_id in enumerate(test_id):
    for j, tr_id in enumerate(train_val_id):
        if tst_id == tr_id:
            inters_id.append(tr_id)
            inters_label.append(train_label[j])
            break
print inters_id
print inters_label
res_out = open("res_out.txt", 'w')
for i, id in enumerate(inters_id):
    res_out.write(str(id) + ' ' +str(inters_label[i])+ "\n")