import numpy as np
import cv2
import sys,os

caffe_root = '/home/zyh/caffe/'
sys.path.insert(0, caffe_root + 'python')
import caffe
os.chdir(caffe_root)
caffe.set_device(0)
caffe.set_mode_gpu()

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--imgtxt', required=True, help='train txt')
parser.add_argument('--ffpath', required=True, help='ffpath')
opt = parser.parse_args()

dic = {'24': '51', '25': '50', '26': '53', '27': '52', '20': '129', '21': '59', '22': '54', '23': '57', '28': '115', '29': '114', '0': '133', '4': '26', '8': '22', '59': '65', '58': '64', '55': '61', '54': '60', '57': '63', '56': '62', '51': '36', '50': '37', '53': '34', '52': '35', '88': '5', '89': '9', '82': '45', '83': '42', '80': '46', '81': '47', '86': '41', '87': '1', '84': '43', '85': '40', '3': '25', '7': '21', '39': '85', '38': '84', '33': '83', '32': '82', '31': '111', '30': '88', '37': '87', '36': '86', '35': '81', '34': '80', '60': '66', '61': '67', '62': '68', '63': '69', '64': '2', '65': '6', '66': '95', '67': '94', '68': '97', '69': '11', '2': '24', '6': '20', '99': '78', '98': '79', '91': '76', '90': '77', '93': '74', '92': '75', '95': '72', '94': '73', '97': '70', '96': '71', '11': '29', '10': '28', '13': '4', '12': '0', '15': '120', '14': '8', '17': '126', '16': '123', '19': '128', '18': '127', '48': '31', '49': '30', '46': '33', '47': '32', '44': '39', '45': '38', '42': '109', '43': '101', '40': '3', '41': '7', '1': '132', '5': '27', '9': '23', '77': '18', '76': '19', '75': '16', '74': '17', '73': '14', '72': '12', '71': '13', '70': '10', '79': '49', '78': '48'}

net_file = '/home/zyh/PycharmProjects/baidu_dog/inceptionv4/deploy_inception-v4.prototxt'
caffe_model = '/home/zyh/PycharmProjects/baidu_dog/inceptionv4/inceptionv4_iter_20000.caffemodel'
mean_file = '/home/xudong/Demo/Data/Baidu/BigDataDog/test-01/mean.npy'

net = caffe.Net(net_file, caffe_model, caffe.TEST)
#transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
#transformer.set_transpose('data', (2, 0, 1))
#transformer.set_mean('data', np.load(mean_file).mean(1).mean(1))
#transformer.set_raw_scale('data', 255)
#transformer.set_channel_swap('data', (2,1,0))

train_file = open('/home/zyh/PycharmProjects/baidu_dog/Baidu-Dog2017-master/mu_val_aug.txt', 'r').readlines()

imagepath = '/home/xudong/Demo/Data/Baidu/BigDataDog/test-01/crop/image/'

#result = open('/home/xudong/Demo/Data/Baidu/BigDataDog/data/caffe/inception/inception-v4/test1/result/inception0809-001.txt', 'wa+')

# print 'train'
# train_feature = []
# for line in train_file:
# 	image_file = line.split(' ')[0]
# 	image = cv2.imread(image_file)
# 	reimage = cv2.resize(image, (395, 395))
# 	dst = np.asarray(reimage, dtype='float32')
# 	temp = dst
# 	#temp = np.transpose(dst, [1, 0, 2])
# 	#temp = (temp - 127.5) / 128
# 	temp[:, :, 0] = temp[:, :, 0] - 104.0
# 	temp[:, :, 1] = temp[:, :, 1] - 117.0
# 	temp[:, :, 2] = temp[:, :, 2] - 123.0
# 	#print temp[:, :, 0]
#
# 	temp = np.transpose(temp, [1, 0, 2])
# 	#cv2.imshow('temp', temp)
# 	#cv2.waitKey(0)
# 	temp1 = temp.copy()
# 	#temp1[:, :, 0], temp1[:, :, 1], temp1[:, :, 2] = temp[:, :, 2], temp[:, :, 1], temp[:, :, 0]
# 	#temp1 = cv2.flip(temp1, 0)
# 	#cv2.imshow('temp1', temp1)
# 	#cv2.waitKey(0)
#
# 	dogimage = np.swapaxes(temp1, 0, 2)
# 	net.blobs['data'].reshape(1, 3, 395, 395)
# 	net.blobs['data'].data[...] = dogimage
# 	out = net.forward()
#
# 	feat = net.blobs['fc7'].data[0].flatten()
# 	#print feat, np.shape(feat)
# 	train_feature.append(feat)
# 	#order = prob.argsort()[-1]
# 	#print order
# 	#result.write(str(dic[str(order)]) + '\t' + line.strip() + '\n')
# 	#result.write(str(order) + '\t' + line.strip() + '\n')
# 	#cv2.imshow('image', image)
# 	#cv2.waitKey(1)
print 'val'
image_txt = opt.imgtxt
val_txt = open(image_txt).readlines()
val_feature = []
for line in val_txt:
	image_file = line.split(' ')[0]
	image = cv2.imread(image_file)
	reimage = cv2.resize(image, (395, 395))
	dst = np.asarray(reimage, dtype='float32')
	temp = dst
	#temp = np.transpose(dst, [1, 0, 2])
	#temp = (temp - 127.5) / 128
	temp[:, :, 0] = temp[:, :, 0] - 104.0
	temp[:, :, 1] = temp[:, :, 1] - 117.0
	temp[:, :, 2] = temp[:, :, 2] - 123.0
	#print temp[:, :, 0]

	temp = np.transpose(temp, [1, 0, 2])

	temp1 = temp.copy()
	dogimage = np.swapaxes(temp1, 0, 2)
	net.blobs['data'].reshape(1, 3, 395, 395)
	net.blobs['data'].data[...] = dogimage
	out = net.forward()

	feat = net.blobs['fc7'].data[0].flatten()
	val_feature.append(feat)

# print 'test'
# test_txt = open('/home/zyh/PycharmProjects/baidu_dog/Baidu-Dog2017-master/test.txt').readlines()
# test_feature = []
# for line in test_txt:
# 	image_file = line.strip('\n')
# 	image = cv2.imread(image_file)
# 	reimage = cv2.resize(image, (395, 395))
# 	dst = np.asarray(reimage, dtype='float32')
# 	temp = dst
# 	#temp = np.transpose(dst, [1, 0, 2])
# 	#temp = (temp - 127.5) / 128
# 	temp[:, :, 0] = temp[:, :, 0] - 104.0
# 	temp[:, :, 1] = temp[:, :, 1] - 117.0
# 	temp[:, :, 2] = temp[:, :, 2] - 123.0
# 	#print temp[:, :, 0]
#
# 	temp = np.transpose(temp, [1, 0, 2])
# 	#cv2.imshow('temp', temp)
# 	#cv2.waitKey(0)
# 	temp1 = temp.copy()
# 	#temp1[:, :, 0], temp1[:, :, 1], temp1[:, :, 2] = temp[:, :, 2], temp[:, :, 1], temp[:, :, 0]
# 	#temp1 = cv2.flip(temp1, 0)
# 	#cv2.imshow('temp1', temp1)
# 	#cv2.waitKey(0)
#
# 	dogimage = np.swapaxes(temp1, 0, 2)
# 	net.blobs['data'].reshape(1, 3, 395, 395)
# 	net.blobs['data'].data[...] = dogimage
# 	out = net.forward()
#
# 	feat = net.blobs['fc7'].data[0].flatten()
# 	test_feature.append(feat)
print val_feature
import h5py
with h5py.File("/home/zyh/PycharmProjects/baidu_dog/inceptionv4/" + opt.ffpath, "w") as f:
	#f.create_dataset('train_feature', data=train_feature)
	f.create_dataset('test_feature', data=val_feature)
	#f.create_dataset('test_feature', data=test_feature)
	print 1