import caffe
import cv2
import numpy as np
import time


caffe.set_mode_gpu()

def RGB2BlackWhite(img):
    im = img
    #print "image info,", im.format, im.mode, im.size
    im = cv2.resize(im,(500,500))
    (w,h, c) = np.shape(im)
    #print (c,w,h)
    R = 0
    G = 0
    B = 0

    for x in xrange(w):
        for y in xrange(h):
            pos = (x, y)
            rgb = im[x,y]
            (r, g, b) = rgb
            R = R + r
            G = G + g
            B = B + b

    rate1 = R * 1000 / (R + G + B)
    rate2 = G * 1000 / (R + G + B)
    rate3 = B * 1000 / (R + G + B)

    #print "rate:", rate1, rate2, rate3

    for x in xrange(w):
        for y in xrange(h):
            pos = (x, y)
            rgb = im[x,y]
            (r, g, b) = rgb
            n = r * rate1 / 1000 + g * rate2 / 1000 + b * rate3 / 1000
            # print "n:",n
            if n <= 80:
                #im.putpixel(pos, (255, 255, 255))
                im[pos] = (255,255,255)
            else:
                im[pos] = (0, 0, 0)
    #cv2.imshow('1',im)
    #cv2.waitKey()
    return im

def transform_img(img, img_width=114, img_height=114):

    #Histogram Equalization
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    img[:, :, 1] = cv2.equalizeHist(img[:, :, 1])
    img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])
    return img

def get_img_info():
    file_path = r'/home/zyh/PycharmProjects/mnist/test.txt'
    files = open(file_path).readlines()
    return files


# net = caffe.Net('/home/zyh/PycharmProjects/baidu_dog/caffe/finetune_from_bvlc/deploy.prototxt',
#                 '/home/zyh/PycharmProjects/baidu_dog/caffe/finetune_from_bvlc/googlenet__iter_50000.caffemodel',
#                 caffe.TEST)
net = caffe.Net('/home/zyh/PycharmProjects/baidu_dog/caffe/Resnet50/ResNet-50-val.prototxt',
                '/home/zyh/PycharmProjects/baidu_dog/caffe/Resnet50/re_iter_12000.caffemodel',
                caffe.TEST)

#Define image transformers
from caffe.proto import caffe_pb2
mean_blob = caffe_pb2.BlobProto()
with open('/home/zyh/PycharmProjects/baidu_dog/caffe/mean.binaryproto') as f:
    mean_blob.ParseFromString(f.read())

mean_array = np.asarray(mean_blob.data, dtype=np.float32).reshape(mean_blob.channels, mean_blob.height, mean_blob.width)
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', mean_array)
transformer.set_transpose('data', (2,0,1))

file_path = r'/home/zyh/PycharmProjects/mnist/noise_train.txt'
files = get_img_info()
import glob
files = open('./val_crop.txt').readlines()#glob.glob('/home/zyh/PycharmProjects/mnist/t10k-images/'+'*.bmp')
print files

hit_per_class = [0] * 100
count_per_class = [0] * 100
count = 0
hit = 0
start = time.time()
for file in files:
    #print count
    #if count > 1000:
     #   break
    #filename = file.split(' ')[0]
    #filename = '/home/zyh/PycharmProjects/mnist/t10k-images/' + file
    #print filename
    label = int(file.split(' ')[-1])

    file = file.split(' ')[0]
    img = cv2.imread(file,cv2.IMREAD_COLOR)
    #img = RGB2BlackWhite(img)
    #cv2.imwrite(file,img)
    #cv2.imshow("11",img)
    #cv2.waitKey()
    try:
        img = cv2.resize(img,(224, 224))
    except:
        print file
        continue
        #break
    #print img
    #img = transform_img(img)
    #print img
    #cv2.imshow('1',img)
    #img = [img]
    net.blobs['data'].data[...] = transformer.preprocess('data',img)

    out = net.forward()
    pred_probas = out['prob']

    if count >= 0 and count < 10:
        #cv2.imshow('1',img)
        #cv2.waitKey()
        #print (pred_probas)
        #break
        pass
    pred = pred_probas.argmax()
    print 'true',label," pred:",pred
    if pred == label:
        hit += 1
        hit_per_class[int(label)] += 1
    else:
        pass
        #cv2.imwrite('./wrongimg/'+str(pred)+'_'+str(count)+'.jpg',img)
    count += 1
    count_per_class[int(label)] += 1
end = time.time()
print hit_per_class
print count_per_class
print 1.0*np.array(hit_per_class) / np.array(count_per_class)
result_file = open('./result.txt', 'w')
for i in xrange(100):
    result_file.write(str(i) + ' ' + str(hit_per_class[i]) + ' ' + str(count_per_class[i]) + ' ' +str(1.*hit_per_class[i]/count_per_class[i]) + '\n')

print 'hit',hit,' count',count,1.0*hit/count
print end - start
