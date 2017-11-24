import os
import cv2
import numpy as np
import shutil

def get_img_bbox_info(bbox_txt, state = 'TRAIN'):
    bbox_data = open(bbox_txt).readlines()
    bbox_dict = {}
    for line in bbox_data:
        info = line.strip('\n').split(' ')
        index = 0
        id = info[index]
        index += 1
        if(state != 'TEST'):
            index += 1
        #label = info[1]
        bboxes = []
        for i in xrange((len(info) - index) / 5):
            bbox = []
            bbox.append(float(info[index]))
            index += 1
            bbox.append(float(info[index]))
            index += 1
            bbox.append(float(info[index]))
            index += 1
            bbox.append(float(info[index]))
            index += 1
            bbox.append(float(info[index]))
            index += 1
            bboxes.append(bbox)
        bbox_dict[id] = bboxes
    return bbox_dict
#print train_bbox_dict

def crop_wh(img_w, img_h, bbox):
    xmin, ymin, xmax, ymax = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
    bbox_w = int(bbox[2]) - bbox[0]
    bbox_h = int(bbox[3]) - bbox[1]
    center = (xmin + (xmax - xmin) / 2, ymin + (ymax - ymin) / 2)
    new_size = max(bbox_h, bbox_w)
    new_size = min(min(img_h, img_w), new_size)
    xmin, ymin, xmax, ymax = center[0] - new_size / 2, center[1] - new_size / 2, center[0] + new_size / 2, center[1] + new_size / 2,
    xmin, ymin, xmax, ymax = xmin - 20, ymin - 20, xmax + 20, ymax + 20,
    xmin, ymin, xmax, ymax = max(0, xmin), max(0, ymin), min(img_w, xmax), min(img_h, ymax)
    return int(xmin), int(ymin), int(xmax), int(ymax)

def precessing(bbox_txt, original_image_dir, new_dir, state = 'TRAIN'):
    """
    my directory is form as :
    root:
        |-- 0
        |-- 1
        .
        .
        |-- 99
    :param bbox_txt  :
    :param original_image_dir, original image path, please use absolute path:
    :param new_dir,  new image path, please use absolute path:
    :return:
    """
    bbox_dict = get_img_bbox_info(bbox_txt, 'TRAIN')
    for path, subdir, files in os.walk(original_image_dir):
        error = 0
        for file in files:
            id = file.split('.')[0]
            image_name = path + '/' + file
            img = cv2.imread(image_name, cv2.IMREAD_COLOR)
            #cv2.imshow('origent', img)
            bboxes = bbox_dict[id]
            if len(bbox_dict) == 0:
                error += 1
            else:
                pass
                #print id
            h, w, c = np.shape(img)
            for i, bbox in enumerate(bboxes):
                prob = float(bbox[-1])
                if state == 'TRAIN' and prob < 0.7:
                    continue
                xmin, ymin, xmax, ymax = crop_wh(w, h, bbox)
                new_im = img[ymin : ymax, xmin : xmax]
                #cv2.imshow("crop" + str(i), new_im)
                new_path = new_dir + '/'
                new_sub_dir = path.split('/')[-1]
                new_path = new_path + new_sub_dir
                if(False == os.path.exists(new_path)):
                    os.mkdir(new_path)
                new_image_name = ''
                if state != 'TEST':
                    new_image_name = new_path + '/' + id + '_' + str(i) + '.jpg'
                else:
                    new_image_name = new_path + '/' + file

                cv2.imwrite(new_image_name, new_im)
                if(state == 'TEST'):
                    break
        print error
        #cv2.waitKey(1000)

if __name__ == '__main__':
    '''precessing('/home/zyh/PycharmProjects/baidu_dog/trainresult.txt',
               '/home/zyh/PycharmProjects/baidu_dog/all_data',
               '/home/zyh/PycharmProjects/baidu_dog/crop_train',
               'TRAIN')'''

    '''precessing('/home/zyh/PycharmProjects/baidu_dog/testresult.txt',
               '/home/zyh/PycharmProjects/baidu_dog/image',
               '/home/zyh/PycharmProjects/baidu_dog/crop_test_img',
               'TEST')'''

    precessing('/home/zyh/PycharmProjects/baidu_dog/valresult.txt',
               '/home/zyh/PycharmProjects/baidu_dog/val_data',
               '/home/zyh/PycharmProjects/baidu_dog/crop_val',
               'TEST')