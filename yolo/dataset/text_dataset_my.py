from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import random
import cv2
#from PIL import Image, ImageDraw
import numpy as np
from multiprocessing import Queue
#from Queue import Queue
from threading import Thread

from yolo.dataset.dataset import DataSet

class TextDataset(DataSet):
    """数据集的读取类别"""

    def __init__(self, common_params, dataset_params):
        self.data_path = str(dataset_params['path'])
        self.batch_size = int(common_params['batch_size'])
        self.width = int(common_params['image_size'])
        self.height = int(common_params['image_size'])
        self.thread_num = int(dataset_params['thread_num'])
        self.max_objects = int(common_params['max_objects_per_image'])


        #创建队列
        self.record_queue = Queue(maxsize=1000)
        self.image_label_queue = Queue(maxsize=512)

        self.record_list = []
        input_file = open(self.data_path, 'r')

        for line in input_file:
            line = line.strip()
            ss = line.split(' ')
            #将数字转换为浮点型
            ss[1:] = [float(num) for num in ss[1:]]
            self.record_list.append(ss)
        #record:['path', position1,2,3,4]
        #print(self.record_list[0])

        self.record_point = 0
        self.record_numbet = len(self.record_list)

        self.num_batch_per_epoch = int (self.record_numbet / self.batch_size)
        #开启线程调用函数，读取数据
        t_record_producter = Thread(target=self.record_producter)
        #设置所有线程一起结束
        t_record_producter.daemon = True
        t_record_producter.start()
        #print(t_record_producter)

        for i in range(self.thread_num):
            t = Thread(target=self.record_customer)
            t.daemon = True
            t.start()

    def record_customer(self):
        #队列的操作，从文件队列中取数据信息，处理信息之后放入label队列
        while True:
            #从文件队列里面读取记录
            item = self.record_queue.get()
            #print(item)
            #根据读取的记录，读取图像和label
            out = self.record_propross(item)
            #print(out)
            #添加进图像队列
            self.image_label_queue.put(out)

    def record_producter(self):
        """队列处理,图像列表随机生成"""
        while True:
            #每循环列表一次随机生成一次列表
            if self.record_point % self.record_numbet == 0:
                random.shuffle(self.record_list)
                self.record_point = 0
            self.record_queue.put(self.record_list[self.record_point])
            self.record_point += 1

    def record_propross(self, record):
        """数据读取与处理， record:image-path， label ， class_num
        return：image:
        labels:
        object_num:
        """
        # image = Image.open(record[0])
        # #print(image.mode)
        #
        # image = image.convert("RGB")
        # #OpenCV stores color image in BGR format.
        # # So, the converted PIL image is also in BGR-format.
        # # The standard PIL image is stored in RGB format.
        # h = image.size[0]
        # w = image.size[1]

        image = cv2.imread(record[0])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h = image.shape[0]
        w = image.shape[1]
        #得出宽高比，计算相应比例的gt
        height_rate = self.height * 1.0 / h
        width_rate = self.width * 1.0 / w

        #image = image.resize((self.height, self.width), Image.ANTIALIAS)
        image = cv2.resize(image, (self.height, self.width))
        #print(type(image))
        #image = np.asarray(image, dtype=np.float32)

        labels = [[0, 0, 0, 0, 0]] * self.max_objects

        i = 1
        object_num = 0
        while i < len(record):
            #读取坐标数据
            xmin = record[i]
            ymin = record[i + 1]
            xmax = record[i + 2]
            ymax = record[i + 3]
            class_num = record[i + 4]

            xcenter = (xmin + xmax) * 1.0/2 * width_rate
            ycenter = (ymin + ymax) * 1.0/2 * height_rate

            box_w = (xmax - xmin)  * width_rate
            box_h = (ymax - ymin)  * height_rate

            labels[object_num] = [xcenter, ycenter, box_w, box_h, class_num]
            object_num += 1
            i += 5

            if object_num > self.max_object:
                break
        #print(type(labels))
        return [image, labels, object_num]

    def batch(self):
        """get batch
        Returns:
          images: 4-D ndarray [batch_size, height, width, 3]
          labels: 3-D ndarray [batch_size, max_objects, 5]
          objects_num: 1-D ndarray [batch_size]
        """
        images = []
        labels = []
        objects_num = []
        for i in range(self.batch_size):
            image, label, object_num = self.image_label_queue.get()
            # print(image, label, object_num)
            images.append(image)
            labels.append(label)
            objects_num.append(object_num)
        #print(labels)
        #print(type(labels))
        images = np.asarray(images, dtype=np.float32)
        images = images / 255 * 2 - 1
        labels = np.asarray(labels, dtype=np.float32)
        objects_num = np.asarray(objects_num, dtype=np.int32)
        return images, labels, objects_num