import tensorflow as tf
import pandas as pd
import pylab as P
import time
import datetime

def int64_feature(self, value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def bytes_feature(self, value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def fromDatetoTs(dataStr):
    '''
    输入格式类似："2015-03-10 22:51:41.0"
    最后带毫秒、微妙时间的
    返回的是UNIX时间戳
    '''
    return time.mktime(datetime.datetime.strptime(s, "%Y-%m-%d %H:%M:%S.%f").timetuple())

def toUserMap(df, id_key):
    '''
    以df中的id_key为key，将相同ID的数据放到map的同一个key中
    ret:
    dict: key:user_id, value:[df_row1, df_row2...]
    '''
    ret_map = dict()
    for i in range(0, len(df)):
        user_id = df[id_key][i]
        if(not user_id in ret_map):
            ret_map[user_id] = []
        ret_map[user_id].append(df[i:i+1])
    return ret_map

def createTfRecord():
    writer = tf.python_io.TFRecordWriter(tf_file)
    example = tf.train.Example(features=tf.train.Features(feature={
                    'height': self.int64_feature(image_height),
                    'width': self.int64_feature(image_width),
                    'label_length': self.int64_feature(len(code)),
                    'label_0':self.int64_feature(real_labels[0]),
                    'label_1':self.int64_feature(real_labels[1]),
                    'label_2':self.int64_feature(real_labels[2]),
                    'label_3':self.int64_feature(real_labels[3]),
                    'label_4':self.int64_feature(real_labels[4]),
                    'label_5':self.int64_feature(real_labels[5]),
                    'image_raw': self.bytes_feature(img_raw.tobytes())}))
    writer.write(example.SerializeToString())

def readData():
    pass

if __name__ == "__main__":
    print("Begin to run")
