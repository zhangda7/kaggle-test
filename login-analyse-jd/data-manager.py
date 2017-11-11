import os
import tensorflow as tf
import numpy as np
import pandas as pd
import pylab as P
import time
import datetime

"""
TF record manager
"""
IMAGE_WIDTH = 60
IMAGE_HEIGHT = 22
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
NUM_CLASSES=26 + 26 + 10 + 1
SPACE_INDEX=0
SPACE_TOKEN=''
MAX_PREDICT_LENGTH = 7
MAX_TRAIN_IMG_NUM = 5000
MAX_EVAL_IMG_NUM = 200
charset = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
encode_maps = {}
decode_maps = {}
for i, char in enumerate(charset, 1):
    encode_maps[char] = i
    decode_maps[i] = char
encode_maps[SPACE_TOKEN]=SPACE_INDEX
decode_maps[SPACE_INDEX]=SPACE_TOKEN

class TFRecordManager():
    def __init__(self):
        pass

    def fromDatetoTs(self, dataStr):
        '''
        输入格式类似："2015-03-10 22:51:41.0"
        最后带毫秒、微妙时间的
        返回的是UNIX时间戳
        '''
        return time.mktime(datetime.datetime.strptime(dataStr, "%Y-%m-%d %H:%M:%S.%f").timetuple())

    def toUserMap(self, df, id_key, sortKey = None):
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
        if(sortKey != None):
            for key in ret_map:
                ret_map[key].sort(key=lambda x:x[sortKey].values[0])
        return ret_map

    def concatLoginTrade(self, login_map, trade_map, kept_login = 1):
        '''
        按照时间规则，将login_map和trade_map结合起来
        kept_login 是根据当前的trade记录，要保留前面几个login的行为
        规则：
        1.根据每个ID进行遍历
        2.根据时间戳进行升序排序
        3.遍历trade_map，当有一个trade的时间晚于login_map中的最后一次时间时，则将此次trade绑定到对应的login上
        4.对于无法找到login的trade，先填写默认值吧
        '''
        login_trade = []
        for user_id in trade_map:
            trade_data = trade_map[user_id]
            if(not user_id in login_map):
                #print("WARN, user id {} not in login map".format(user_id))
                continue
            login_data = login_map[user_id]
            login_index = 0
            trade_index = 0
            for i in range(0, len(trade_data)):
                one_trade = trade_data[i]
                trade_ts = self.fromDatetoTs(one_trade["time"].values[0])
                login_sum = 0
                #reverse search
                for j in range(len(login_data) - 1, -1, -1):
                    #print("ret : {}".format(login_data[j]["timestamp"].values[0] - trade_ts))
                    if(login_data[j]["timestamp"].values[0] < trade_ts):
                        #insert
                        login_trade.append((login_data[j], one_trade))
                        #print("Find One")
                        login_sum = login_sum + 1
                        break
                if(login_sum == 0):
                    print("WARN, can not find login for trade {} {} {}".format(one_trade["rowkey"].values, one_trade["time"].values, one_trade["id"].values))
                    pass
        print("Finish")
        return login_trade

    def int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def createTf(self, MAX_IMG_NUM = 10):
        print("Begin create tf")
        train_login_df = pd.read_csv('E://Projects//python//kaggle-test//login-analyse-jd//data//t_login-small.csv', header=0)
        train_trade_df = pd.read_csv('E://Projects//python//kaggle-test//login-analyse-jd//data/t_trade-small.csv', header=0)
        login_map = self.toUserMap(train_login_df, "id", sortKey = "timestamp")
        trade_map = self.toUserMap(train_trade_df, "id")
        login_trade = self.concatLoginTrade(login_map, trade_map)
        self.createTf(login_trade, "tf")

    def createTf(self, login_trade, tf_file, MAX_IMG_NUM = 10):
        print("Begin create tf")
        writer = tf.python_io.TFRecordWriter(tf_file)
        for i in range(0, range(login_trade)):
            login_df = login_trade[i][0]
            trade_df = login_trade[i][1]

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
            if(i >= MAX_IMG_NUM):
                break
            i += 1

    def readTf_simple(self, tf_file):
        images = []
        labels_0 = []
        index = 1
        for serialsed_example in tf.python_io.tf_record_iterator(tf_file):
            example = tf.train.Example()
            example.ParseFromString(serialsed_example)
            
            images.append(image)
            labels_0.append(label_info["label_0"])
            if(index >= 128):
                break
            index += 1
        return np.array(images), np.array(labels_0).reshape(len(labels_0))

def test_tf():
    tfManager = TFRecordManager()
    img, label = tfManager.read_tf_queue("/tmp/test/tf_eval1")
    #img_batch, label_batch = tf.train.shuffle_batch([img, label], batch_size=1, capacity=2, min_after_dequeue=2)
    init = tf.initialize_all_variables()
    with tf.Session() as sess:
        sess.run(init)
        threads = tf.train.start_queue_runners(sess=sess)
        for i in range(100):
            val, l= sess.run([img, label])
            #我们也可以根据需要对val， l进行处理
            #l = to_categorical(l, 12) 
            #val = val.reshape(60, 22)
            #cv2_show_img(val)
            #print(type(val), val.shape, l)
            print(l)

#%%
if(__name__ == "__main__"):
    print("Test Start...")
    #test_tf()
    tfManager = TFRecordManager()
    tfManager.createTf()
    #tfManager.createTf("E://Projects//python//lstm_ctc_ocr//train", "/tmp/test/tf_train1", 60, 22, MAX_IMG_NUM = MAX_TRAIN_IMG_NUM)    
    # tfManager.createTf("E://Projects//python//lstm_ctc_ocr//val", "/tmp/test/tf_eval1", 60, 22, MAX_IMG_NUM = MAX_EVAL_IMG_NUM)
    #tfManager.readTf_simple("/tmp/test/tf1")
    #tfManager.read_tf_queue("/tmp/test/tf1")