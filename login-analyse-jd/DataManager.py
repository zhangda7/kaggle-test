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

MAX_RECORD_NUM = 1000000

class TFRecordManager():
    def __init__(self):
        pass

    def int64_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    def int64_list_feature(self, value):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    def bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

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
            if(not (user_id in ret_map)):
                ret_map[user_id] = []
            ret_map[user_id].append(df[i:i + 1])
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
        matched = 0
        notMatched = 0
        trade_all_sum = 0
        for user_id in trade_map:
            trade_data = trade_map[user_id]
            trade_all_sum += len(trade_data)
            if(not user_id in login_map):
                #print("WARN, user id {} not in login map".format(user_id))
                #append None login info
                for i in range(0, len(trade_data)):
                    one_trade = trade_data[i]
                    login_trade.append((None, one_trade))
                    notMatched = notMatched + 1
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
                    if(login_data[j]["timestamp"].values[0] < trade_ts):
                        #insert
                        login_trade.append((login_data[j], one_trade))
                        #print("Find One")
                        login_sum = login_sum + 1
                        matched = matched + 1
                        break
                if(login_sum == 0):
                    #print("WARN, can not find login for trade {} {} {}".format(one_trade["rowkey"].values, one_trade["time"].values, one_trade["id"].values))
                    login_trade.append((None, one_trade))
                    notMatched = notMatched + 1
                    pass
        print("Finish concat, total {}, matched {}, not matched {}".format(trade_all_sum, matched, notMatched))
        return login_trade

    def oneHot(self, ll, num_classes=None):
        if(num_classes == None):
            num_classes = np.max(np.array(ll)) + 1
        return np.eye(num_classes, dtype=int)[np.array(ll)]

    def transFormat(self, login_df, trade_df):
        #train data: timelong,device,log_from,ip,city,result,timestamp,type,id,is_scan,is_sec,trade_ts
        one_row = np.zeros([12], dtype="int64")
        if(login_df is not None):
            one_row[0] = int(login_df["timelong"].values[0])
            one_row[1] = login_df["device"].values[0]
            one_row[2] = login_df["log_from"].values[0]
            one_row[3] = login_df["ip"].values[0]
            one_row[4] = login_df["city"].values[0]
            one_row[5] = login_df["result"].values[0]
            one_row[6] = login_df["timestamp"].values[0]
            one_row[7] = login_df["type"].values[0]
            one_row[8] = login_df["id"].values[0]
            one_row[9] = login_df["is_scan"].values[0]
            one_row[10] = login_df["is_sec"].values[0]
            one_row[11] = self.fromDatetoTs(trade_df["time"].values[0])
        #is_risk
        label = self.oneHot(trade_df["is_risk"].values[0], 2)

        return one_row, label

    def readCsvCreateTf(self, MAX_IMG_NUM = 10):
        print("Begin csv ---> tf")
        train_login_df = pd.read_csv('E://Projects//python//kaggle-test//login-analyse-jd//data//t_login.csv', header=0)
        train_trade_df = pd.read_csv('E://Projects//python//kaggle-test//login-analyse-jd//data/t_trade.csv', header=0)
        print("Login records {}, Trade records {}".format(len(train_login_df), len(train_trade_df)))
        login_map = self.toUserMap(train_login_df, "id", sortKey = "timestamp")
        trade_map = self.toUserMap(train_trade_df, "id")
        login_trade = self.concatLoginTrade(login_map, trade_map)
        print("Find {} mapping records".format(len(login_trade)))
        self.createTf(login_trade, 
            "E://Projects//python//kaggle-test//login-analyse-jd//data//tf-train-all", MAX_RECORD_NUM)

    def createTf(self, login_trade, tf_file, MAX_RECORD_NUM = 10):
        print("Begin create tf")
        writer = tf.python_io.TFRecordWriter(tf_file)
        for i in range(0, len(login_trade)):
            login_df = login_trade[i][0]
            trade_df = login_trade[i][1]

            one_row, label = self.transFormat(login_df, trade_df)
            example = tf.train.Example(features=tf.train.Features(feature={
                    'data': self.int64_list_feature(one_row),
                    'label': self.int64_list_feature(label)}))
            writer.write(example.SerializeToString())
            if(i >= MAX_RECORD_NUM):
                break
            i += 1

    def readTf_simple(self, tf_file):
        datas = []
        labels = []
        index = 1
        for serialsed_example in tf.python_io.tf_record_iterator(tf_file):
            example = tf.train.Example()
            example.ParseFromString(serialsed_example)
            
            datas.append(example.features.feature["data"].int64_list.value)
            labels.append(example.features.feature["label"].int64_list.value)
            if(index >= 128):
                break
            index += 1
        return np.array(datas), np.array(labels)

    def test_tf(self, tf_file):
        index = 0
        for serialsed_example in tf.python_io.tf_record_iterator(tf_file):
            example = tf.train.Example()
            example.ParseFromString(serialsed_example)
            #example.features is type of tensorflow.core.example.feature_pb2.Features
            label_info = {}
            label_info["data"] = example.features.feature["data"].int64_list.value
            label_info["label"] = example.features.feature["label"].int64_list.value
            
            print("Tf file {} {}".format(index, label_info["label"]))
            index = index + 1

#%%
if(__name__ == "__main__"):
    print("Test Start...")
    tfManager = TFRecordManager()
    #tfManager.readCsvCreateTf()
    #tfManager.test_tf("E://Projects//python//kaggle-test//login-analyse-jd//data//tf-train-all")