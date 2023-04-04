# -*- coding: utf-8 -*-
#在程序目录下创建output目录
# 读取csv文件
import csv
import time

import h5py
import numpy as np
import pandas as pd
import os
import shutil

def delete(path):
    """
    删除一个文件/文件夹
    :param path: 待删除的文件路径
    :return:
    """
    if not os.path.exists(path):
        return
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.isfile(path):
        os.remove(path)
    elif os.path.islink(path):
        os.remove(path)
    else:
        pass

def generate_datasets(cvspathlist, clusterpath, n_clusters, iteration_times):
    #时间处理9/11/2014 00:10:00
    def time_resolve(timestr='2014-04-01 00:00:00'):
        # nyc 2014-04-01 00:00:00 --》1396281600
        # dc 2011-01-01 00:00:00 --> 1293811200
        #9/11/2014 00:00:00 -->1410364800.0 +1h-->1410368400=3600s
        try:
            mktime = time.mktime(time.strptime(timestr, '%Y-%m-%d %H:%M:%S'))
        except Exception as ex:
            try:
                mktime = time.mktime(time.strptime(timestr, '%m/%d/%Y %H:%M:%S'))
            except Exception as ex:
                print(ex)
        mark=(mktime-1396281600)//3600+1  #取9/11/2014 00:10:00 为1，一次类推
        return int(mark)
    
    def read_txt(path):
        file = open(path, 'r')
        line_list=[]
        while True:
            line = file.readline()
            if not line :
                break
            else:
                list = line.split("\t")
                list[2] = list[2].replace('\n', '')
                line_list.append(list)
        return line_list
    
    #读取cvs文件
    def readcvs(paths):
        type_dict = dict()
        stations= read_txt(clusterpath)
        for sta in stations:
            type_dict[str(sta[0])+str(sta[1])] = str(sta[2])

        for single_csv in paths:
            single_data_frame = pd.read_csv(single_csv)
            if single_csv == paths[0]:
                all_data_frame = single_data_frame
            else:  # concatenate all csv to a single dataframe, ingore index
                all_data_frame = pd.concat([all_data_frame, single_data_frame], ignore_index=True)
        all_data_frame['starttime'] = all_data_frame['starttime'].transform(time_resolve)
        all_data_frame['stoptime'] = all_data_frame['stoptime'].transform(time_resolve)

        all_data_frame = all_data_frame[all_data_frame['starttime']>0]
        all_data_frame = all_data_frame[all_data_frame['starttime']<=4392]
        all_data_frame['start_mk'] = all_data_frame[['start station latitude', 'start station longitude']].apply(lambda x: type_dict.get(str(x["start station latitude"])+str(x["start station longitude"]), -1), axis=1)
        all_data_frame['stop_mk'] = all_data_frame[['end station latitude', 'end station longitude']].apply(lambda x: type_dict.get(str(x["end station latitude"])+str(x["end station longitude"]), -1), axis=1)
        all_data_frame = all_data_frame[['starttime', 'start_mk', 'stoptime', 'stop_mk']]
        all_data_frame = all_data_frame.values
        return all_data_frame

    sanweis = []
    siweis = []

    startlist = readcvs(cvspathlist)
    for i in range(1,4393):
        sanweis = []
        # sanweis_outout = []
        list_num=[] #一整天的起点数量
        erweis_output = [] 
        for a in startlist:
            if a[0]==i:#起始时间在对应小时文件内
                list_num.append(a)
        #print list_num
        for i1 in range(0,n_clusters):#定义起点序列
            list_excel = []
            for i2 in range(0,n_clusters):#定义终点序列
                count_num=0
                count_start=0
                cont_stop=0
                for mm in list_num:#遍历一小时的序列
                    if mm[1]==str(i1) :
                        count_start+=1#某一类坐标作为起点的数量
                        if mm[3]==str(i2):
                            cont_stop+=1#某一类坐标作为终点的数量
                #横向写入
                if count_start==0:
                    list_excel.append(count_start)
                    #pass
                else:
                    list_excel.append(float(cont_stop) )
            erweis_output.append(list_excel)
            erweis_input = np.transpose(erweis_output)
        sanweis.append(erweis_output)
        sanweis.append(erweis_input)
        # print '------------------------------------------erweis_output---------------------------------------------------------'
        # print erweis_output
        # print '++++++++++++++++++++++++++++++++++++++++++sanweis++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++'
        # print sanweis
        siweis.append(sanweis)
        # print '===========================================siweis=================================================================='

    np.array(siweis)
    delete('./STResNet/data/BikeNYC/NYC14_T60_NewEnd.h5')
    f = h5py.File('./STResNet/data/BikeNYC/NYC14_T60_NewEnd.h5', 'w')  # 打开h5文件
    f.create_dataset('data', shape=(4392, 2, n_clusters, n_clusters), dtype='float64', data=siweis)
    fd = h5py.File('data.h5','r')
    f.create_dataset('date',  data=fd['date'][:])
    f.close()
    fd.close()

if __name__ == "__main__":
    generate_datasets(['/home/hanyang/workspace-1/ideas-validate/New-exp/NYCdata/2014-04 - Citi Bike trip data.csv'], '/home/hanyang/workspace-1/ideas-validate/New-exp/TMC/spectral-nyc-cluster-result-10/cluster_5.txt', 10, 5)
