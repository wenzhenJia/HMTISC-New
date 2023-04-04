# %%
import os
import time

import numpy as np
import pandas as pd
import seaborn as sns
from generate_dataset import generate_datasets, delete
from sklearn.cluster import KMeans


def generate_spectral(n_clusters=20, total_iteration = 11):

    def spectral_cluster(X, n_clusters=3, sigma=0.1): 
        '''
        n_cluster : cluster into n_cluster subset
        sigma: a parameter of the affinity matrix
        
        '''
        def affinity_matrix(X, sigma=0.01):
            N = len(X)
            A = np.zeros((N, N))
            for i in range(N):
                for j in range(N):
                    dist = np.sqrt((X[i][0]-X[j][0])**2 + (X[i][1]-X[j][1])**2)
                    A[i,j] = np.exp(-1 * dist/(2*sigma**2))
            return A
        
        A = affinity_matrix(X, sigma)
        
        def laplacianMatrix(A):
            dm = np.sum(A, axis=1)
            D = np.diag(dm)
            sqrtD = np.diag(1.0 / (dm ** 0.5))
            return np.dot(np.dot(sqrtD, A), sqrtD)
        
        L = laplacianMatrix(A)
        
        def largest_n_eigvec(L, n_clusters):
            eigval, eigvec = np.linalg.eig(L)
            index = np.argsort(np.sum(abs(eigvec), 0))[-n_clusters:]  
            # n_clusters largest eigenvectors' indexes.
            return eigvec[:, index] 
            
        newX = largest_n_eigvec(L, n_clusters)
        
        def renormalization(newX):
            Y = newX.copy()
            for i in range(len(newX)):
                norm = 0
                for j in newX[i]:
                    norm += (newX[i] ** 2)
                norm = norm ** 0.5
                Y[i] /= norm
            return Y
        
        Y = renormalization(newX)
        
        kmeans = KMeans(n_clusters=n_clusters).fit(Y)
        return kmeans.labels_


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

    # nyc 4392
    # dc 52608
    time_steps = 4392
    stations=pd.read_table('./nyc.txt',sep='\t',header=None).values     #读入txt文件，分隔符为\t
    station_num = stations.shape[0]

    NYCdata_dir = list(map(lambda x : os.path.join(os.getcwd(), 'NYCdata', x), os.listdir('./NYCdata')))
    for single_csv in NYCdata_dir:
        single_data_frame = pd.read_csv(single_csv)
        if single_csv == NYCdata_dir[0]:
            all_data_frame = single_data_frame
        else:  # concatenate all csv to a single dataframe, ingore index
            all_data_frame = pd.concat([all_data_frame, single_data_frame], ignore_index=True)
    all_data_frame['starttime'] = all_data_frame['starttime'].transform(time_resolve)
    all_data_frame['stoptime'] = all_data_frame['stoptime'].transform(time_resolve)
    all_data_frame = all_data_frame.values

    def fcm_classify(data, n_clusters, iteration_times=0):
        
        # af = SpectralClustering(n_clusters).fit(data)
        fcm_labels = spectral_cluster(data, n_clusters)
        delete('./spectral-nyc-cluster-result-{}'.format(n_clusters))
        os.mkdir('./spectral-nyc-cluster-result-{}'.format(n_clusters))

        fcm_labels = np.expand_dims(fcm_labels, axis=1)
        station_labels = np.concatenate((stations, fcm_labels), axis=1)

        if total_iteration == iteration_times:
            with open("./spectral-nyc-cluster-result-{}/cluster_data_{}.txt".format(n_clusters,iteration_times), 'w') as f:
                for row in data:
                    f.write('{}\n'.format('\t'.join(map(str, row.tolist()))))
            with open("./spectral-nyc-cluster-result-{}/cluster_{}.txt".format(n_clusters,iteration_times), 'w') as f:
                for row in station_labels:
                    f.write('{}\t{}\t{}\n'.format(row[0], row[1], int(row[2])))
            data_plot = pd.DataFrame(station_labels, columns = ['lng','lat','type'])
            fig = sns.scatterplot(x = "lng", y = "lat", hue="type",  data=data_plot, legend = False)
            scatter_fig = fig.get_figure()
            scatter_fig.savefig('./spectral-nyc-cluster-result-{}/cluster_{}.png'.format(n_clusters,iteration_times), dpi = 400)

            generate_datasets(NYCdata_dir ,"./spectral-nyc-cluster-result-{}/cluster_{}.txt".format(n_clusters,iteration_times), n_clusters, iteration_times)
    
        return station_labels


    station_label_idx = dict()
    for idx, station_label in enumerate(stations):
        station_label_idx[str(station_label[0])+str(station_label[1])] = idx

    def cal_station_label_map_idx(data, n_clusters, i):
        # 迭代根据四维进行分类
        station_labels = fcm_classify(data, n_clusters, i)
        station_label_map = dict()
        for station_label in station_labels:
            station_label_map[str(station_label[0])+str(station_label[1])] = int(station_label[2])
        return station_label_map

    def cal_extra_information(all_data_frame, station_label_idx, station_label_map):
        trend_in = dict()
        trend_out = dict()
        for row in all_data_frame:
            start_station = str(row[5])+str(row[6])
            end_station = str(row[9]) + str(row[10])
            if start_station not in station_label_idx or end_station not in station_label_idx: continue
            # starttime 
            if row[1] > time_steps or row[2] > time_steps: continue
            # 每个小时该站点到每个簇流入矩阵、流出矩阵
            if start_station in trend_out:
                trend_out[start_station][row[1]-1, station_label_map[end_station]] += 1
            else:
                trend_out[start_station] = np.zeros((time_steps,station_num))
            if end_station in trend_in:
                trend_in[end_station][row[2]-1, station_label_map[start_station]] += 1
            else:
                trend_in[end_station] = np.zeros((time_steps,station_num))
        station_with_extra_information = np.concatenate((stations, np.zeros((station_num, 2)) ), axis=1)
        for station in station_with_extra_information:
            station_name = str(station[0])+str(station[1])
            # 流入矩阵、流出矩阵 求斐波那契范数
            station[2] = np.linalg.norm(trend_in.get(station_name, 0))
            station[3] = np.linalg.norm(trend_out.get(station_name, 0))
        extra_information = station_with_extra_information[:, 2:4]
        # 趋势信息不进行归一化
        # from sklearn.preprocessing import MinMaxScaler
        # min_max_scaler = MinMaxScaler()
        # extra_information = min_max_scaler.fit_transform(extra_information)
        # station_with_normal_extra_information = np.concatenate((stations, extra_information, station_with_extra_information[:,4:]), axis=1)
        station_with_normal_extra_information = np.concatenate((stations, extra_information), axis=1)
        return station_with_normal_extra_information


    stations_data = stations
    for i in range(1, total_iteration+1):
        station_label_map = cal_station_label_map_idx(stations_data, n_clusters, i)
        station_with_normal_extra_information = cal_extra_information(all_data_frame, station_label_idx, station_label_map)
        stations_data = station_with_normal_extra_information


if __name__ == "__main__":
    generate_spectral()
