# encoding=utf-8
import csv
import numpy as np
from numpy import linalg as LA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import normalize
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def set_position(longitude,latitude,altitude,minlong,minlat,minalt):
    tmp = []
    grid_x = int((longitude - minlong)/0.1)
    grid_y = int((latitude - minlat)/0.1)
    grid_z = int((altitude - minalt)/300)
    tmp.append(grid_x)
    tmp.append(grid_y)
    tmp.append(grid_z)
    return tmp

def similarity_function(points):
    """
    相似性函数，利用径向基核函数计算相似性矩阵，对角线元素置为０
    对角线元素为什么要置为０我也不清楚，但是论文里是这么说的
    :param points:
    :return:
    """
    res = rbf_kernel(points)
    for i in range(len(res)):
        res[i, i] = 0
    return res

def spectral_clustering(points, k):
    """
    谱聚类
    :param points: 样本点
    :param k: 聚类个数
    :return: 聚类结果
    """
    W = points

    Dn = np.diag(np.power(np.sum(W, axis=1), -0.5))

    L = np.eye(len(points)) - np.dot(np.dot(Dn, W), Dn)
    eigvals, eigvecs = LA.eig(L)

    indices = np.argsort(eigvals)[:k]

    k_smallest_eigenvectors = normalize(eigvecs[:, indices])

    return KMeans(n_clusters=k).fit_predict(k_smallest_eigenvectors)

with open(inputfile,'r') as f:
    read = csv.reader(f)
    originFile = []
    for i in read:
        originFile.append(i)

    count = 0
    newFile = []
    flight_serie = {}

    originFile[0].append(count)
    newFile.append(originFile[0]) 
    flight_serie[count] = originFile[0][1]
#
    for row in range(1,len(originFile)-1):
        if originFile[row][1] == newFile[len(newFile)-1][1]:
            originFile[row].append(count)
            newFile.append(originFile[row])
        else:
            count +=1
            originFile[row].append(count)
            newFile.append(originFile[row])
            flight_serie[count] = originFile[row][1]

    print('Store Data')

    tmp_w_file = []
    s_w_file = []
    w_file_set = []
    s_w_file_dict ={}
    minalt = round(float(newFile[0][4]))
    for i in newFile:
        if int(i[4])%300 == 0:
            tmp_w_file.append([i[0],i[1],round(float(i[2]),2),round(float(i[3]),2),int(i[4]),i[6]])
            minlong = 116.6
            minlat = 40.2
            minalt = min(minalt,round(float(i[4]),2))
#             w_file store time, flight number, longitude, latitude
    s_w_file = np.arange(0,81)

    w_file_set.append(tmp_w_file[0])
    for j in range(1,len(tmp_w_file)-1):
        if int((tmp_w_file[j][2]-minlong)/0.1) == int((tmp_w_file[j-1][2]-minlong)/0.1) and int((tmp_w_file[j][3]-minlat)/0.1) == int((tmp_w_file[j-1][3]-minlat)/0.1):
            continue
        else:
            w_file_set.append(tmp_w_file[j])
##                w_file_set filter the same position

    w_file_dict = {}
###         w_file_dict construct flight-trajectory dict
    w_file_dict[w_file_set[0][1]] = [set_position(w_file_set[0][2],w_file_set[0][3],w_file_set[0][4],minlong,minlat,minalt)]
##    print(w_file_dict)
    for j in range(1,len(w_file_set)-1):
        if w_file_set[j][1] == w_file_set[j-1][1]:
            w_file_dict[w_file_set[j][1]].append(set_position(w_file_set[j][2],w_file_set[j][3],w_file_set[j][4],minlong,minlat,minalt))
        else:
            w_file_dict[w_file_set[j][1]] = [set_position(w_file_set[j][2],w_file_set[j][3],w_file_set[j][4],minlong,minlat,minalt)]
    
    sort_dict = {}#sort every trajectory to confirm they follow same direction
    
    count = 0
    sim_dict ={}
    for keys in w_file_dict:
        sim_dict[count] = keys
        count +=1
    
    matrix = np.load("similar_matrix.npy")
    for i in range(len(matrix)):
        for j in range(i):
            matrix[i][j] = matrix[j][i]
    print(matrix[0][0])
    mat = np.array(matrix)
    # print(mat)
    labels = spectral_clustering(mat,13)
    print(labels)
    
    dict_length = max(labels)
    flight_dict = {}
    for k in range(dict_length + 1):
        flight_dict[k] = []
    for j in range(len(sim_dict)):
        flight_dict[labels[j]].append(sim_dict[j])
    print(flight_dict)
    
    for keys in flight_dict:
            FileNameSerie = str(keys)
            ClassData = []
            with open(outputfile,'w',newline='') as wf:
                writer = csv.writer(wf)
                for serie in flight_dict[keys]:
    #                print(serie)
                    for flight in newFile:
                        if flight[1] == serie:
                            writer.writerow(flight)
