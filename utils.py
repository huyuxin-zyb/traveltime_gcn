import numpy as np
import os
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

import datetime
import time
import gc
import warnings

warnings.filterwarnings("ignore")

class myDate:

    def __init__(self, day):
        try:
            day = time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(float(day)))
        except:pass
        if len(day)>11:
            self.day=datetime.datetime.strptime(day, '%Y-%m-%d %H:%M:%S')
        else:
            self.day = datetime.datetime.strptime(day, '%Y-%m-%d')

    def get_date(self):
        return self.day.strftime('%Y-%m-%d')

    def get_bin(self,min=30):
        return self.day.hour*(60/min)+ self.day.minute/min+1

    def get_day(self,x_1=True):
        if x_1:
            return datetime.date.isoweekday(self.day)
        if datetime.date.isoweekday(self.day)<5:
            return 1
        if datetime.date.isoweekday(self.day)==5:
            return 2
        return 3

    def get_ago(self,delta):
        delta=datetime.timedelta(days=delta)
        return (self.day-delta).strftime('%Y-%m-%d')


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(v1,v2):
    """Create mask."""
    v2=v2[0]
    if v1[0]>0 and v2>0:
        return 1
    return 0


def his_graph(value,bin1,ind=11):
    if ind==0:
        return True
    if bin1 - ind in value:
        return his_graph(value,bin1,ind-1)
    return False


def load_train(city, startday = 0, end_day = 7,sjd=12):
    """Load data."""

    list_weather = {}
    reader = pd.read_csv('weather/guiyang.csv')
    for aa in reader[['date', 'weather']].values.tolist():
        aa[0] = myDate(aa[0]).get_date()
        list_weather[aa[0]] = aa[1]
    link_dict = {}
    with open('raw/link_id_num.csv', 'r') as file_to_read:
        i = 0
        for lines in file_to_read:
            i += 1
            if i == 1: continue
            lines = lines.strip()
            line = lines.split(',')
            link_dict[line[1]] = line[0]
    link_inf = {}
    with open('raw/gy_contest_link_info.txt', 'r') as file_to_read:
        i = 0
        for lines in file_to_read:
            i += 1
            if i == 1: continue
            lines = lines.strip()
            line = lines.split(',')
            link_id = int(link_dict[line[0]])
            link_inf[link_id] = float(line[1])

    enc1 = OneHotEncoder()
    enc1.fit([[i] for i in range(1, 8)])
    enc2 = OneHotEncoder()
    enc2.fit([[i] for i in range(sjd,sjd+12)])
    enc3 = OneHotEncoder()
    enc3.fit([[i] for i in range(0,4)])
    f_list1 = os.listdir('data/' + city + '/')
    f_list1.sort()
    f_list2 = os.listdir('data/' + city + '_mask/')
    f_list2.sort()
    sample = [[] for _ in range(3)]
    for n in range(startday, end_day):
        #if f_list1[n].split('.')[1] not in list_weather:continue
        print (f_list1[n])
        value = pkl.load(open('data/' + city + '/' + f_list1[n], 'rb'))
        mask = pkl.load(open('data/' + city + '_mask/' + f_list2[n], 'rb'))
        date=myDate(f_list1[n].split('.')[1]).get_date()
        weather_one=enc3.transform(0).toarray()[0]
        day = myDate(f_list1[n].split('.')[1]).get_day()
        day_one = enc1.transform(day).toarray()[0]
        for bin1, feature in value.items():
            # print bin1/6
            bin2 = bin1 - 5
            if bin2/6<sjd or bin2/6>=sjd+12:continue
            bin_one = enc2.transform(bin2 / 6).toarray()[0]
            if his_graph(value, bin1):

                x = [[0 for __ in range(6)] for _ in feature]
                for k in range(6, 12):
                    for j, va in enumerate(value[bin1 - k]):
                        x[j][k - 6] = va
                #x = [vas + [link_inf[j]] + list(bin_one) + list(day_one)+list(weather_one) for j, vas in enumerate(x)]
                x = [vas  + list(bin_one) + list(day_one)  for j, vas in enumerate(x)]
                y = [[1 for __ in range(6)] for _ in feature]
                m=[0 for _ in feature]
                for k in range(len(m)):
                    if k%10<10:
                        m[k]=1
                for k in range(0, 6):
                    for j, va in enumerate(value[bin1 - k]):
                        if va != 0: y[j][5 - k] = va
                        if mask[bin1 - k][j]==0:m[j]=0
                sample[0].append(np.array(x))
                sample[1].append(np.array(y))
                sample[2].append(np.array([[m1 for _ in range(6)] for m1 in m]))
        del value,mask
        gc.collect()

    return sample


def load_data(city,val,startday = 200, end_day = 230,sjd=12):
    """Load data."""

    list_weather = {}
    reader = pd.read_csv('weather/guiyang.csv')
    for aa in reader[['date', 'weather']].values.tolist():
        aa[0] = myDate(aa[0]).get_date()
        list_weather[aa[0]] = aa[1]
    link_dict = {}
    with open('raw/link_id_num.csv', 'r') as file_to_read:
        i = 0
        for lines in file_to_read:
            i += 1
            if i == 1: continue
            lines = lines.strip()
            line = lines.split(',')
            link_dict[line[1]] = line[0]
    link_inf = {}
    length=[[] for _ in range(len(link_dict))]
    with open('raw/gy_contest_link_info.txt', 'r') as file_to_read:
        i = 0
        for lines in file_to_read:
            i += 1
            if i == 1: continue
            lines = lines.strip()
            line = lines.split(',')
            link_id = int(link_dict[line[0]])
            link_inf[link_id] = float(line[1])
            length[link_id]=[float(line[1]) for _ in range(6)]

    graph=pkl.load(open('data/ind.'+city+'.graph','rb'))
    G = nx.Graph()
    elist = []
    for key, value in graph.items():
        l=len(value)
        for v in value:
            # weight=np.exp(-1*(link_inf[key] + link_inf[v]/4))
            # weight = link_inf[key] / (link_inf[key] + link_inf[v])
            # print (link_inf[key] + link_inf[v], 1.0/l)
            elist.append((key, v, 1.0))

    G.add_weighted_edges_from(elist)
    adj=nx.adjacency_matrix(G)
    print (graph)
    # ss
    # adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    enc1 = OneHotEncoder()
    enc1.fit([[i] for i in range(1, 8)])
    enc2 = OneHotEncoder()
    enc2.fit([[i] for i in range(sjd,sjd+12)])
    enc3 = OneHotEncoder()
    enc3.fit([[i] for i in range(0,4)])
    f_list1 = os.listdir('data/' + city + '/')
    f_list1.sort()
    f_list2 = os.listdir('data/' + city + '_mask/')
    f_list2.sort()
    sample = [[] for _ in range(3)]
    i=0
    for n in range(startday, end_day):
        # if f_list1[n].split('.')[1] not in list_weather:continue
        print (f_list1[n])
        value = pkl.load(open('data/' + city + '/' + f_list1[n], 'rb'))
        mask = pkl.load(open('data/' + city + '_mask/' + f_list2[n], 'rb'))
        date=myDate(f_list1[n].split('.')[1]).get_date()
        weather_one=enc3.transform(0).toarray()[0]
        day = myDate(f_list1[n].split('.')[1]).get_day()
        # print(enc1.transform(day).toarray())
        day_one = enc1.transform(day).toarray()[0]
        for bin1, feature in value.items():
            # print bin1/6
            bin2 = bin1 - 5
            if bin2/6<sjd or bin2/6>=sjd+12:continue
            bin_one = enc2.transform(bin2 / 6).toarray()[0]
            if his_graph(value, bin1):
                x = [[0 for __ in range(6)] for _ in feature]
                for k in range(6, 12):
                    for j, va in enumerate(value[bin1 - k]):
                        x[j][k - 6] = va
                #x = [vas + [link_inf[j]] + list(bin_one) + list(day_one)+list(weather_one) for j, vas in enumerate(x)]
                x = [vas + list(bin_one) + list(day_one)  for j, vas in enumerate(x)]
                y = [[1 for __ in range(6)] for _ in feature]

                m=[1 for _ in feature]
                for k in range(0, 6):
                    for j, va in enumerate(value[bin1 - k]):
                        #if va==0:print (str(va)+'\t'+str(mask[bin1 - k][j]))
                        if va!=0:y[j][5 - k] = va
                        if mask[bin1 - k][j]==0:m[j]=0
                if val:
                    if i%1==0:
                        sample[0].append(np.array(x))
                        sample[1].append(np.array(y))
                        sample[2].append(np.array([[m1 for _ in range(6)] for m1 in m]))
                else:
                    sample[0].append(np.array(x))
                    sample[1].append(np.array(y))
                    sample[2].append(np.array([[m1 for _ in range(6)] for m1 in m]))
                i+=1
        del value,mask
        gc.collect()
    # print(sample[0][0])
    # ssss
    if val:
        return [adj] + sample+[length]
    return sample


def shuffle_sample(x,y,m):
    import random
    ind=[i for i in range(len(x))]
    random.shuffle(ind)
    n_x,n_y,n_m=[],[],[]
    for i in ind:
        n_x.append(x[i])
        n_y.append(y[i])
        n_m.append(m[i])
    del x,y,m
    gc.collect()
    return n_x,n_y,n_m


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, placeholders,mask):
    """Construct feed dictionary."""
    # print labels
    feed_dict={}
    feed_dict.update({placeholders['mask'][i]: mask[i] for i in range(len(mask))})
    feed_dict.update({placeholders['labels'][i]: labels[i] for i in range(len(labels))})
    feed_dict.update({placeholders['features'][i]: features[i] for i in range(len(features))})

    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    # feed_dict.update({placeholders['length']: len_list})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def count_err(pre,y_test,x_test):
    ll1=[]
    ll2=[]
    for j in range(len(pre[0])):
        s1,s2, c = 0.0, 0.0,0.0
        for i in range(len(pre)):
                s1 += abs(pre[i][j] - y_test[i][j]) / y_test[i][j]
                s2 += abs(x_test[i][0] - y_test[i][j]) / y_test[i][j]
                c += 1
        ll1.append(s1/c)
        ll2.append(s2/c)
    return ll1,ll2


def get_length():
    link_dict = {}
    with open('raw/link_id_num.csv', 'r') as file_to_read:
        i = 0
        for lines in file_to_read:
            i += 1
            if i == 1: continue
            lines = lines.strip()
            line = lines.split(',')
            link_dict[line[1]] = line[0]
    link_inf = {}
    with open('raw/gy_contest_link_info.txt', 'r') as file_to_read:
        i = 0
        for lines in file_to_read:
            i += 1
            if i == 1: continue
            lines = lines.strip()
            line = lines.split(',')
            link_id = int(link_dict[line[0]])
            link_inf[link_id] = float(line[1])
    return link_inf


def sim_trip(tim,link_inf,tirp):
    t=0
    for ti in tirp:
        ind=int(t/600)
        t+=max(link_inf[ti]/(tim[ti][min(ind,5)]),0.0)
    return t



def trip_err(p,y):
    import pandas as pd
    link_inf=get_length()
    err_list=[]
    err_tr = []
    print (len(p),len(y))
    for bin1 in range(len(p)):
        pre = p[bin1]
        y_test = y[bin1]
        for i in range(len(pre)):
            for j in range(len(pre[i])):
                err_list.append([bin1,j,i,abs(max(pre[i][j],0.0) - y_test[i][j]) / y_test[i][j]])
        with open('data/trip.txt', 'r') as file_to_read:
            i = 0
            for lines in file_to_read:
                i += 1
                if i == 1: continue
                lines = lines.strip()
                line = lines.split(',')
                trip = map(int, line[-1].split('#'))
                c = 0.0
                l=0.0
                for ti in trip:
                    c += max(0.0,link_inf[ti]/y_test[ti][0])
                    l+=link_inf[ti]
                s=sim_trip(pre,link_inf,trip)
                err = abs(s - c) / c
                if err>0.4 or c<120 or l<1000:
                    continue
                # print line[2],err,s,c+
                err_tr.append([bin1,int(line[0]), float(line[2]), float(line[3]), err,c,s])

    df1 = pd.DataFrame(err_list, columns=['bin','tim','id','err'])
    print (df1.describe())
    df = pd.DataFrame(err_tr, columns=['bin','id', 'num', 'length', 'err','tra_time','pre_time'])
    return df



def mergesort(seq):
    if len(seq) <= 1:
        return seq
    mid = int(len(seq) / 2)
    left = mergesort(seq[:mid])
    right = mergesort(seq[mid:])
    return merge(left, right)


def merge(left, right):
    result = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        if left[i][0] <= right[j][0]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result += left[i:]
    result += right[j:]
    return result

