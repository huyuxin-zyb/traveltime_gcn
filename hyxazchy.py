import numpy as np
import scipy.sparse as sp
import pickle as pkl
import sys
import networkx as nx
from utils import *
import tensorflow as tf
from utils import *
import matplotlib.pyplot as plt

# adj,x_train, x_val, x_test, y_train, y_val, y_test,mask_train, \
# mask_val, mask_test = load_data('2')
# # print x_train[0]
# # print y_train[0]
# # print mask_train[0]
# s, c = 0.0, 0
# for j in range(1):
#     pre = x_test[j]
#     test = y_test[j]
#     for i in range(len(pre)):
#         if mask_test[j][i] != 0:
#             s += abs(pre[i][0] - test[i][0]) / test[i][0]
#             print pre[i], test[i][0]
#             c += 1
# print s / c

# a=tf.sparse_placeholder(tf.float32)
# b=tf.placeholder(tf.float32,shape=(None, 3))
# # e=tf.sparse_to_dense(a)
# c=tf.sparse_tensor_dense_matmul(a,b)
# with tf.Session() as sess:
#     a1=[[(0,0),(2,1),(0,1),(1,2)],[.5,1.,.5,1.],(3,3)]
#     b1=np.array([[1,2,3],[2,3,4],[4,5,6]])
#     d=sess.run([a,b,c],feed_dict={a:a1,b:b1})
#     print d[0]
#     print d[1]
#     print d[2]

b=tf.placeholder(tf.float32,shape=(2, 2))
a=tf.placeholder(tf.float32,shape=(2, 2))
# d=tf.reshape(a,(1,-1,-1))
c=tf.matmul(a,b)
a = tf.cast(a, dtype=tf.float32)
d=a*b
d=tf.reduce_mean(d)
with tf.Session() as sess:

    b1=np.array([[1,5],[5,3]])
    a1=np.array([[1,0],[0,1]])
    print (a1)
    d1=sess.run([b,c,d],feed_dict={b:b1,a:a1})
    print (d1[2])
    print (d1[1])
# print 5e-4
# pre,y_test,x_test,mask_test=pkl.load(open('his_d/test_in','rb'))
# for i in range(len(pre)):
#     ddd=count_err(pre[i],y_test[i],x_test[i],mask_test[i])
#     print ddd[0]
#     print 'yuce inf',np.mean(np.array(ddd[0]))
#     print ddd[1]
#     print 'shishi inf',np.mean(np.array(ddd[1]))
# # print y_test
# # print
# # print pre
# df=trip_err(pre,y_test)
# for id,group in df.groupby('id'):
#     if len(group)==32:
#         plt.plot(group['length'],group['err'],'x-')
# # plt.hist(df['num'])
# plt.show()
# print  df.describe()

# adj,x_train, x_val, x_test, y_train, y_val, y_test,mask_train, \
# mask_val, mask_test = pkl.load(open('his_d/sample','rb'))
# x_train, y_train, mask_train=shuffle_sample(x_train, y_train, mask_train)
# print adj.nnz
# adj_normalized = normalize_adj(adj)
# laplacian = sp.eye(adj.shape[0]) - adj_normalized
# largest_eigval, _ = eigsh(laplacian, 1, which='LM')
# scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])
# t_k = list()
# t_k.append(sp.eye(adj.shape[0]))
# t_k.append(scaled_laplacian)
# s_lap = sp.csr_matrix(scaled_laplacian, copy=True)
#
# # adj_normalized = normalize_adj(adj)
# print '++=================================================================++'
# t_k.append(2* s_lap.dot(t_k[-1]) - t_k[-2])
# print [k.nnz for k in t_k]
