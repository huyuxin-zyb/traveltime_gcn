from __future__ import division

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

from utils import *
from models import GCN, MLP

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
batch_size=120
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'guiyang', 'Dataset string.')  # 'guiyang', 'beijing', 'guiyang_out'
flags.DEFINE_string('model', 'gcn_cheby', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('layer_number', 3, 'Initial number of layer.')
flags.DEFINE_integer('is_train', 0, 'modle is traineds')  #train 0,or 1
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 50, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 2, 'Maximum Chebyshev polynomial degree.')

adj, x_val,  y_val,_ = load_data(FLAGS.dataset,1)
print len(x_val)
j_start=time.time()
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support=[preprocess_adj(adj)]
    num_supports = 1
    model_func = MLP
else:
    raise ValueError('Invalid argument for model: ' + str(FLAGS.model))
del adj

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': [tf.placeholder(tf.float32,
                shape=(x_val[0].shape[0], x_val[0].shape[1])) for _ in range(batch_size)],
    'labels': [tf.placeholder(tf.float32,
              shape=(x_val[0].shape[0], y_val[0].shape[1])) for _ in range(batch_size)],
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=x_val[0].shape[1], logging=False)
# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(tests, support, labels, placeholders,):
    t_test = time.time()
    loss_t,acc_t=0.0,0.0
    outs_val=''
    out_list=[]
    ii=0
    while (ii + batch_size <= len(tests)):
        start_ind=ii
        end_ind=ii+batch_size
        feed_dict_val = construct_feed_dict(tests[start_ind:end_ind], support, labels[start_ind:end_ind],
                                          placeholders)
        outs_val = sess.run([model.loss, model.accuracy,model.outputs], feed_dict=feed_dict_val)
        acc_t+=outs_val[1]
        loss_t+=outs_val[0]
        out_list.extend(outs_val[-1])
        ii+=batch_size
    return loss_t,acc_t, (time.time() - t_test),out_list,ii


def train_data():
    acc_list = []
    data_set = [[0,200]]
    for se in data_set:

        x_train, y_train = load_train(FLAGS.dataset, se[0], se[1])
        x_train, y_train = shuffle_sample(x_train, y_train)

        epoch = 0
        while (epoch + batch_size <= len(x_train)):
            for step in range(FLAGS.epochs):
                t = time.time()
                feed_dict = construct_feed_dict(x_train[epoch:epoch + batch_size], support,
                                                y_train[epoch:epoch + batch_size], placeholders)
                feed_dict.update({placeholders['dropout']: FLAGS.dropout})
                outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
                if (epoch * FLAGS.epochs / 10 + step) % 10 == 0:
                    cost, acc, duration, _, itr= evaluate(x_val, support, y_val, placeholders)
                    acc_list.append(acc / itr)

                    print("Epoch:", '%04d' % (epoch * FLAGS.epochs / 10 + step + 1), "t_los=",
                          "{:.5f}".format(outs[1] / batch_size),"t_acc=",
                          "{:.5f}".format(outs[2] / batch_size), "v_los=","{:.5f}".format(cost / itr),
                          "v_acc=", "{:.5f}".format(acc / itr), "time=", "{:.5f}".format(time.time() - t),
                          'len_v=','%04d' % (itr))

                # if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
                #     print("Early stopping...")
                #     break
            epoch += 10
        del x_train
        del y_train
        gc.collect()

    print("Optimization Finished!")
    # model.save(sess=sess)
    plt.plot(acc_list)
    plt.xlabel('the time of iter')
    plt.ylabel('mape')
    plt.show()
    # pd.DataFrame(acc_list,columns=['acc']).to_csv('his_d/acc.csv',index=None)

    # plt.savefig('image/shoulian.png')
# Init variables
sess.run(tf.global_variables_initializer())

print j_start-time.time()
if FLAGS.is_train:
    model.load(sess=sess,dir='model_5_layer')

train_data()

del y_val,x_val
gc.collect()
# Testing
x_test,  y_test = load_data(FLAGS.dataset,0)
test_cost, test_acc, test_duration,pre ,itr= evaluate(x_test, support, y_test, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost/itr),
      "accuracy=", "{:.5f}".format(test_acc/itr), "time=", "{:.5f}".format(test_duration),
      'len_test=', '%04d' % (itr))
# print ll
# plk.dump([pre,y_test],open('his_d/test','wb'))
# for i in range(len(pre)):
#     ddd = count_err(pre[i], y_test[i], x_test[i])
#     print 'yuce in', np.mean(np.array(ddd[0]))
#     print 'shishi in', np.mean(np.array(ddd[1]))
