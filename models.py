from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', True)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        i=1
        for layer in self.layers:
            with tf.name_scope('GraphConvolution_'+str(i)):
                hidden = layer(self.activations[-1])

                tf.summary.histogram('layer_'+str(i)+'/prediction',hidden)
            with tf.name_scope('activation'):
                self.activations.append(hidden)
                tf.summary.histogram('layer_'+str(i)+'/activation',self.activations[-1])
            i+=1
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()
        with tf.name_scope('summaries_wight'):
            i = 1
            for ll in self.layers:
                for key, var in ll.vars.items():
                    self.var_summaries(var,'layer_'+str(i)+'_'+key)
                # self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
                i += 1
                # break
        with tf.name_scope('train'):
            self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def var_summaries(self,var,name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)
        std = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('std/' + name, std)
    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "model_5_layer/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None,dir='tmp'):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = dir+"/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)


class MLP(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'][0].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)


        self.build()

    def _loss(self):
        # Weight decay loss
        # for ll in self.layers:
        #     for var in ll.vars.values():
        #         self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        #     # break

        # Cross entropy error
        # self.lo,self.ll=masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'])
        self.ll=masked_loss(self.outputs, self.placeholders['labels'],self.placeholders['mask'])
        self.loss += masked_loss(self.outputs, self.placeholders['labels'],self.placeholders['mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],self.placeholders['mask'])

    def _build(self):
        self.layers.append(Dense(input_dim=self.input_dim,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 logging=self.logging))
        for i in range(FLAGS.layer_number):
            self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                     output_dim=FLAGS.hidden1,
                                     placeholders=self.placeholders,
                                     act=tf.nn.relu,
                                     dropout=True,
                                     logging=self.logging))
        self.layers.append(Dense(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x: x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)


class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)
        with tf.name_scope('input'):
            self.inputs = placeholders['features']
            self.input_dim = input_dim
            # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
            self.output_dim = placeholders['labels'][0].get_shape().as_list()[1]
            self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        with tf.name_scope('L2_loss'):
            for var in self.layers[0].vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        # self.lo,self.ll=masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'])
        with tf.name_scope('loss'):
            self.ll=masked_loss(self.outputs, self.placeholders['labels'],self.placeholders['mask'])
            self.loss += masked_loss(self.outputs, self.placeholders['labels'],self.placeholders['mask'])
            tf.summary.scalar('loss/', self.loss/120)

    def _accuracy(self):
        with tf.name_scope('mape'):
            self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],self.placeholders['mask'])
            tf.summary.scalar('mape/', self.accuracy/120 )

    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            logging=self.logging))
        for i in range(FLAGS.layer_number):
            # self.layers.append(Dense(input_dim=FLAGS.hidden1,
            #                                 output_dim=FLAGS.hidden1,
            #                                 placeholders=self.placeholders,
            #                                 act=tf.nn.relu,
            #                                 dropout=True,
            #                                 logging=self.logging))
            self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                 output_dim=FLAGS.hidden1,
                                 placeholders=self.placeholders,
                                 act=tf.nn.relu,
                                 dropout=True,
                                 logging=self.logging))
        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                 output_dim=self.output_dim,
                                 placeholders=self.placeholders,
                                 act=lambda x:x,
                                 dropout=True,
                                 logging=self.logging))

    def predict(self):
        return tf.nn.softmax(self.outputs)
