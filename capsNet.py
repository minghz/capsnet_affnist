"""
License: Apache-2.0
Author: Huadong Liao
E-mail: naturomics.liao@gmail.com
"""

import tensorflow as tf

from config import cfg
from load_data import get_batch_data
from utils import softmax
from utils import reduce_sum
from capsLayer import CapsLayer


epsilon = 1e-9


class CapsNet(object):
    def __init__(self, is_training=True):
        if is_training:
            self.X, self.labels = get_batch_data(cfg.batch_size, cfg.num_threads)
            self.Y = tf.one_hot(self.labels, depth=10, axis=1, dtype=tf.float32)

            self.build_arch()
            self.loss()
            self._summary()
            self._accuracy()

            # t_vars = tf.trainable_variables()
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
            #self.optimizer = tf.train.AdamOptimizer()
            self.optimizer = tf.train.GradientDescentOptimizer(0.01) # uses less memory
            self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)  # var_list=t_vars)
        else:
            self.X = tf.placeholder(tf.float32, shape=(cfg.batch_size, 40, 40, 1))
            self.labels = tf.placeholder(tf.int32, shape=(cfg.batch_size, ))
            self.Y = tf.reshape(self.labels, shape=(cfg.batch_size, 10, 1))
            self.build_arch()

        tf.logging.info('Seting up the main structure')

    def build_arch(self):
        with tf.variable_scope('Conv1_layer'):
            # Conv1, [batch_size, 20, 20, 256]
            self.conv1 = tf.contrib.layers.conv2d(self.X, num_outputs=256,
                                             kernel_size=9, stride=1,
                                             padding='VALID')
            assert self.conv1.get_shape() == [cfg.batch_size, 32, 32, 256]

        # Primary Capsules layer, return [batch_size, 1152, 8, 1]
        with tf.variable_scope('PrimaryCaps_layer'):
            self.primaryCaps = CapsLayer(num_outputs=10, vec_len=8, with_routing=False, layer_type='CONV')
            self.caps1 = self.primaryCaps(self.conv1, kernel_size=9, stride=2)
            assert self.caps1.get_shape() == [cfg.batch_size, 1440, 8, 1]

        # DigitCaps layer, return [batch_size, 10, 16, 1]
        with tf.variable_scope('DigitCaps_layer'):
            self.digitCaps = CapsLayer(num_outputs=10, vec_len=16, with_routing=True, layer_type='FC')
            self.caps2 = self.digitCaps(self.caps1)

        # Decoder structure in Fig. 2
        # 1. Do masking, how:
        with tf.variable_scope('Masking'):
            # a). calc ||v_c||, then do softmax(||v_c||)
            # [batch_size, 10, 16, 1] => [batch_size, 10, 1, 1]
            self.v_length = tf.sqrt(reduce_sum(tf.square(self.caps2),
                                               axis=2, keepdims=True) + epsilon)
            self.softmax_v = softmax(self.v_length, axis=1)
            assert self.softmax_v.get_shape() == [cfg.batch_size, 10, 1, 1]

            # b). pick out the index of max softmax val of the 10 caps
            # [batch_size, 10, 1, 1] => [batch_size] (index)
            self.argmax_idx = tf.to_int32(tf.argmax(self.softmax_v, axis=1))
            assert self.argmax_idx.get_shape() == [cfg.batch_size, 1, 1]
            self.argmax_idx = tf.reshape(self.argmax_idx, shape=(cfg.batch_size, ))

            # Method 1.
            if not cfg.mask_with_y:
                # c). indexing
                # It's not easy to understand the indexing process with argmax_idx
                # as we are 3-dim animal
                masked_v = []
                for batch_size in range(cfg.batch_size):
                    v = self.caps2[batch_size][self.argmax_idx[batch_size], :]
                    masked_v.append(tf.reshape(v, shape=(1, 1, 16, 1)))

                self.masked_v = tf.concat(masked_v, axis=0)
                assert self.masked_v.get_shape() == [cfg.batch_size, 1, 16, 1]
            # Method 2. masking with true label, default mode
            else:
                # self.masked_v = tf.matmul(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, 10, 1)), transpose_a=True)
                self.masked_v = tf.multiply(tf.squeeze(self.caps2), tf.reshape(self.Y, (-1, 10, 1)))
                self.v_length = tf.sqrt(reduce_sum(tf.square(self.caps2), axis=2, keepdims=True) + epsilon)

        # 2. Reconstructe the MNIST images with 3 FC layers
        # [batch_size, 1, 16, 1] => [batch_size, 16] => [batch_size, 512]
        with tf.variable_scope('Decoder'):
            vector_j = tf.reshape(self.masked_v, shape=(cfg.batch_size, -1))
            fc1 = tf.contrib.layers.fully_connected(vector_j, num_outputs=512)
            assert fc1.get_shape() == [cfg.batch_size, 512]
            fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=1024)
            assert fc2.get_shape() == [cfg.batch_size, 1024]
            self.decoded = tf.contrib.layers.fully_connected(fc2, num_outputs=1600, activation_fn=tf.sigmoid)

    def loss(self):
        # 1. The margin loss

        # [batch_size, 10, 1, 1]
        # max_l = max(0, m_plus-||v_c||)^2
        max_l = tf.square(tf.maximum(0., cfg.m_plus - self.v_length))
        # max_r = max(0, ||v_c||-m_minus)^2
        max_r = tf.square(tf.maximum(0., self.v_length - cfg.m_minus))
        assert max_l.get_shape() == [cfg.batch_size, 10, 1, 1]

        # reshape: [batch_size, 10, 1, 1] => [batch_size, 10]
        max_l = tf.reshape(max_l, shape=(cfg.batch_size, -1))
        max_r = tf.reshape(max_r, shape=(cfg.batch_size, -1))

        # calc T_c: [batch_size, 10]
        # T_c = Y, is my understanding correct? Try it.
        T_c = self.Y
        # [batch_size, 10], element-wise multiply
        L_c = T_c * max_l + cfg.lambda_val * (1 - T_c) * max_r

        self.margin_loss = tf.reduce_mean(tf.reduce_sum(L_c, axis=1))

        # 2. The reconstruction loss
        orgin = tf.reshape(self.X, shape=(cfg.batch_size, -1))
        squared = tf.square(self.decoded - orgin)
        self.reconstruction_err = tf.reduce_mean(squared)

        # 3. Total loss
        # The paper uses sum of squared error as reconstruction error, but we
        # have used reduce_mean in `# 2 The reconstruction loss` to calculate
        # mean squared error. In order to keep in line with the paper,the
        # regularization scale should be 0.0005*784=0.392
        self.total_loss = self.margin_loss + cfg.regularization_scale * self.reconstruction_err

    # Summary
    def _summary(self):
        '''
        The reason all the sumaries are defined and merged here is because
        the _summary method is not called every step, so this is more efficient
        '''
        tf.summary.histogram('Conv1_layer/conv1', self.conv1)
        tf.summary.histogram('PrimaryCaps_layer/unsquashed_capsules', self.primaryCaps.unsquashed_capsules)
        tf.summary.histogram('PrimaryCaps_layer/capsules', self.primaryCaps.capsules)

        tf.summary.histogram('DigitCaps_layer/W', self.digitCaps.W)
        tf.summary.histogram('DigitCaps_layer/biases', self.digitCaps.biases)
        tf.summary.histogram('DigitCaps_layer/u_hat', self.digitCaps.u_hat)
        tf.summary.histogram('DigitCaps_layer/c_IJ', self.digitCaps.c_IJ)
        tf.summary.histogram('DigitCaps_layer/s_J', self.digitCaps.s_J)
        tf.summary.histogram('DigitCaps_layer/v_J', self.digitCaps.v_J)
        tf.summary.histogram('DigitCaps_layer/u_produce_v', self.digitCaps.u_produce_v)
        tf.summary.histogram('DigitCaps_layer/capsules', self.digitCaps.capsules)

        tf.summary.scalar('margin_loss', self.margin_loss)
        tf.summary.scalar('reconstruction_loss', self.reconstruction_err)
        tf.summary.scalar('total_loss', self.total_loss)

        # Reconstructed image
        recon_img = tf.reshape(self.decoded, shape=(cfg.batch_size, 40, 40, 1))
        tf.summary.image('reconstruction_img', recon_img)

        self.train_summary = tf.summary.merge_all()


    def _accuracy(self):
        correct_prediction = tf.equal(tf.to_int32(self.labels), self.argmax_idx)
        self.accuracy = tf.reduce_sum(tf.cast(correct_prediction, tf.float32))
