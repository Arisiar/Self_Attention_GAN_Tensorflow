import os
import numpy as np
import tensorflow as tf
from glob import glob
from utils import *
from opts import *

GPU = '/gpu:0'
DIM = 128
ITERATION = 100000
PRINT_RATIO = 25
BATCH_SIZE = 42
G_LEARNING_RATE = 0.0001
D_LEARNING_RATE = 0.0004
BETA1 = 0.0
BETA2 = 0.9

class GAN():
    def __init__(self, num_classes = 105, dataset_path = "./dataset\\dogs\\*.*"):
        self.image_path = glob(os.path.join(dataset_path))
        self.image_num = len(self.image_path)
        self.num_classes = num_classes

        self.is_t = tf.placeholder(tf.bool)
        
        self.d_iterator = create_dataset(self.image_path, self.image_num, BATCH_SIZE, GPU)
        self.dataset, self.labels = self.d_iterator.get_next()

        self.z = tf.random.truncated_normal(shape=[BATCH_SIZE, 1, 1, DIM], name='random_z')
        self.gen_sparse_class = tf.squeeze(tf.multinomial(tf.zeros((BATCH_SIZE, self.num_classes)), 1))

        self.fake_image = self.generator(self.z, self.gen_sparse_class, self.num_classes, is_t = self.is_t)

        self.real_logit = self.discriminator(self.dataset, self.labels, None, self.num_classes)
        self.fake_logit = self.discriminator(self.fake_image, self.gen_sparse_class, 'NO_OPS', self.num_classes)

        self.d_loss = tf.reduce_mean(relu(1.0 - self.real_logit)) + tf.reduce_mean(relu(1.0 + self.fake_logit))
        self.g_loss = -tf.reduce_mean(self.fake_logit)

        self.g_var = [var for var in tf.trainable_variables() if 'gen' in var.name]
        self.d_var = [var for var in tf.trainable_variables() if 'dis' in var.name]
            
        self.g_optm = tf.train.AdamOptimizer(G_LEARNING_RATE, beta1 = BETA1, beta2 = BETA2).minimize(self.g_loss, var_list = self.g_var)
        self.d_optm = tf.train.AdamOptimizer(D_LEARNING_RATE, beta1 = BETA1, beta2 = BETA2).minimize(self.d_loss, var_list = self.d_var)

        
        self.sample = self.generator(self.z, self.gen_sparse_class, self.num_classes, is_t = self.is_t)

    def __call__(self):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for itr in range(ITERATION):
                for _ in range(2):
                    _, d_loss = sess.run([self.d_optm, self.d_loss], feed_dict = {self.is_t: True})
                _, g_loss = sess.run([self.g_optm, self.g_loss], feed_dict = {self.is_t: True})

                print("Epoch[%2d]: D_Loss: %7f, G_Loss: %7f" % (itr, d_loss, g_loss))

                if np.mod(itr + 1, PRINT_RATIO) == 0:
                    samples = sess.run(self.sample, feed_dict = {self.is_t: False})
                    savefile(samples[:9, :, :, :], [3, 3])

            for i in range(self.image_num):
                result = sess.run(self.sample, feed_dict = {self.is_t: False})
                savefile(result[:1, :, :, :], [1, 1], name = "./images\\" + str(i) + ".png")

    def generator(self, x, labels, num_classes, is_t):
        channels = 512
        with tf.variable_scope("gen", reuse = tf.AUTO_REUSE):
            ntype = ['none', 'attention', 'none', 'none', 'none']

            x = linear(x, channels = 4 * 4 * channels, scope = 'input')
            x = tf.reshape(x, [BATCH_SIZE, 4, 4, channels])
            for i in range(4):
                x = upblock(x, labels, channels = channels, num_classes = num_classes, is_t = is_t,  ntype = ntype[i], scope = 'upsample_' + str(i))
                channels = channels // 2
            x = batch_norm(x, labels, num_classes, is_t = is_t, scope = 'conditional_batch_norm')
            x = relu(x)
            rtn = conv(x, channels = 3, kernel = 3, scope = 'g_logit')
        
        return tf.nn.tanh(rtn)

    def discriminator(self, x, labels, update_collection, num_classes):
        channels = 64
        with tf.variable_scope("dis", reuse = tf.AUTO_REUSE):
            ntype = ['none', 'attention', 'none', 'none', 'none']
            
            x = inputblock(x, channels, update_collection = update_collection, scope = "downsample")
            channels = channels * 2
            for i in range(4):
                scale = 2**(min(2, i))
                down = False if i == 3 else True
                x = downblock(x, channels = channels * scale, ntype = ntype[i], update_collection = update_collection, down = down, scope = "downsample_" + str(i))
            
            x = relu(x)
            x = tf.reduce_sum(x, axis = [1, 2])
            rtn = linear(x, channels = 1, update_collection = update_collection, scope = 'd_logit') + embedding(x, labels, num_classes)

        return rtn


