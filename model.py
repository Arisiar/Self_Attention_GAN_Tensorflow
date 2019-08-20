import os
import numpy as np
import tensorflow as tf
from utils import *
from opts import *

GPU = '/gpu:0'
DIM = 128
ITERATION = 10000
PRINT_RATIO = 25
BATCH_SIZE = 21
G_LEARNING_RATE = 0.0001
D_LEARNING_RATE = 0.0004
BETA1 = 0.0
BETA2 = 0.9

class GAN():
    def __init__(self, num_classes = 105, dataset_path = "./dataset\\dogs\\*.*"):
        self.image_path = tf.io.gfile.glob(os.path.join(dataset_path))
        self.image_num = len(self.image_path)
        self.num_classes = num_classes

        self.is_t = tf.placeholder(tf.bool)
        
        self.d_iterator = create_dataset(self.image_path, self.image_num, BATCH_SIZE, GPU)
        self.dataset, self.labels = self.d_iterator.get_next()

        self.z = tf.random.truncated_normal(shape = [BATCH_SIZE, 1, 1, DIM], name = 'random_z')
        self.gen_sparse_class = tf.squeeze(tf.multinomial(tf.zeros((BATCH_SIZE, self.num_classes)), 1))

        self.fake_image = self.generator(self.z, self.gen_sparse_class, self.num_classes, is_t = self.is_t)

        self.real_logit = self.discriminator(self.dataset, self.labels, None, self.num_classes)
        self.fake_logit = self.discriminator(self.fake_image, self.gen_sparse_class, 'NO_OPS', self.num_classes, reuse = True)

        self.d_loss = tf.reduce_mean(relu(1.0 - self.real_logit)) + tf.reduce_mean(relu(1.0 + self.fake_logit))
        self.g_loss = -tf.reduce_mean(self.fake_logit)

        self.g_var = [var for var in tf.trainable_variables() if 'gen' in var.name]
        self.d_var = [var for var in tf.trainable_variables() if 'dis' in var.name]
            
        self.g_optm = tf.train.AdamOptimizer(G_LEARNING_RATE, beta1 = BETA1, beta2 = BETA2).minimize(self.g_loss, var_list = self.g_var)
        self.d_optm = tf.train.AdamOptimizer(D_LEARNING_RATE, beta1 = BETA1, beta2 = BETA2).minimize(self.d_loss, var_list = self.d_var)

        self.sample = self.generator(self.z, self.gen_sparse_class, self.num_classes, is_t = self.is_t, reuse = True)

    def __call__(self):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for itr in range(ITERATION):
                for i in range(2):
                    _, d_loss = sess.run([self.d_optm, self.d_loss], feed_dict = {self.is_t: True})
                _, g_loss = sess.run([self.g_optm, self.g_loss], feed_dict = {self.is_t: True})

                print("Epoch[%2d]: D_Loss: %7f, G_Loss: %7f" % (itr, d_loss, g_loss))
                if np.mod(itr + 1, PRINT_RATIO) == 0:
                    samples = sess.run(self.sample, feed_dict = {self.is_t: False})
                    savefile(samples[:21, :, :, :], [3, 7])

            for i in range(self.image_num):
                result = sess.run(self.sample, feed_dict = {self.is_t: False})
                savefile(result[:1, :, :, :], [1, 1], name = "./images\\" + str(i) + ".png")

    def generator(self, x, y, num_classes, is_t, reuse = False):
        channels = 512
        with tf.variable_scope("gen", reuse = reuse):
            x = linear(x, channels = 4 * 4 * channels, scope = 'upsample_0')
            x = tf.reshape(x, [BATCH_SIZE, 4, 4, channels])
            x = upblock(x, y, 512, num_classes, is_t, scope = 'upsample_1')
            x = upblock(x, y, 256, num_classes, is_t, scope = 'upsample_2')
            x = self_attention(x, 256, scope = "g_attenion")
            x = upblock(x, y, 128, num_classes, is_t, scope = 'upsample_3')
            x = upblock(x, y, 64, num_classes, is_t, scope = 'upsample_4')
            x = batch_norm(x, is_t = is_t, cond = False, scope = 'batch_norm')
            x = relu(x)
            x = conv(x, channels = 3, kernel = 3, scope = 'g_logit')
            x = tf.nn.tanh(x)
        return x

    def discriminator(self, x, y, update_collection, num_classes, reuse = False):
        with tf.variable_scope("dis", reuse = reuse):
            x = inputblock(x, 64, update_collection, scope = "downsample_0")
            x = downblock(x, 128, update_collection, scope = "downsample_1")
            x = self_attention(x, 128, update_collection, scope = "d_attenion")
            x = downblock(x, 256, update_collection, scope = "downsample_2")
            x = downblock(x, 512, update_collection, scope = "downsample_3")
            x = downblock(x, 512, update_collection, down = False, scope = "downsample_4")
            x = relu(x)
            x = tf.reduce_sum(x, axis = [1, 2], keep_dims = False)
            emb = embedding(x, y, num_classes)
            x = linear(x, 1, update_collection, scope = 'd_logit')
            x += emb
        return x


