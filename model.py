import os
import numpy as np
import tensorflow as tf
from utils import *
from opts import *

GPU = '/gpu:0'
DIM = 128
Classes = 105
ITERATION = 100000
PRINT_RATIO = 10
BATCH_SIZE = 21
G_LEARNING_RATE = 0.0001
D_LEARNING_RATE = 0.0004
BETA1 = 0.0
BETA2 = 0.9

class GAN():
    def __init__(self, image_path, image_num):
        self.image_path = image_path
        self.image_num = image_num
        
        images, labels = create_dataset(self.image_path, self.image_num, BATCH_SIZE)
        labels = tf.squeeze(labels)

        z = tf.random_normal(shape = [BATCH_SIZE, DIM], name = 'random_z', dtype=tf.float32)
        gen_sparse_class = tf.squeeze(tf.multinomial(tf.zeros((BATCH_SIZE, Classes)), 1))

        fake_image = self.generator(z, gen_sparse_class, Classes, is_t = True)

        self.real_logit = self.discriminator(images, labels, None, Classes)
        self.fake_logit = self.discriminator(fake_image, gen_sparse_class, 'NO_OPS', Classes, reuse = True)

        self.d_loss = tf.reduce_mean(relu(1.0 - self.real_logit)) + tf.reduce_mean(relu(1.0 + self.fake_logit))
        self.g_loss = -tf.reduce_mean(self.fake_logit)

        self.g_var = [var for var in tf.trainable_variables() if 'gen' in var.name]
        self.d_var = [var for var in tf.trainable_variables() if 'dis' in var.name]
            
        self.g_optm = tf.train.AdamOptimizer(G_LEARNING_RATE, beta1 = BETA1, beta2 = BETA2).minimize(self.g_loss, var_list = self.g_var)
        self.d_optm = tf.train.AdamOptimizer(D_LEARNING_RATE, beta1 = BETA1, beta2 = BETA2).minimize(self.d_loss, var_list = self.d_var)

        self.sample = self.generator(z, gen_sparse_class, Classes, is_t = False, reuse = True)


    def __call__(self):
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)
            for itr in range(ITERATION):
                for i in range(2):
                    _, d_loss = sess.run([self.d_optm, self.d_loss])
                _, g_loss = sess.run([self.g_optm, self.g_loss])

                print("Epoch[%2d]: D_Loss: %7f, G_Loss: %7f" % (itr, d_loss, g_loss))
                if np.mod(itr + 1, PRINT_RATIO) == 0:
                    samples = sess.run(self.sample)
                    savefile(samples[:BATCH_SIZE, :, :, :], [3, 7])

            for i in range(self.image_num):
                result = sess.run(self.sample)
                savefile(result[:1, :, :, :], [1, 1], name = "./images\\" + str(i) + ".png")

    def generator(self, x, labels, num_classes, is_t, reuse = False):
        with tf.variable_scope("gen", reuse = reuse):
            x = tf.reshape(linear(x, channels = 4 * 4 * 512, scope = 'input'), [BATCH_SIZE, 4, 4, 512])
            x = upblock(x, labels, 512, num_classes, is_t, scope = 'upsample_1')
            x = upblock(x, labels, 256, num_classes, is_t, scope = 'upsample_2')
            x = upblock(x, labels, 128, num_classes, is_t, scope = 'upsample_3')
            x = upblock(x, labels,  64, num_classes, is_t, scope = 'upsample_4')
            x = self_attention(x)

            x = relu(batch_norm(x, is_t = is_t, cond = False))
            x = conv(x, channels =  3, kernel = 3, scope = 'g_logit')
            x = tf.nn.tanh(x)

            return x

    def discriminator(self, x, labels, update_collection, num_classes, reuse = False):
        with tf.variable_scope("dis", reuse = reuse):
            x = inputblock(x, 64, update_collection = update_collection, scope = "downsample_1")
            x = self_attention(x, update_collection = update_collection)
            x = downblock(x, 128, update_collection = update_collection, scope = "downsample_2")
            x = downblock(x, 256, update_collection = update_collection, scope = "downsample_3")
            x = downblock(x, 512, update_collection = update_collection, scope = "downsample_4")
            x = downblock(x, 512, update_collection = update_collection, down = False, scope = "downsample_5")
            
            x = tf.reduce_sum(relu(x), axis = [1, 2], keep_dims = False)
            emb = embedding(x, labels, num_classes, update_collection = update_collection)
            x = linear(x, 1, update_collection = update_collection, scope = 'd_logit')

            return x + emb


