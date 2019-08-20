import numpy as np
import tensorflow as tf
import tensorflow.contrib as tf_contrib

weight_init = tf.contrib.layers.xavier_initializer()

def relu(x):
     return tf.nn.relu(x)
     
def lrelu(x, leak=0.2):
     return tf.nn.leaky_relu(x, leak)

def conv(x, channels, kernel = 3, stride = 1, update_collection = None, sn = True, scope = "conv"):
     with tf.variable_scope(scope):
          w = tf.get_variable('w', [kernel, kernel, x.get_shape()[-1], channels], initializer = weight_init)
          b = tf.get_variable('b', [channels], initializer = tf.zeros_initializer())
          conv = tf.nn.conv2d(x, spectral_norm(w, update_collection), strides = [1, stride, stride, 1], padding = 'SAME')   
          conv = tf.nn.bias_add(conv, b)         

     return conv

def linear(x, channels, sn = True, update_collection = None, scope = 'linear'):
     shape = x.get_shape().as_list()
     with tf.variable_scope(scope):
          w = tf.get_variable("w", [shape[-1], channels], tf.float32, initializer = weight_init)
          b = tf.get_variable('b', [channels], initializer = tf.constant_initializer(0.0))
          rtn = tf.matmul(x, spectral_norm(w, update_collection)) + b
     return rtn

def downsample(x):
     return tf.layers.average_pooling2d(x, pool_size = 2, strides = 2, padding = 'SAME')

def upsample(x, scale = 2):
     _, h, w, _ = x.get_shape().as_list()
     return tf.image.resize_bilinear(x, size = [h * scale, w * scale])

def upblock(x_init, labels, channels, num_classes, is_t, scope = 'upblock'):
     with tf.variable_scope(scope):
          x = batch_norm(x_init, is_t, num_classes, y = labels, scope = 'cbn_1')
          x = relu(x)
          x = upsample(x, scale = 2)
          x = conv(x, channels, kernel = 3, scope = 'conv_1')
          x = batch_norm(x, is_t, num_classes, y = labels, scope = 'cbn_2')
          x = relu(x)
          x = conv(x, channels, kernel = 3, scope = 'conv_2')
          res = upsample(x_init, scale = 2)
          res = conv(res, channels, kernel = 1, scope = 'res_1')
          x = x + res
     return x

def downblock(x_init, channels, update_collection = None, down = True, scope = 'downblock'):
     with tf.variable_scope(scope):  
          x = relu(x_init)
          x = conv(x, channels, kernel = 3, update_collection = update_collection, scope = 'conv_1')
          x = relu(x)
          x = conv(x, channels, kernel = 3, update_collection = update_collection, scope = 'conv_2')
          res = conv(x_init, channels, kernel = 1, stride = 1, update_collection = update_collection, scope = 'res')   
          if down:
               x = downsample(x)
               res = downsample(res)
          x = x + res
     return x

def inputblock(x_init, channels, update_collection = None, scope = ''):
     with tf.variable_scope(scope):
          x = conv(x_init, channels, kernel = 3, update_collection = update_collection, scope = 'conv_1')
          x = relu(x)
          x = conv(x, channels, kernel = 3, update_collection = update_collection, scope = 'conv_2')
          x = downsample(x)

          res = downsample(x_init)
          res = conv(res, channels, kernel = 1, stride = 1, update_collection = update_collection, scope = 'res')

          x = x + res
     return x
     
def self_attention(x, channels, update_collection = None, scope = 'attention'):
     with tf.variable_scope(scope):
          batch_size, height, width, num_channels = x.get_shape().as_list()
          f = conv(x, channels // 8, kernel = 1, stride = 1, update_collection = update_collection, scope = 'f_conv')
          f = tf.layers.max_pooling2d(f, pool_size = 2, strides = 2, padding = 'SAME')

          g = conv(x, channels // 8, kernel = 1, stride=1, update_collection = update_collection, scope = 'g_conv')

          h = conv(x, channels // 2, kernel = 1, stride = 1, update_collection = update_collection, scope = 'h_conv')
          h = tf.layers.max_pooling2d(h, pool_size = 2, strides = 2, padding = 'SAME')

          s = tf.matmul(tf.reshape(g, shape = [g.shape[0], -1, g.shape[-1]]), tf.reshape(f, shape = [f.shape[0], -1, f.shape[-1]]), transpose_b = True)

          beta = tf.nn.softmax(s)

          o = tf.matmul(beta, tf.reshape(h, shape = [h.shape[0], -1, h.shape[-1]]))
          gamma = tf.get_variable("gamma", [1], initializer = tf.constant_initializer(0.0))

          o = tf.reshape(o, shape=[batch_size, height, width, num_channels // 2])
          o = conv(o, channels, kernel = 1, stride = 1, update_collection = update_collection, scope='attn_conv')
          x = gamma * o + x

     return x

def embedding(x, y, number_classes, scope = 'embedding'):
     with tf.variable_scope(scope):
          embedding = tf.get_variable('embedding_map', shape = [number_classes, x.shape[-1]], initializer = weight_init)
          embedding = tf.transpose(spectral_norm(tf.transpose(embedding)))
          embedding = tf.nn.embedding_lookup(embedding, y)
          
     return tf.reduce_sum(embedding * x, axis=1, keep_dims=True)

def spectral_norm(w, update_collection = None, num_iters = 1):
     w_shape = w.shape.as_list()
     w = tf.reshape(w, [-1, w_shape[-1]])

     u = tf.get_variable("u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable = False)

     u_hat = u
     v_hat = None
     for _ in range(num_iters):
          v_ = tf.matmul(u_hat, tf.transpose(w))
          v_hat = tf.nn.l2_normalize(v_)

          u_ = tf.matmul(v_hat, w)
          u_hat = tf.nn.l2_normalize(u_)

     u_hat = tf.stop_gradient(u_hat)
     v_hat = tf.stop_gradient(v_hat)

     sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
     w_bar = w / sigma
     if update_collection is None:
          with tf.control_dependencies([u.assign(u_hat)]):
               w_bar = tf.reshape(w_bar, w_shape)
     else:
          w_bar = tf.reshape(w_bar, w_shape)
          if update_collection != 'NO_OPS':
               tf.add_to_collection(update_collection, u.assign(u_hat))

     return w_bar

def batch_norm(x, is_t, number_classes = 105, y = None, cond = True, scope = 'batch_norm', decay = 0.9):
     with tf.variable_scope(scope):
          if cond:    
               moving_shape = tf.TensorShape([1, 1, 1]).concatenate(x.shape[-1])
               
               gamma = tf.get_variable('gamma', [number_classes, x.shape[-1]], initializer = tf.ones_initializer()) 
               gamma = tf.reshape(tf.nn.embedding_lookup(gamma, y), [-1, 1, 1, x.shape[-1]])
               
               beta = tf.get_variable('beta', [number_classes, x.shape[-1]], initializer = tf.zeros_initializer())
               beta = tf.reshape(tf.nn.embedding_lookup(beta, y), [-1, 1, 1, x.shape[-1]])
               
               moving_mean = tf.get_variable('moving_mean', moving_shape, initializer = tf.zeros_initializer())
               moving_var = tf.get_variable('moving_var', moving_shape, initializer = tf.ones_initializer()) 

               def is_training():
                    batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], name = 'moments', keep_dims = True)  
                    mean = tf.assign(moving_mean, moving_mean * decay + batch_mean * (1 - decay))
                    var = tf.assign(moving_var, moving_var * decay + batch_var * (1 - decay))
                    with tf.control_dependencies([mean, var]):
                         return tf.identity(mean), tf.identity(var)
               
               mean, var = tf.cond(is_t, is_training, lambda: (moving_mean, moving_var))

               return tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-5)
          else:
               return tf_contrib.layers.batch_norm(x , epsilon = 1e-5, decay = 0.9, is_training = is_t, scale = True, scope = scope)


     
