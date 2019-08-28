import numpy as np
import tensorflow as tf
import tensorflow.contrib as tf_contrib

zeros_init = tf.zeros_initializer()
ones_init = tf.ones_initializer()
weight_init = tf.contrib.layers.xavier_initializer()

def relu(x):
     return tf.nn.relu(x)
     
def lrelu(x, leak = 0.1):
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
     return tf.nn.avg_pool(x, [1, 2, 2, 1], [1, 2, 2, 1], 'VALID')

def upsample(x, scale = 2):
     _, h, w, _ = x.get_shape().as_list()
     return tf.image.resize_bilinear(x, size = [h * scale, w * scale])

def upblock(x_init, labels, channels, num_classes, is_t, scope = 'upblock'):
     with tf.variable_scope(scope):
          x = relu(batch_norm(x_init, is_t, num_classes, labels, scope = 'cbn_1'))
          x = upsample(x, scale = 2)
          x = conv(x, channels, kernel = 3, scope = 'conv_1')
          x = relu(batch_norm(x, is_t, num_classes, labels, scope = 'cbn_2'))
          x = conv(x, channels, kernel = 3, scope = 'conv_2')

          res = upsample(x_init, scale = 2)
          res = conv(res, channels, kernel = 1, scope = 'res_1')

          return  x + res

def downblock(x_init, channels, update_collection = None, down = True, scope = 'downblock'):
     with tf.variable_scope(scope):  
          x = relu(x_init)
          x = conv(x, channels, kernel = 3, update_collection = update_collection, scope = 'conv_1')
          x = relu(x)
          x = conv(x, channels, kernel = 3, update_collection = update_collection, scope = 'conv_2')
          
          if down:
               x = downsample(x)

          res = conv(x_init, channels, kernel = 1, stride = 1, update_collection = update_collection, scope = 'res')   
          if down:
               res = downsample(res)

          return x + res

def inputblock(x_init, channels, update_collection = None, scope = 'inputblock'):
     with tf.variable_scope(scope):
          x = conv(x_init, channels, kernel = 3, update_collection = update_collection, scope = 'conv_1')
          x = relu(x)
          x = conv(x, channels, kernel = 3, update_collection = update_collection, scope = 'conv_2')
          x = downsample(x)

          res = downsample(x_init)
          res = conv(res, channels, kernel = 1, stride = 1, update_collection = update_collection, scope = 'res')

          return x + res
     
def self_attention(x, update_collection = None, scope = 'attention'):
     with tf.variable_scope(scope):
          batch_size, height, width, num_channels = x.get_shape().as_list()
 
          g = conv(x, num_channels // 8, kernel = 1, update_collection = update_collection, scope = 'g_conv')
          g =  tf.reshape(g, [g.shape[0], -1, g.shape[-1]])

          f = conv(x, num_channels // 8, kernel = 1, update_collection = update_collection, scope = 'f_conv')
          f = tf.layers.max_pooling2d(f, pool_size = 2, strides = 2)
          f =  tf.reshape(f, [f.shape[0], -1, f.shape[-1]])

          attn_map = tf.matmul(g, f, transpose_b = True)
          attn_map = tf.nn.softmax(attn_map)

          h = conv(x, num_channels // 2, kernel = 1, update_collection = update_collection, scope = 'h_conv')
          h = tf.layers.max_pooling2d(h, pool_size = 2, strides = 2)
          h =  tf.reshape(h, [h.shape[0], -1, h.shape[-1]])

          sigma  = tf.get_variable("sigma", [], initializer = tf.constant_initializer(0.0))

          o = tf.matmul(attn_map, h)
          o = tf.reshape(o, shape=[batch_size, height, width, num_channels // 2])
          o = conv(o, num_channels, kernel = 1, update_collection = update_collection, scope='attn_conv')
          
          return sigma * o + x

def embedding(x, labels, number_classes, update_collection = None, scope = 'embedding'):
     with tf.variable_scope(scope):
          embedding = tf.get_variable('embedding_map', shape = [number_classes, x.shape[-1]], initializer = weight_init)
          embedding = tf.transpose(spectral_norm(tf.transpose(embedding), update_collection = update_collection))
          embedding = tf.nn.embedding_lookup(embedding, labels)
          
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

def batch_norm(x, is_t, number_classes = 0, labels = None, cond = True, scope = 'batch_norm'):
     with tf.variable_scope(scope):
          decay = 0.9
          if cond:    
               shape = tf.TensorShape([number_classes]).concatenate(x.shape[-1:])
               moving_shape = tf.TensorShape([1, 1, 1]).concatenate(x.shape[-1:])
               
               gamma = tf.get_variable('gamma', shape, initializer = ones_init) 
               gamma = tf.reshape(tf.nn.embedding_lookup(gamma, labels), [-1, 1, 1, x.shape[-1]])
               
               beta = tf.get_variable('beta', shape, initializer = zeros_init)
               beta = tf.reshape(tf.nn.embedding_lookup(beta, labels), [-1, 1, 1, x.shape[-1]])
               
               moving_mean = tf.get_variable('moving_mean', moving_shape, initializer = zeros_init, trainable = False)
               moving_var = tf.get_variable('moving_var', moving_shape, initializer = ones_init, trainable = False) 

               batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2], keep_dims = True)  
               update_mean = tf.assign(moving_mean, moving_mean * decay + batch_mean * (1 - decay))
               update_var = tf.assign(moving_var, moving_var * decay + batch_var * (1 - decay))
               tf.add_to_collection(tf.GraphKeys, update_mean)
               tf.add_to_collection(tf.GraphKeys, update_var)
               if is_t:
                    return tf.nn.batch_normalization(x, update_mean, update_var, beta, gamma, 1e-3)
               else:
                    return tf.nn.batch_normalization(x, moving_mean, moving_var, beta, gamma, 1e-3)
          else:
               return tf_contrib.layers.batch_norm(x , epsilon = 1e-3, decay = 0.9, 
                              is_training = is_t, scale = True, scope = scope, updates_collections = None)


     
