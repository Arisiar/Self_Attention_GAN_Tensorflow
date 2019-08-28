import tensorflow as tf
import numpy as np
import scipy.misc
import os

def create_dataset(data_dir, image_num, batch, image_size = 64): 
    def getlabels(record):
        features = tf.parse_single_example(
            record,
            features={
                'image': tf.FixedLenFeature([], tf.string),
                'label': tf.FixedLenFeature([], tf.int64)
        })  
        image = tf.decode_raw(features['image'], tf.uint8)    
        image.set_shape(image_size * image_size * 3) 
        image = tf.reshape(image, [image_size, image_size, 3])
        image = tf.cast(image, tf.float32) * (2. / 255) - 1.
        label = tf.cast(features['label'], tf.int32)
        return image, label
    image_path = tf.gfile.Glob(os.path.join(data_dir))

    files = tf.data.Dataset.from_tensor_slices(image_path)
    files = files.shuffle(image_num)

    dataset = files.interleave(lambda fn: tf.data.TFRecordDataset(fn).prefetch(batch), cycle_length = 1)
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(image_num)) \
                    .apply(tf.contrib.data.map_and_batch(getlabels, batch, num_parallel_batches = 16, drop_remainder = True))

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    return images, labels
      
def readfile(image_path, labels):
    labels = tf.strings.to_number(labels, tf.dtypes.int32)

    x = tf.read_file(image_path)
    x_decode = tf.image.decode_jpeg(x, channels = 3)
    img = tf.image.resize_images(x_decode, [64, 64])
    img = tf.cast(img, tf.float32) / 127.5 - 1
    return img, labels

def savefile(images, size, name = "sample.png"):
    transform = (images + 1.) / 2.
    return scipy.misc.imsave(name, merge(transform, size))

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    if (images.shape[3] in (3,4)):
        c = images.shape[3]
        img = np.zeros((h * size[0], w * size[1], c))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w, :] = image
        return img
    elif images.shape[3]==1:
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i = idx % size[1]
            j = idx // size[1]
            img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
        return img


