import tensorflow as tf
import numpy as np
import scipy.misc

def create_dataset(image_path, image_num, batch, gpu):   
    labels = [getlabels(lab) for lab in image_path]

    dataset = tf.data.Dataset.from_tensor_slices((image_path, labels))
    dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(image_num)) \
                    .apply(tf.contrib.data.map_and_batch(readfile, batch, num_parallel_batches = 16, drop_remainder = True)) \
                    .apply(tf.contrib.data.prefetch_to_device(gpu, batch))
    iterator = dataset.make_one_shot_iterator()
    return iterator
      
def readfile(image_path, labels):
    labels = tf.strings.to_number(labels, tf.dtypes.int32)

    x = tf.read_file(image_path)
    x_decode = tf.image.decode_jpeg(x, channels = 3)
    img = tf.image.resize_images(x_decode, [64, 64])
    img = tf.cast(img, tf.float32) / 127.5 - 1
    return img, labels

def getlabels(image_path):
    image_name = image_path.split('\\')[-1].split('_')[0]
    return image_name

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


