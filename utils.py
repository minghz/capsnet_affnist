import os
import scipy
import numpy as np
import tensorflow as tf

import scipy.io as spio
from config import cfg
from PIL import Image


TOTAL_TRAINING_IMAGES = 60000


def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)


def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict


def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict


def load_affmnist(batch_size, is_training=True):
    if is_training:
        data_file = os.path.join(cfg.affmnist_data_dir,
                                 'peppered_training_and_validation_batches',
                                 cfg.centered + '_percent_centered_' + cfg.peppered + '_percent_transformed.mat')
        
        images_per_transformation = int((TOTAL_TRAINING_IMAGES * int(cfg.peppered)/100) / 32)
        num_base_img = int(TOTAL_TRAINING_IMAGES * int(cfg.centered)/100)
        num_inputs = images_per_transformation * 32 + num_base_img

        num_training = num_inputs * 84/100
        num_training_eval = num_inputs - num_training

        # NOTE: Assert we have the correct number of total inputs, as expected
        data = loadmat(data_file)
        images = data['affNISTdata']['image'].transpose().reshape(num_inputs, 40, 40, 1).astype(np.float32)
        labels = data['affNISTdata']['label_int'].astype(np.uint8)
        assert images.shape == (num_inputs, 40, 40, 1)
        assert labels.shape == (num_inputs,)

        trX = images[:num_training] / 255.
        trY = labels[:num_training]

        valX = images[num_training_eval:, ] / 255.
        valY = labels[num_training_eval:]

        num_tr_batch = num_training // cfg.batch_size
        num_val_batch = num_training_eval // cfg.batch_size

        return trX, trY, num_tr_batch, valX, valY, num_val_batch

    else:
        # NOTE: Swap those two lines below to get some basic transformed test
        if cfg.peppered == '0':
            data_file = os.path.join(cfg.affmnist_data_dir, 'just_centered', 'test.mat')
        else:
            data_file = os.path.join(cfg.affmnist_data_dir, 'transformed', 'test_batches', '15.mat')

        data = loadmat(data_file)
        images = data['affNISTdata']['image'].transpose().reshape(10000, 40, 40, 1).astype(np.float32)
        labels = data['affNISTdata']['label_int'].astype(np.float32)
        assert images.shape == (10000, 40, 40, 1)
        assert labels.shape == (10000,)

        imgs = images / 255.
        labs = labels
        num_te_batch = 10000 // cfg.batch_size

        return imgs, labs, num_te_batch


def load_data(batch_size, is_training=True, one_hot=False):
    return load_affmnist(batch_size, is_training)


def get_batch_data(batch_size, num_threads):
    trX, trY, num_tr_batch, valX, valY, num_val_batch = load_affmnist(batch_size, is_training=True)

    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=num_threads,
                                  batch_size=batch_size,
                                  capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32,
                                  allow_smaller_final_batch=False)

    return(X, Y)


def save_images(imgs, size, path):
    '''
    Args:
        imgs: [batch_size, image_height, image_width]
        size: a list with tow int elements, [image_height, image_width]
        path: the path to save images
    '''
    imgs = (imgs + 1.) / 2  # inverse_transform
    return(scipy.misc.imsave(path, mergeImgs(imgs, size)))


def mergeImgs(images, size):
    h, w = images.shape[1], images.shape[2]
    imgs = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        imgs[j * h:j * h + h, i * w:i * w + w, :] = image

    return imgs


# For version compatibility
def reduce_sum(input_tensor, axis=None, keepdims=False):
    try:
        return tf.reduce_sum(input_tensor, axis=axis, keepdims=keepdims)
    except:
        return tf.reduce_sum(input_tensor, axis=axis, keep_dims=keepdims)


# For version compatibility
def softmax(logits, axis=None):
    try:
        return tf.nn.softmax(logits, axis=axis)
    except:
        return tf.nn.softmax(logits, dim=axis)
