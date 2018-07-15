import os
import tensorflow as tf
import numpy as np

from loadmat import loadmat
from config import cfg


TOTAL_TRAINING_IMAGES = 60000


def load_data(batch_size, is_training=True):
    if is_training:
        data_file = os.path.join(cfg.affnist_data_dir,
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
            data_file = os.path.join(cfg.affnist_data_dir, 'just_centered', 'test.mat')
        else:
            data_file = os.path.join(cfg.affnist_data_dir, 'transformed', 'test_batches', '15.mat')

        data = loadmat(data_file)
        images = data['affNISTdata']['image'].transpose().reshape(10000, 40, 40, 1).astype(np.float32)
        labels = data['affNISTdata']['label_int'].astype(np.float32)
        assert images.shape == (10000, 40, 40, 1)
        assert labels.shape == (10000,)

        imgs = images / 255.
        labs = labels
        num_te_batch = 10000 // cfg.batch_size

        return imgs, labs, num_te_batch


def get_batch_data(batch_size, num_threads):
    trX, trY, num_tr_batch, valX, valY, num_val_batch = load_data(batch_size, is_training=True)

    data_queues = tf.train.slice_input_producer([trX, trY])
    X, Y = tf.train.shuffle_batch(data_queues, num_threads=num_threads,
                                  batch_size=batch_size,
                                  capacity=batch_size * 64,
                                  min_after_dequeue=batch_size * 32,
                                  allow_smaller_final_batch=False)

    return(X, Y)


