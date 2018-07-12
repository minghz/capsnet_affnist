import os
import sys
import shutil
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from datetime import datetime

from config import cfg
from utils import load_data
from capsNet import CapsNet

RESULTS_DIR = cfg.results + '_' + cfg.centered + '_centered_' + cfg.peppered + '_peppered'
CHECKPOINT_DIR = cfg.checkpoint_dir + '_' + cfg.centered + '_centered_' + cfg.peppered + '_peppered'

def save_to():
    if not os.path.exists(RESULTS_DIR):
        os.mkdir(RESULTS_DIR)
    if cfg.is_training:
        loss = RESULTS_DIR + '/loss.csv'
        train_acc = RESULTS_DIR + '/train_acc.csv'
        val_acc = RESULTS_DIR + '/val_acc.csv'

        if os.path.exists(val_acc):
            os.remove(val_acc)
        if os.path.exists(loss):
            os.remove(loss)
        if os.path.exists(train_acc):
            os.remove(train_acc)

        fd_train_acc = open(train_acc, 'w')
        fd_train_acc.write('step,train_acc\n')
        fd_loss = open(loss, 'w')
        fd_loss.write('step,loss\n')
        fd_val_acc = open(val_acc, 'w')
        fd_val_acc.write('step,val_acc\n')
        return(fd_train_acc, fd_loss, fd_val_acc)
    else:
        test_acc = RESULTS_DIR + '/test_acc.csv'
        if os.path.exists(test_acc):
            os.remove(test_acc)
        fd_test_acc = open(test_acc, 'w')
        fd_test_acc.write('test_acc\n')
        return(fd_test_acc)


def prepare_output_dir():
    if os.path.exists(RESULTS_DIR):
        os.rename(RESULTS_DIR, RESULTS_DIR + datetime.now().isoformat())

    if os.path.exists(CHECKPOINT_DIR):
        shutil.rmtree(CHECKPOINT_DIR, CHECKPOINT_DIR + datetime.now().isoformat())

    if os.path.exists(cfg.logdir):
        shutil.rmtree(cfg.logdir)


def train(model, session):
    trX, trY, num_tr_batch, valX, valY, num_val_batch = load_data(cfg.batch_size, is_training=True)

    fd_train_acc, fd_loss, fd_val_acc = save_to()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with session as sess:
        print("\nNote: all of results will be saved to directory: " + RESULTS_DIR)
        for epoch in range(cfg.epoch):
            print("Training for epoch %d/%d:" % (epoch, cfg.epoch))
            if session.should_stop():
                print('session stoped!')
                break
            for step in tqdm(range(num_tr_batch), total=num_tr_batch, ncols=70, leave=False, unit='b'):
                start = step * cfg.batch_size
                end = start + cfg.batch_size
                global_step = epoch * num_tr_batch + step

                if global_step % cfg.train_sum_freq == 0:
                    _, loss, train_acc = sess.run(
                            [model.train_op, model.total_loss, model.accuracy])
                    assert not np.isnan(loss), 'Something wrong! loss is nan...'

                    fd_loss.write(str(global_step) + ',' + str(loss) + "\n")
                    fd_loss.flush()
                    fd_train_acc.write(str(global_step) + ',' + str(train_acc / cfg.batch_size) + "\n")
                    fd_train_acc.flush()
                else:
                    sess.run(model.train_op)

                if cfg.val_sum_freq != 0 and (global_step) % cfg.val_sum_freq == 0:
                    val_acc = 0
                    for i in range(num_val_batch):
                        start = i * cfg.batch_size
                        end = start + cfg.batch_size
                        acc = sess.run(
                                model.accuracy,
                                { model.X: valX[start:end],
                                  model.labels: valY[start:end] })
                        val_acc += acc
                    val_acc = val_acc / (cfg.batch_size * num_val_batch)
                    fd_val_acc.write(str(global_step) + ',' + str(val_acc) + '\n')
                    fd_val_acc.flush()


        fd_val_acc.close()
        fd_train_acc.close()
        fd_loss.close()


def evaluation(model, session, saver):
    teX, teY, num_te_batch = load_data(cfg.batch_size, is_training=False)
    fd_test_acc = save_to()
    with session as sess:
        saver.restore(sess, tf.train.latest_checkpoint(CHECKPOINT_DIR))
        tf.logging.info('Model restored!')

        test_acc = 0
        for i in tqdm(range(num_te_batch), total=num_te_batch, ncols=70, leave=False, unit='b'):
            start = i * cfg.batch_size
            end = start + cfg.batch_size
            acc = sess.run(model.accuracy,
                           {model.X: teX[start:end], model.labels: teY[start:end]})
            test_acc += acc
        test_acc = test_acc / (cfg.batch_size * num_te_batch)
        fd_test_acc.write(str(test_acc))
        fd_test_acc.close()
        print('Test accuracy has been saved to ' + RESULTS_DIR + '/test_acc.csv')


def main(_):
    if cfg.is_training:
        prepare_output_dir()

    graph = tf.Graph()
    with graph.as_default():
        tf.set_random_seed(1)
        model = CapsNet()
        saver = tf.train.Saver()

        if cfg.is_training:
            session = tf.train.MonitoredTrainingSession(
                hooks=[tf.train.NanTensorHook(model.total_loss),
                       tf.train.CheckpointSaverHook(checkpoint_dir=CHECKPOINT_DIR,
                                                    save_steps=cfg.save_checkpoint_steps,
                                                    saver=saver),
                       tf.train.SummarySaverHook(save_steps=cfg.train_sum_freq,
                                                 output_dir=cfg.logdir,
                                                 summary_op=model.train_summary)],
            )
            train(model, session)
        else:
            session = tf.train.MonitoredSession()
            evaluation(model, session, saver)


if __name__ == "__main__":
    tf.app.run()
