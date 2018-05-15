from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import re
import sys
import time
from datetime import datetime

from yolo.solver.solver import Solver

class YoloSolver(Solver):

    """Yolo Solver
    """
    def __init__(self, dataset, net, common_params, solver_params):
        self.image_size = int(common_params['image_size'])
        self.batch_size = int(common_params['batch_size'])
        self.max_objects = int(common_params['max_objects_per_image'])
        self.learning_rate = float(solver_params['learning_rate'])
        self.moment = float(solver_params['moment'])
        self.pretrain_path = str(solver_params['pretrain_model_path'])
        self.train_dir = str(solver_params['train_dir'])
        self.max_iterators = int(solver_params['max_iterators'])

        self.dataset = dataset
        self.net = net

        self.construct_graph()

    def _train(self):
        #学习率的优化方式
        opt = tf.train.MomentumOptimizer(self.learning_rate, self.moment)
        grads = opt.compute_gradients(self.total_loss)
        apply_gradient_op = opt.apply_gradients(grads, global_step=self.global_step)

        return apply_gradient_op

    def construct_graph(self):
        #constuct graph
        self.global_step = tf.Variable(0, trainable = False)
        self.images = tf.placeholder(tf.float32, (self.batch_size, self.image_size, self.image_size, 3))
        self.labels = tf.placeholder(tf.float32, (self.batch_size, self.max_objects, 5))
        self.objects_num = tf.placeholder(tf.int32, (self.batch_size))

        self.predicts = self.net.inference(self.images)
        #loss
        self.total_loss, self.nilboy = self.net.loss(self.predicts, self.labels, self.objects_num)

        tf.summary.scalar('loss', self.total_loss)
        #优化方式和参数
        self.train_op = self._train()

    def solve(self):
        saver1 = tf.train.Saver(self.net.pretrained_collection, write_version=tf.train.SaverDef.V2)
        #saver1 = tf.train.Saver(self.net.trainable_collection)
        saver2 = tf.train.Saver(self.net.trainable_collection, write_version=tf.train.SaverDef.V2)

        #init = tf.global_variables_initializer()
        init = tf.initialize_all_variables()
        summary_op = tf.summary.merge_all()

        with tf.Session() as sess:
            sess.run(init)
            saver1.restore(sess, self.pretrain_path)

            summary_writer = tf.summary.FileWriter(self.train_dir, sess.graph)

            for step in range(self.max_iterators):
                start_time = time.time()
                np_images, np_labels, np_objects_num = self.dataset.batch()
                #print(np_images.shape)
                #sess.run([self.trian_op],feed_dict={self.images: np_images, self.labels:np_labels, self.objects_num:np_objects_num})
                _, loss_value, nilboy = sess.run([self.train_op, self.total_loss, self.nilboy],
                                             feed_dict={self.images: np_images, self.labels: np_labels,
                                                        self.objects_num: np_objects_num})
                duration = time.time() - start_time

                #assert not np.isnan(loss_value),"loss = Nan"

                if step % 10 == 0:
                    num_examples_per_step = self.dataset.batch_size
                    examples_per_sec = num_examples_per_step / duration
                    sec_per_batch = float(duration)

                    format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                                  'sec/batch)')
                    print(format_str % (datetime.now(), step, loss_value,
                                        examples_per_sec, sec_per_batch))

                sys.stdout.flush()

                if step % 100 == 0:
                    summary_str = sess.run(summary_op, feed_dict={self.images: np_images, self.labels: np_labels,
                                                        self.objects_num: np_objects_num})
                    summary_writer.add_summary(summary_str, step)

                if step % 5000 == 0:
                    saver2.save(sess, self.train_dir + '/model.ckpt', global_step=step)
                    