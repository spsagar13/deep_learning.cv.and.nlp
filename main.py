#!/usr/bin/python
import tensorflow as tf
import os

from config import Config
from model import CaptionGenerator
from dataset import prepare_train_data, prepare_eval_data, prepare_test_data


# Eager execution provides an imperative interface to TensorFlow. With eager execution enabled, TensorFlow functions
# execute operations immediately (as opposed to adding to a graph to be executed later in a tf.compat.v1.Session) and
# return concrete values (as opposed to symbolic references to a node in a computational graph)
if not tf.executing_eagerly():
    tf.enable_eager_execution()

# Allow the program to continue to execute, to avoid static linking of the OpenMP runtime in any library.
# https://stackoverflow.com/questions/20554074/sklearn-omp-error-15-when-fitting-models
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('phase', 'train',
                       'The phase can be train, eval or test')

tf.flags.DEFINE_boolean('load', False,
                        'Turn on to load a pretrained model from either \
                        the latest checkpoint or a specified file')

tf.flags.DEFINE_string('model_file', None,
                       'If sepcified, load a pretrained model from this file')

tf.flags.DEFINE_boolean('load_cnn', False,
                        'Turn on to load a pretrained CNN model')

tf.flags.DEFINE_string('cnn_model_file', './vgg16_no_fc.npy',
                       'The file containing a pretrained CNN model')

tf.flags.DEFINE_boolean('train_cnn', False,
                        'Turn on to train both CNN and RNN. \
                         Otherwise, only RNN is trained')

tf.flags.DEFINE_integer('beam_size', 3,
                        'The size of beam search for caption generation')


def main(argv):
    #FLAGS.model_file = 'models/vgg-gen.npy'
    #FLAGS.model_file = 'models/vgg-ppl.npy'
    #FLAGS.beam_size = 3
    config = Config()
    config.phase = FLAGS.phase
    config.train_cnn = FLAGS.train_cnn
    config.beam_size = FLAGS.beam_size

    # FLAGS.phase = 'train'
    # FLAGS.model_file = "./resnet50_no_fc.npy"  # vgg16_no_fc.npy
    # FLAGS.load_cnn = 'load_cnn'
    # FLAGS.train_cnn = 'train_cnn'

    # FLAGS.model_file = 'models/vgg-ppl.npy'

    with tf.Session() as sess:
        if FLAGS.phase == 'train':
            # training phase
            data = prepare_train_data(config)
            model = CaptionGenerator(config)
            sess.run(tf.global_variables_initializer())
            if FLAGS.load:
                model.load(sess, FLAGS.model_file)
            if FLAGS.load_cnn:
                model.load_cnn(sess, FLAGS.cnn_model_file)
            tf.get_default_graph().finalize()
            model.train(sess, data)

        elif FLAGS.phase == 'eval':
            # evaluation phase
            coco, data, vocabulary = prepare_eval_data(config)
            model = CaptionGenerator(config)
            model.load(sess, FLAGS.model_file)
            tf.get_default_graph().finalize()
            model.eval(sess, coco, data, vocabulary)

        else:
            # testing phase
            data, vocabulary = prepare_test_data(config)
            model = CaptionGenerator(config)
            model.load(sess, FLAGS.model_file)
            tf.get_default_graph().finalize()
            model.test(sess, data, vocabulary)


if __name__ == '__main__':
    tf.app.run()
