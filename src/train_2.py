import numpy as np
import tensorflow as tf
import random
import time

from librispeech import LibriSpeechDataset
from ReadData import ReadData
from utils import preprocess_instances, BatchPreProcessor

samplerate = 16000
batchAudioLength = 3.

n_seconds = 3
downsampling = 4
batchsize = 64
#clipNorm = 1.0

training_set = ['train-clean-360']
validation_set = 'dev-clean'

X_1 = tf.placeholder(tf.float32, [None, int(samplerate/downsampling*batchAudioLength)])
X_2 = tf.placeholder(tf.float32, [None, int(samplerate/downsampling*batchAudioLength)])
Y = tf.placeholder(tf.float32, [None, 1])

lr = tf.placeholder(tf.float32, [])

def ConvolutionalEncoder1D(input_x, filters=64):
    with tf.variable_scope('conv', reuse=tf.AUTO_REUSE):
        inputX = tf.reshape(input_x, shape=[-1, int(samplerate/downsampling*batchAudioLength), 1])

        conv1d_1 = tf.layers.conv1d(inputs=inputX, filters=filters, kernel_size=32, padding='same', activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer)
        conv1d_1 = tf.layers.batch_normalization(inputs=conv1d_1, training=True)
        conv1d_1 = tf.layers.max_pooling1d(inputs=conv1d_1, pool_size=4, strides=4) # 12000 * 32

        conv1d_2 = tf.layers.conv1d(inputs=conv1d_1, filters=filters*2, kernel_size=3, padding='same', activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer)
        conv1d_2 = tf.layers.batch_normalization(inputs=conv1d_2, training=True)
        conv1d_2 = tf.layers.max_pooling1d(inputs=conv1d_2, pool_size=2, strides=2) # 6000 * 64

        conv1d_3 = tf.layers.conv1d(inputs=conv1d_2, filters=filters*3, kernel_size=3, padding='same', activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer)
        conv1d_3 = tf.layers.batch_normalization(inputs=conv1d_3, training=True)
        conv1d_3 = tf.layers.max_pooling1d(inputs=conv1d_3, pool_size=2, strides=2) # 3000 * 96

        conv1d_4 = tf.layers.conv1d(inputs=conv1d_3, filters=filters*4, kernel_size=3, padding='same', activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer)
        conv1d_4 = tf.layers.batch_normalization(inputs=conv1d_4, training=True)
        conv1d_4 = tf.layers.max_pooling1d(inputs=conv1d_4, pool_size=2, strides=2) # 1500 * 128

        globalMaxPool = tf.math.reduce_max(conv1d_4, axis=1)

        dense = tf.layers.dense(globalMaxPool, units=64, kernel_initializer=tf.glorot_uniform_initializer)
        
    return dense

def Networks():
    encoded_1 = ConvolutionalEncoder1D(X_1)
    encoded_2 = ConvolutionalEncoder1D(X_2)

    subtracted = tf.math.subtract(encoded_1, encoded_2)
    distance = tf.math.sqrt(tf.reduce_sum(tf.math.square(subtracted), axis=-1, keepdims=True))

    logits = tf.layers.dense(inputs=distance, units=1, activation=tf.nn.sigmoid, kernel_initializer=tf.glorot_uniform_initializer)
    #cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=Y, logits=logits)
    #loss = tf.reduce_mean(cross_entropy)

    cross_entropy = tf.keras.losses.binary_crossentropy(y_true=Y, y_pred=logits, from_logits=True)
    loss = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer(lr)
    #optimizer = tf.train.AdadeltaOptimizer(lr)

    #gradients, variables = zip(*optimizer.compute_gradients(loss))
    #gradients, _ = tf.clip_by_global_norm(gradients, clipNorm)
    #train_op = optimizer.apply_gradients(zip(gradients, variables))

    #grad_and_vars = optimizer.compute_gradients(loss)
    #clipped_grads = [(tf.clip_by_value(grad, -1.0, 1.0), var) for grad, var in grad_and_vars]
    #train_op = optimizer.apply_gradients(clipped_grads)
    
    train_op = optimizer.minimize(loss)

    correct_pred = tf.equal(tf.round(logits), Y)
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    return train_op, loss, accuracy, logits

train_op, loss, accuracy, logits = Networks()

init = tf.global_variables_initializer()

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    #train = LibriSpeechDataset(training_set, n_seconds, pad=True)
    #valid = LibriSpeechDataset(validation_set, n_seconds, stochastic=False, pad=True)

    #batch_preprocessor = BatchPreProcessor('siamese', preprocess_instances(downsampling))

    batchData = ReadData(3., batchsize, fileNumperSpeaker=1)

    sess.run(init)

    #for  _ in range(200):
    #    lrVal = 0.01
    #    ([batch_x_1, batch_x_2], batch_y) = batch_preprocessor(train.build_verification_batch(batchsize))
    #
    #    lg, lo, _ = sess.run([logits, loss, train_op], {X_1: batch_x_1, X_2: batch_x_2, Y: batch_y, lr: lrVal})
    #
    #    print
    #    print(lg)
    #    print
    #    print(lo)
    #    print('min: {}    max: {}'.format(np.min(lg), np.max(lg)))
    #    print

    for epoch in range(50):
        lrVal = 0.005
        start = time.time()

        for i in range(300):
            #([batch_x_1, batch_x_2], batch_y) = batch_preprocessor(train.build_verification_batch(batchsize))
            batch_x_1, batch_x_2, batch_y = batchData.nextBatch()

            sess.run(train_op, {X_1: batch_x_1, X_2: batch_x_2, Y: batch_y, lr: lrVal})

            if lrVal > 0.005:
                lrVal *= 0.93
            else:
                lrVal *= 0.98

            if i % 50 == 0 and i != 0:
                #([batch_x_1, batch_x_2], batch_y) = batch_preprocessor(train.build_verification_batch(batchsize))
                batch_x_1, batch_x_2, batch_y = batchData.nextBatch()
                lg, lo, ac = sess.run([logits, loss, accuracy], {X_1: batch_x_1, X_2: batch_x_2, Y: batch_y})

                if epoch == 0 and i == 50:
                    print
                    print(lg)
                    print
                
                print('epoch {}-{}'.format(epoch, i))
                print('loss: {}'.format(lo))
                print('min: {}    max: {}'.format(np.min(lg), np.max(lg)))
                print('acc: {}\n'.format(ac))

                #([batch_x_1, batch_x_2], batch_y) = batch_preprocessor(valid.build_verification_batch(batchsize*2))
                #lo, ac = sess.run([loss, accuracy], {X_1: batch_x_1, X_2: batch_x_2, Y: batch_y})

                #print('valid loss: {}, valid acc: {}\n\n'.format(lo, ac))
        
        trainingTime = time.time() - start

        batch_x_1, batch_x_2, batch_y = batchData.nextBatch()
        lo, ac = sess.run([loss, accuracy], {X_1: batch_x_1, X_2: batch_x_2, Y: batch_y})

        print('epoch: {}'.format(epoch))
        print('training time: {}'.format(trainingTime))
        print('loss: {}, acc: {}\n\n'.format(lo, ac))
        print

        #loss_v = 0
        #acc_v = 0
        #for _ in range(10):
        #    ([batch_x_1, batch_x_2], batch_y) = batch_preprocessor(valid.build_verification_batch(batchsize*2))
        #    lo, ac = sess.run([loss, accuracy], {X_1: batch_x_1, X_2: batch_x_2, Y: batch_y})
#
        #    loss_v += lo
        #    acc_v += ac
#
        #print('epoch: {}'.format(epoch))
        #print('training time: {}'.format(trainingTime))
        #print('valid loss: {}, valid acc: {}\n\n'.format(loss_v/10, acc_v/10))
        #print
    
    #print
    #print
    #print('summary')
    #print
    #print
    #
    #for _ in range(10):
    #    ([batch_x_1, batch_x_2], batch_y) = batch_preprocessor(valid.build_verification_batch(batchsize*2))
    #    lo, ac = sess.run([loss, accuracy], {X_1: batch_x_1, X_2: batch_x_2, Y: batch_y})
    #    print('valid loss: {}, valid acc: {}'.format(lo, ac))
    
