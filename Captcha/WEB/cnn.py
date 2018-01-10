import tensorflow as tf
import numpy as np
from PIL import Image


def reshape_image(img_path):
    a = Image.open(img_path).convert('L')
    img1 = np.array(a.resize((48, 64)))
    image = np.reshape(img1, [-1, 3072])
    return image


def cnn(images, result = None):
    img = reshape_image(images)
    X = tf.placeholder(tf.float32, [None, 48, 64, 1])

    with tf.variable_scope('layer1'):
        W1 = tf.Variable(tf.truncated_normal([5, 5, 1, 16], stddev = 0.1)) 
        b1 = tf.Variable(tf.constant(0.0, shape=[16]))
        conv1 = tf.nn.conv2d(X, W1, strides = [1,1,1,1], padding = 'SAME')
        Y0 = tf.nn.relu(tf.nn.bias_add(conv1, b1))
        pool0 = tf.nn.max_pool(Y0,ksize=[1,2,2,1], strides=[1,2,2,1],padding = 'SAME')
        
        
    with tf.variable_scope('layer2'):
        W2 = tf.Variable(tf.truncated_normal([5,5,16,32], stddev = 0.1))
        b2 = tf.Variable(tf.constant(0.0, shape=[32]))
        conv2 = tf.nn.conv2d(Y0, W2, strides = [1,1,1,1], padding = 'SAME') 
        Y1 = tf.nn.relu(tf.nn.bias_add(conv2, b2))
        pool1 = tf.nn.max_pool(Y1,ksize=[1,2,2,1], strides=[1,2,2,1],padding = 'SAME')
        
        
    with tf.variable_scope('layer3'):
        W3 = tf.Variable(tf.truncated_normal([3, 5, 32, 48], stddev = 0.1))
        b3 = tf.Variable(tf.constant(0.0, shape = [48]))
        conv3 = tf.nn.conv2d(pool1, W3, strides = [1,1,1,1], padding = 'SAME')
        Y2 = tf.nn.relu(tf.nn.bias_add(conv3, b3))
        pool2 = tf.nn.max_pool(Y2,ksize=[1,2,2,1], strides=[1,2,2,1],padding = 'SAME')
        reshape = tf.reshape(pool2,[-1,12*16*48])
    #     pool_size = tf.shape(pool2)
    #     reshape = tf.reshape(pool2,[pool_size[0],-1])
    with tf.variable_scope('layer4'):
        fc1_weight = tf.Variable(tf.truncated_normal([12*16*48,512],stddev = 0.1))
        fc1_bias=tf.Variable(tf.constant(0.0,shape=[512]))
        fc1 = tf.nn.relu(tf.matmul(reshape,fc1_weight)+fc1_bias)
        
        
    with tf.variable_scope('layer5'):
        fc2_weight = tf.Variable(tf.truncated_normal([512,44], stddev = 0.1))
        fc2_bias=tf.Variable(tf.constant(0.0,shape=[44]))
        fc2 = tf.matmul(fc1,fc2_weight)+fc2_bias
        fc21 = tf.reshape(fc2,[-1,4,11])

    with tf.Session() as sess:
        saver = tf.train.Saver()
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, './static/model/model.ckpt')

        coord = tf.train.Coordinator()
        thread = tf.train.start_queue_runners(coord=coord)
        Res = sess.run(tf.argmax(fc21, 2), feed_dict={X: img})

        coord.request_stop()
        coord.join(thread)
        Res = list(Res[0])
        result = ''
        for i in Res:
            if i == 10:
                pass
            else:
                result += str(i)
    return result
