import tensorflow as tf
import tensorflow.contrib as tf_contrib

weight_init = tf_contrib.layers.variance_scaling_initializer() # kaming init for encoder / decoder
weight_regularizer = tf_contrib.layers.l2_regularizer(scale=0.0001)


def conv(x, channels, kernel=4, stride=2, pad=0, pad_type='zero', use_bias=True, scope='conv'):
    with tf.variable_scope(scope):
        if scope.__contains__("discriminator") :
            weight_init = tf.random_normal_initializer(mean=0.0, stddev=0.02)
        else :
            weight_init = tf_contrib.layers.variance_scaling_initializer()

        if pad_type == 'zero' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
        if pad_type == 'reflect' :
            x = tf.pad(x, [[0, 0], [pad, pad], [pad, pad], [0, 0]], mode='REFLECT')

        x = tf.layers.conv2d(inputs=x, filters=channels,
                             kernel_size=kernel, kernel_initializer=weight_init,
                             kernel_regularizer=weight_regularizer,
                             strides=stride, use_bias=use_bias)

        return x

def linear(x, units, use_bias=True, scope='linear'):
    with tf.variable_scope(scope):
        x = flatten(x)
        x = tf.layers.dense(x, units=units, kernel_initializer=weight_init, kernel_regularizer=weight_regularizer, use_bias=use_bias)

        return x

def down_sample(x) :
    return tf.layers.average_pooling2d(x, pool_size=3, strides=2, padding='SAME')

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [scale_factor * h, scale_factor * w]
    return tf.image.resize_nearest_neighbor(x, size=new_size)


def resblock(x_init, channels, use_bias=True, scope='resblock'):
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = instance_norm(x)

        return x + x_init

def adaptive_resblock(x_init, channels, mu, sigma, use_bias=True, scope='adaptive_resblock') :
    with tf.variable_scope(scope):
        with tf.variable_scope('res1'):
            x = conv(x_init, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = adain(x, mu, sigma)
            x = relu(x)

        with tf.variable_scope('res2'):
            x = conv(x, channels, kernel=3, stride=1, pad=1, pad_type='reflect', use_bias=use_bias)
            x = adain(x, mu, sigma)

        return x + x_init


def adaptive_avg_pooling(x):
    """
    Incoming Tensor shape must be 4-D
    """
    _, _, _, c = x.get_shape().as_list()
    gap = tf.reduce_mean(x, axis=[1, 2])
    gap = tf.reshape(gap, shape=[-1, 1, 1, c])

    return gap


def flatten(x) :
    return tf.layers.flatten(x)



def lrelu(x, alpha=0.01):
    # pytorch alpha is 0.01
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)

def adain(content, gamma, beta, epsilon=1e-5):
    # gamma, beta = style_mean, style_std from MLP

    c_mean, c_var = tf.nn.moments(content, axes=[1, 2], keep_dims=True)
    c_std = tf.sqrt(c_var + epsilon)

    return gamma * ((content - c_mean) / c_std) + beta


def instance_norm(x, scope='instance_norm'):
    return tf_contrib.layers.instance_norm(x,
                                           epsilon=1e-05,
                                           center=True, scale=True,
                                           scope=scope)

def layer_norm(x, scope='layer_norm') :
    return tf_contrib.layers.layer_norm(x,
                                        center=True, scale=True,
                                        scope=scope)

def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))

    return loss

def discriminator_loss(real, fake):
    n_scale = len(real)
    loss = []

    for i in range(n_scale) :
        real_loss = tf.reduce_mean(tf.squared_difference(real[i], 1.0))
        fake_loss = tf.reduce_mean(tf.square(fake[i]))
        loss.append((real_loss + fake_loss) * 0.5)

    return sum(loss)


def generator_loss(fake):
    n_scale = len(fake)
    loss = []

    for i in range(n_scale) :
        loss.append(tf.reduce_mean(tf.squared_difference(fake[i], 1.0)) * 0.5)


    return sum(loss)

