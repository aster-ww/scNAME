import tensorflow.compat.v1 as tf
tf.disable_eager_execution()
import keras.backend as K
import numpy as np

MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)

def _nan2zero(x):
    return tf.where(tf.is_nan(x), tf.zeros_like(x), x)

def _nan2inf(x):
    return tf.where(tf.is_nan(x), tf.zeros_like(x)+np.inf, x)

def _nelem(x):
    nelem = tf.reduce_sum(tf.cast(~tf.is_nan(x), tf.float32))
    return tf.cast(tf.where(tf.equal(nelem, 0.), 1., nelem), x.dtype)

def _reduce_mean(x):
    nelem = _nelem(x)
    x = _nan2zero(x)
    return tf.divide(tf.reduce_sum(x), nelem)

def neighbor_k_original(k,bank,z):
    result=tf.matmul(z, tf.transpose(bank))
    result=tf.nn.top_k(result, k=k)[1]
    #print('neighbor_k_original:',result)
    return result

def distance_k_mask(batchsize,k,bank,z,z_tilde):
    h_index =tf.reshape(neighbor_k_original(k, bank, z),[-1,1])
    h_index=tf.cast(h_index, dtype=tf.int32)
    line = tf.reshape(tf.repeat(np.arange(batchsize), k),[-1,1])
    line = tf.cast(line, dtype=tf.int32)
    index = tf.stack([line, h_index],1)
    index=tf.reshape(index,[k*batchsize,2])

    distance=tf.matmul(z_tilde, tf.transpose(bank))
    result = tf.gather_nd(distance, index)
    result = tf.reshape(result, [-1, k])
    return result
def neighbor_k_original_loss(batchsize,k,bank,z,temperature):
    bank_normal = bank / tf.reshape(tf.norm(bank, axis=1), [-1, 1])
    z_normal = z / tf.reshape(tf.norm(z, axis=1), [-1, 1])
    result = tf.matmul(z_normal, tf.transpose(bank_normal)) 
    result = tf.nn.top_k(result, k=k)[0]
    numerator =tf.exp(result / temperature)
    numerator = tf.reduce_sum(numerator,1) 
    denominator = tf.exp(tf.matmul(z_normal, tf.transpose(bank_normal)) / temperature) 
    denominator = tf.reduce_sum(denominator, 1) 
    loss = tf.log(numerator / denominator) 
    loss = -tf.reduce_sum(loss, [0])
    return loss /batchsize
def neighbor_k_loss(batchsize,k,bank,z,z_tilde,temperature):
    bank_normal = bank / tf.reshape(tf.norm(bank, axis=1), [-1, 1])
    z_normal = z / tf.reshape(tf.norm(z, axis=1), [-1, 1])
    z_tilde_normal = z_tilde / tf.reshape(tf.norm(z_tilde, axis=1), [-1, 1])
    numerator =tf.exp(distance_k_mask(batchsize, k, bank_normal, z_normal, z_tilde_normal) / temperature)
    numerator = tf.reduce_sum(numerator,1) 
    denominator = tf.exp(tf.matmul(z_tilde_normal, tf.transpose(bank_normal)) / temperature) 
    denominator = tf.reduce_sum(denominator, 1) 
    loss = tf.log(numerator / denominator) 
    loss = -tf.reduce_sum(loss, [0])
    return loss /batchsize

def numerator(batchsize, k, bank, z, z_tilde,temperature):
    numerator = tf.exp(distance_k_mask(batchsize, k, bank, z, z_tilde) / temperature)
    numerator = tf.reduce_sum(numerator, 1) 
    return numerator

def denominator(bank,z_tilde,temperature):
    denominator = tf.exp(tf.matmul(z_tilde, tf.transpose(bank)) / temperature) 
    denominator = tf.reduce_sum(denominator, 1) 
    return denominator

def NB(theta, y_true, y_pred, mask = False, debug = False, mean = False):##y_pred:mu? y_true:x
    eps = 1e-10
    scale_factor = 1.0
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32) * scale_factor
    if mask:
        nelem = _nelem(y_true)
        y_true = _nan2zero(y_true)
    theta = tf.minimum(theta, 1e6)
    t1 = tf.lgamma(theta + eps) + tf.lgamma(y_true + 1.0) - tf.lgamma(y_true + theta + eps)
    t2 = (theta + y_true) * tf.log(1.0 + (y_pred / (theta + eps))) + (y_true * (tf.log(theta + eps) - tf.log(y_pred + eps)))
    if debug:
        assert_ops = [tf.verify_tensor_all_finite(y_pred, 'y_pred has inf/nans'),
                      tf.verify_tensor_all_finite(t1, 't1 has inf/nans'),
                      tf.verify_tensor_all_finite(t2, 't2 has inf/nans')]
        with tf.control_dependencies(assert_ops):
            final = t1 + t2
    else:
        final = t1 + t2
    final = _nan2inf(final)
    if mean:
        if mask:
            final = tf.divide(tf.reduce_sum(final), nelem)
        else:
            final = tf.reduce_mean(final)
    return final

def ZINB(pi, theta, y_true, y_pred, ridge_lambda, mean = True, mask = False, debug = False):
    eps = 1e-10
    scale_factor = 1.0
    nb_case = NB(theta, y_true, y_pred, mean=False, debug=debug) - tf.log(1.0 - pi + eps)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32) * scale_factor
    theta = tf.minimum(theta, 1e6)

    zero_nb = tf.pow(theta / (theta + y_pred + eps), theta)
    zero_case = -tf.log(pi + ((1.0 - pi) * zero_nb) + eps)
    result = tf.where(tf.less(y_true, 1e-8), zero_case, nb_case)
    ridge = ridge_lambda * tf.square(pi)
    result += ridge
    if mean:
        if mask:
            result = _reduce_mean(result)
        else:
            result = tf.reduce_mean(result)

    result = _nan2inf(result)
    return result

def cal_latent(hidden, alpha):
    sum_y = K.sum(K.square(hidden), axis=1)
    num = -2.0 * tf.matmul(hidden, tf.transpose(hidden)) + tf.reshape(sum_y, [-1, 1]) + sum_y
    num = num / alpha
    num = tf.pow(1.0 + num, -(alpha + 1.0) / 2.0)
    zerodiag_num = num - tf.linalg.diag(tf.linalg.diag_part(num))
    latent_p = K.transpose(K.transpose(zerodiag_num) / K.sum(zerodiag_num, axis=1))
    return num, latent_p

def target_dis(latent_p):
    latent_q = tf.transpose(tf.transpose(tf.pow(latent_p, 2)) / tf.reduce_sum(latent_p, axis = 1))
    return tf.transpose(tf.transpose(latent_q) / tf.reduce_sum(latent_q, axis = 1))

def cal_dist(hidden, clusters):
    dist1 = K.sum(K.square(K.expand_dims(hidden, axis=1) - clusters), axis=2)
    temp_dist1 = dist1 - tf.reshape(tf.reduce_min(dist1, axis=1), [-1, 1])
    q = K.exp(-temp_dist1)
    q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
    q = K.pow(q, 2)###inflation=2
    q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
    dist2 = dist1 * q
    return dist1, dist2

def cross_entropy_dec(hidden, cluster, alpha = 1.0):
    dist = K.sum(K.square(K.expand_dims(hidden, axis=1) - cluster), axis=2)
    q = 1.0 / (1.0 + dist / alpha) ** ((alpha + 1.0) / 2.0)
    q = q / tf.reduce_sum(q, axis=1, keepdims=True)
    p = q ** 2 / tf.reduce_sum(q, axis=0)
    p = p / tf.reduce_sum(p, axis=1, keepdims=True)
    crossentropy = -p * tf.log(tf.clip_by_value(q, 1e-10, 1.0)) #- (1 - p) * tf.log(tf.clip_by_value(1 - q, 1e-10, 1.0))
    return dist, crossentropy

def dec(hidden, cluster, alpha = 1.0):
    dist = K.sum(K.square(K.expand_dims(hidden, axis=1) - cluster), axis=2)
    q = 1.0 / (1.0 + dist / alpha) ** ((alpha + 1.0) / 2.0)
    q = q / tf.reduce_sum(q, axis=1, keepdims=True)
    p = q ** 2 / tf.reduce_sum(q, axis=0)
    p = p / tf.reduce_sum(p, axis=1, keepdims=True)
    kl = p * tf.log(tf.clip_by_value(p, 1e-10, 1.0)) - p * tf.log(tf.clip_by_value(q, 1e-10, 1.0))
    return dist, kl
