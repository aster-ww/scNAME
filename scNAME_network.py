#encoding: utf-8
import tensorflow.compat.v1 as tf

tf.enable_eager_execution()
from keras.layers import  Dense, Activation,GaussianNoise
from sklearn.cluster import KMeans
from scNAME_loss import *
import os
from sklearn.metrics import adjusted_rand_score
np.set_printoptions(threshold=np.inf)

def mask_generator(p_m, x):
    mask = np.random.binomial(1, p_m, x.shape)
    return mask

def pretext_generator(m, x):
    # Parameters
    no, dim = x.shape
    # Randomly (and column-wise) shuffle data
    x_bar = np.zeros([no, dim])
    for i in range(dim):
        idx = np.random.permutation(no)
        x_bar[:, i] = x[idx, i]

    # Corrupt samples
    x_tilde = x * (1 - m) + x_bar * m
    # Define new mask matrix
    m_new = 1 * (x != x_tilde)

    return m_new, x_tilde

class autoencoder(object):
    def __init__(self, dataname,n, batch_size,k,temperature, dims, cluster_num, alpha,beta, gamma,learning_rate,noise_sd=1.5, init='glorot_uniform', act='relu'):#
        self.dataname = dataname
        self.n = n
        self.batch_size=batch_size
        self.k=k
        self.temperature=temperature
        self.dims = dims
        self.cluster_num = cluster_num
        self.alpha = alpha
        self.beta=beta
        self.gamma = gamma
        #self.theta = theta
        self.learning_rate = learning_rate
        self.noise_sd = noise_sd
        self.init = init
        self.act = act


        self.n_stacks = len(self.dims) - 1
        self.sf_layer = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        self.x_origin = tf.placeholder(dtype=tf.float32, shape=(None, self.dims[0]))
        self.x_count = tf.placeholder(dtype=tf.float32, shape=(None, self.dims[0]))
        self.x_tilde = tf.placeholder(dtype=tf.float32, shape=(None, self.dims[0]))
        self.batch = tf.placeholder(dtype=tf.int32, shape=(None, 1))
        self.m = tf.placeholder(dtype=tf.float32, shape=(None, self.dims[0]))
        self.bank = tf.get_variable(name=self.dataname + "/bank",dtype=tf.float32, initializer=tf.zeros([self.n, self.dims[-1]]),trainable=False)
        self.clusters = tf.get_variable(name=self.dataname + "/clusters_rep", shape=[self.cluster_num, self.dims[-1]],
                                        dtype=tf.float32, initializer=tf.glorot_uniform_initializer())

        def encoder(x_input,  dims, init, act, n_stacks):
            h = x_input
            h = GaussianNoise(noise_sd, name='input_noise')(h)  
            for i in range(n_stacks - 1):
                h = Dense(units=dims[i + 1], kernel_initializer=init, name='encoder_%d' % i)(h) 
                h = GaussianNoise(noise_sd, name='noise_%d' % i)(h)
                h = Activation(act)(h) 
            latent = Dense(units=dims[-1], kernel_initializer=init, name='encoder_hidden')(h)
            h = latent
            for i in range(n_stacks - 1, 0, -1):
                h = Dense(units=dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(h)
            mask_estimator = Dense(units=dims[0], activation='sigmoid', kernel_initializer=init, name='mask_estimator')(h)
            pi = Dense(units=dims[0], activation='sigmoid', kernel_initializer=init, name='pi')(h)
            disp = Dense(units=dims[0], activation=DispAct, kernel_initializer=init, name='dispersion')(h)
            mean = Dense(units=dims[0], activation=MeanAct, kernel_initializer=init, name='mean')(h)
            return latent, mask_estimator, pi, disp,mean
       
        self.encoder_x_tilde, self.mask_estimator, _, _, _= encoder(self.x_tilde, self.dims,
                                                                    self.init, self.act,self.n_stacks)#,self.noise_sd
        self.encoder_x, _, self.pi, self.disp, self.mean= encoder(self.x_origin, self.dims,
                                                                  self.init, self.act, self.n_stacks)##,self.noise_sd
        self.output = self.mean * tf.matmul(self.sf_layer, tf.ones((1, self.mean.get_shape()[1]), dtype=tf.float32))
        
        self.bank=tf.scatter_nd_update(self.bank, self.batch, self.encoder_x)

        self.likelihood_loss = ZINB(self.pi, self.disp, self.x_count, self.output, ridge_lambda=1.0)

        self.mask_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(self.m, self.mask_estimator), 0)

        self.latent_idx, _ = cal_dist(self.bank, self.clusters)
     
        self.neighbor_loss = neighbor_k_loss(self.batch_size, self.k, self.bank, self.encoder_x, self.encoder_x_tilde, self.temperature)
        self.latent_dist1, self.latent_dist2 = cal_dist(self.encoder_x, self.clusters)
        self.kmeans_loss = tf.reduce_mean(tf.reduce_sum(self.latent_dist2, axis=1))
        self.total_loss = self.likelihood_loss+self.alpha* self.mask_loss + self.beta * self.neighbor_loss \
                          +self.gamma * self.kmeans_loss#+self.theta*self.reconstruct_loss

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.pretrain_op = self.optimizer.minimize(self.likelihood_loss+self.alpha* self.mask_loss)# +self.theta*self.reconstruct_loss)
        self.train_op = self.optimizer.minimize(self.total_loss)

    def pretrain(self, X, count_X,p_m,size_factor, batch_size, pretrain_epoch,gpu_option):
        print("begin the pretraining")
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_option
        config_ = tf.ConfigProto()
        config_.gpu_options.allow_growth = True
        config_.allow_soft_placement = True
        self.sess = tf.Session(config=config_)
        self.sess.run(init)
        pre_index = 0
        for ite in range(pretrain_epoch):
            while True:
                if (pre_index + 1) * batch_size > X.shape[0]:
                    last_index = np.array(list(range(pre_index * batch_size, X.shape[0])) + list(
                        range((pre_index + 1) * batch_size - X.shape[0])))
                    last_index_new=last_index.reshape([-1, 1])
                    # Mask vector generation
                    x_batch=X[last_index]
                    m_batch = mask_generator(p_m, x_batch)
                    # Pretext generator
                    m_batch_new, x_tilde_batch = pretext_generator(m_batch, x_batch)

                    _, mask_loss,likelihood_loss,latent,bank= self.sess.run([self.pretrain_op, self.mask_loss,self.likelihood_loss,self.encoder_x,self.bank],
                        feed_dict={self.x_origin: x_batch, self.x_count: count_X[last_index],self.batch:last_index_new,
                                   self.m : m_batch_new, self.x_tilde:x_tilde_batch, self.sf_layer: size_factor[last_index]})
                    if ite%10==0:
                       print('epoch: ' + str(ite) + '/' + str(pretrain_epoch) +
                          ', iteration: '+ str(pre_index) +
                          ', Current likelihood loss: ' + str(np.round(likelihood_loss, 8))+
                          ', Current mask loss: ' + str(np.round(mask_loss, 8)))

                    pre_index = 0
                    break
                else:
                    batch_index=np.arange(pre_index * batch_size,(pre_index + 1) * batch_size)
                    batch_index_new=batch_index.reshape([-1,1])
                    x_batch = X[batch_index]
                    m_batch = mask_generator(p_m, x_batch)
                    # Pretext generator
                    m_batch_new, x_tilde_batch = pretext_generator(m_batch, x_batch)
                    _, mask_loss, likelihood_loss,latent,bank = self.sess.run(
                        [self.pretrain_op, self.mask_loss,self.likelihood_loss,self.encoder_x,self.bank],
                        feed_dict={self.x_origin: x_batch, self.x_count: count_X[batch_index],self.batch:batch_index_new,
                                   self.m : m_batch_new, self.x_tilde:x_tilde_batch,self.sf_layer: size_factor[batch_index]})
                    self.bank_current=bank              
                    pre_index += 1


    def funetrain(self, dataname,X,Y, count_X, p_m,size_factor, batch_size, funetrain_epoch,update_epoch,error):
        kmeans = KMeans(n_clusters=self.cluster_num, init="k-means++")
        #self.latent_repre = np.nan_to_num(self.latent_repre)
        self.kmeans_pred = kmeans.fit_predict(self.bank_current)
        self.last_pred = np.copy(self.kmeans_pred)
        self.sess.run(tf.assign(self.clusters, kmeans.cluster_centers_))
        self.kmeans_ARI = np.around(adjusted_rand_score(Y, self.kmeans_pred), 5)
        self.Y_pred_best = self.kmeans_pred
        print("kmean_ARI:",self.kmeans_ARI)
        self.best_epoch = -1
        self.best_ite = -1
        print("begin the funetraining")

        fune_index = 0
        for i in range(1, funetrain_epoch + 1):

              while True:
                    if (fune_index + 1) * batch_size > X.shape[0]:
                        last_index = np.array(list(range(fune_index * batch_size, X.shape[0])) + list(
                            range((fune_index + 1) * batch_size - X.shape[0])))
                        last_index_new = last_index.reshape([-1, 1])
                        x_batch = X[last_index]
                        m_batch = mask_generator(p_m, x_batch)
                        # Pretext generator
                        m_batch_new, x_tilde_batch = pretext_generator(m_batch, x_batch)
                        _,mask_loss, likelihood_loss, neighbor_loss,kmeans_loss,total_loss,latent_idx_cur,latent,bank= self.sess.run(
                            [self.train_op, self.mask_loss, self.likelihood_loss, self.neighbor_loss,self.kmeans_loss,self.total_loss,self.latent_idx,self.encoder_x,self.bank],
                            feed_dict={self.x_origin: x_batch, self.x_count: count_X[last_index],self.batch:last_index_new,
                                   self.m : m_batch_new, self.x_tilde: x_tilde_batch,  self.sf_layer: size_factor[last_index]})
                   
                        self.Y_pred = np.argmin(latent_idx_cur, axis=1)
                        self.ARI = np.around(adjusted_rand_score(Y, self.Y_pred), 5)
                  
                        if (i - 1) % 10 == 0:
                            print('epoch: ' + str(i) + '/' + str(funetrain_epoch) +
                                  ', iteration: ' + str(fune_index) +
                                  ', Current likelihood loss: ' + str(np.round(likelihood_loss, 8)) +
                                  ', Current mask loss: ' + str(np.round(mask_loss, 8)) +                       
                                  ', Current neighbor loss: ' + str(np.round(neighbor_loss, 8)) +
                                  ', Current kmeans loss: ' + str(np.round(kmeans_loss, 8)) +
                                  ', Current total loss: ' + str(np.round(total_loss, 8)))
                            print("current clustering ARI:", self.ARI)
                        fune_index = 0
                        break
                    else:
                        batch_index = np.arange(fune_index * batch_size, (fune_index + 1) * batch_size)
                        batch_index_new = batch_index.reshape([-1, 1])
                        x_batch = X[batch_index]
                        m_batch = mask_generator(p_m, x_batch)
                        # Pretext generator
                        m_batch_new, x_tilde_batch = pretext_generator(m_batch, x_batch)
                        _,mask_loss, likelihood_loss, neighbor_loss,kmeans_loss,total_loss,latent_idx_cur,bank,latent=\
                            self.sess.run([self.train_op, self.mask_loss, self.likelihood_loss, self.neighbor_loss,
                                                        self.kmeans_loss,self.total_loss,self.latent_idx,self.bank,self.encoder_x],
                            feed_dict={self.x_origin: x_batch, self.x_count: count_X[batch_index],self.batch:batch_index_new,
                                self.m: m_batch_new, self.x_tilde: x_tilde_batch, self.sf_layer: size_factor[batch_index]})
                      
                        self.Y_pred = np.argmin(latent_idx_cur, axis=1)
                        self.ARI = np.around(adjusted_rand_score(Y, self.Y_pred), 5)
                               
                        fune_index += 1
              if i % update_epoch == 0:
                    current_error=np.sum(self.Y_pred != self.last_pred) / len(self.last_pred)
                    print("current_error:",current_error)
                    if  current_error < error:
                        break
                    else:
                        self.last_pred = self.Y_pred
        self.sess.close()
        return self.Y_pred
