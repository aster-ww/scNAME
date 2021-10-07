#encoding: utf-8
import tensorflow.compat.v1 as tf
#tf.disable_eager_execution()
tf.enable_eager_execution()
from keras.layers import  Dense, Activation,GaussianNoise
from sklearn.cluster import KMeans
from loss import *
import os
from mask_estimation_utils import mask_generator, pretext_generator
from sklearn.metrics import adjusted_rand_score
np.set_printoptions(threshold=np.inf)
#import os
#print(os.getcwd())
#os.chdir("C:/Users/Reykjavik/PycharmProjects/VIME/VIME_softKmeans/VIME_ZINB")

bank= tf.Variable(tf.zeros([10, 5]),dtype=float)

#sess = tf.Session()
#sess.run(tf.global_variables_initializer())
#batch1=tf.constant([[0],[2]])
#batch2=tf.constant([[4],[5]])
#encode_x1=tf.constant([[1,2,3,4,5],[2,2,2,2,2]],dtype=float)
#encode_x2=tf.constant([[1,2,9,4,5],[2,3,2,4,2]],dtype=float)
#bank=tf.multiply(0.5,tf.tensor_scatter_nd_update(bank, batch1, encode_x1))+tf.multiply(0.5,bank)
#sess.run(bank)
#bank=tf.multiply(0.5,bank_old)+tf.multiply(0.5,bank)
#sess.run(tf.assign(bank,bank_new))
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
        #self.bank =tf.Variable(tf.zeros([n, self.dims[-1]]), trainable=False)
        self.clusters = tf.get_variable(name=self.dataname + "/clusters_rep", shape=[self.cluster_num, self.dims[-1]],
                                        dtype=tf.float32, initializer=tf.glorot_uniform_initializer())##均匀分布随机数

        def encoder(x_input,  dims, init, act, n_stacks):#, noise_sd
            h = x_input
            h = GaussianNoise(noise_sd, name='input_noise')(h)  ##为数据施加0均值，标准差为stddev的加性高斯噪声
            for i in range(n_stacks - 1):
                h = Dense(units=dims[i + 1], kernel_initializer=init, name='encoder_%d' % i)(h)  ##全连接层
                h = GaussianNoise(noise_sd, name='noise_%d' % i)(h)  # add Gaussian noise
                h = Activation(act)(h)  ##激活函数
            latent = Dense(units=dims[-1], kernel_initializer=init, name='encoder_hidden')(h)
            h = latent
            for i in range(n_stacks - 1, 0, -1):
                h = Dense(units=dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(h)
            mask_estimator = Dense(units=dims[0], activation='sigmoid', kernel_initializer=init, name='mask_estimator')(h)
            pi = Dense(units=dims[0], activation='sigmoid', kernel_initializer=init, name='pi')(h)
            disp = Dense(units=dims[0], activation=DispAct, kernel_initializer=init, name='dispersion')(h)
            mean = Dense(units=dims[0], activation=MeanAct, kernel_initializer=init, name='mean')(h)
            return latent, mask_estimator, pi, disp,mean
        #self.encoder_x_tilde, self.mask_estimator, _,_,self.mean = encoder(self.x_tilde,self.distribution,self.dims,
        #                                                   self.noise_sd,self.init,self.act,self.n_stacks)
        #self.encoder_x, _,self.pi, self.disp, _= encoder(self.x_origin, self.distribution,self.dims,
        #                                                  self.noise_sd, self.init, self.act, self.n_stacks)
        self.encoder_x_tilde, self.mask_estimator, _, _, _= encoder(self.x_tilde, self.dims,
                                                                    self.init, self.act,self.n_stacks)#,self.noise_sd
        self.encoder_x, _, self.pi, self.disp, self.mean= encoder(self.x_origin, self.dims,
                                                                  self.init, self.act, self.n_stacks)##,self.noise_sd
        self.output = self.mean * tf.matmul(self.sf_layer, tf.ones((1, self.mean.get_shape()[1]), dtype=tf.float32))
        #self.output=self.mean
        #self.bank = tf.multiply(self.moving_rate,tf.tensor_scatter_nd_update(self.bank, self.batch, self.encoder_x)) + tf.multiply(1 - self.moving_rate, self.bank)
        self.bank=tf.scatter_nd_update(self.bank, self.batch, self.encoder_x)

        self.likelihood_loss = ZINB(self.pi, self.disp, self.x_count, self.output, ridge_lambda=1.0)

        self.mask_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(self.m, self.mask_estimator), 0)

        self.latent_idx, _ = cal_dist(self.bank, self.clusters)
        #self.neighbor_loss = neighbor_k_original_loss(self.batch_size, self.k, self.bank, self.encoder_x,self.temperature)
        self.neighbor_loss = neighbor_k_loss(self.batch_size, self.k, self.bank, self.encoder_x, self.encoder_x_tilde, self.temperature)
        self.latent_dist1, self.latent_dist2 = cal_dist(self.encoder_x, self.clusters)
        self.kmeans_loss = tf.reduce_mean(tf.reduce_sum(self.latent_dist2, axis=1))
        #self.reconstruct_loss=tf.losses.mean_squared_error(self.encoder_x, self.encoder_x_tilde)
        #self.bank_normal = self.bank / tf.reshape(tf.norm(self.bank, axis=1), [-1, 1])

        #self.encoder_x_normal = self.encoder_x / tf.reshape(tf.norm(self.encoder_x, axis=1), [-1, 1])
        #self.encoder_x_tilde_normal = self.encoder_x_tilde / tf.reshape(tf.norm(self.encoder_x_tilde, axis=1), [-1, 1])
        #self.neighbor_k = neighbor_k_original(self.k, self.bank_normal, self.encoder_x_normal)
        #self.distance_k = distance_k_mask(self.batch_size, self.k, self.bank_normal, self.encoder_x_normal, self.encoder_x_tilde_normal)

        #self.num = numerator(self.batch_size, self.k, self.bank_normal, self.encoder_x_normal, self.encoder_x_tilde_normal, self.temperature)
        #self.den0= tf.matmul(self.encoder_x_tilde_normal, tf.transpose(self.bank_normal))
        #self.den1= tf.exp(self.den0 / self.temperature)
        #self.den2=tf.reduce_sum(self.den1, 1)
        #self.den = denominator(self.bank_normal, self.encoder_x_tilde_normal, self.temperature)
        #self.loss1 = tf.log(self.num / self.den)  ## batchsize*1
        #self.loss2 = -tf.reduce_sum(self.loss1, [0])

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

        # self.bank_new = self.moving_rate*tf.scatter_nd_update(self.bank, self.batch, self.encoder_x) + (1-self.moving_rate)* self.bank
        # self.bank = self.bank_new
        #self.latent_repre = np.zeros((X.shape[0], self.dims[-1]))
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
                    #if ite==pretrain_epoch-1:
                        #print("latent:", latent.shape,latent[-10:-1,:5])
                        #print("index:",last_index_new)
                        #self.sess.run(tf.tensor_scatter_nd_update(self.bank, last_index_new, latent))
                        #self.bank = tf.multiply(moving_rate, tf.tensor_scatter_nd_update(self.bank, last_index_new, latent)) + tf.multiply(1-moving_rate,self.bank)
                        #self.sess.run(self.bank)
                        #print("bank:",self.sess.run(self.bank)[-10:-1, :5])
                    #self.latent_repre[last_index] = latent
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
                    #if ite == pretrain_epoch - 1:
                        #print("latent:",latent[-10:-1,:5])
                        #self.sess.run(tf.tensor_scatter_nd_update(self.bank, batch_index_new, latent))
                        #self.bank = tf.multiply(moving_rate, tf.tensor_scatter_nd_update(self.bank, batch_index_new,latent)) + tf.multiply( 1 - moving_rate, self.bank)
                        #self.sess.run(self.bank)
                        #print("bank:",self.sess.run(self.bank)[-10:-1, :5])
                    #self.latent_repre[batch_index] = latent
                    #print("latent:", latent[:10, :10])
                    #print("bank:", bank[batch_index[:10], :10])
                    #print('epoch: ' + str(ite) + '/' + str(pretrain_epoch) +
                    #      ', iteration: ' + str(pre_index) +
                    #      ', Current likelihood loss: ' + str(np.round(likelihood_loss, 8)) +
                    #      ', Current mask loss: ' + str(np.round(mask_loss, 8)))
                    #print('epoch: ' + str(ite) + '/' + str(pretrain_epoch) +
                    #      ', iteration: ' + str(pre_index) +
                    #      ', Current likelihood loss: ' + str(np.round(likelihood_loss, 8))+
                    #      ', Current mask loss: ' + str(np.round(mask_loss, 8)))
                    #+      ', Current reconstruct_loss: ' + str(np.round(reconstruct_loss, 8)))
                    #print("latent:",latent[:10,:10])
                    #print("bank:",bank[batch_index[:10],:10])
                    pre_index += 1

                    #print("index:",batch_index)

    def funetrain(self, dataname,X,Y, count_X, p_m,size_factor, batch_size, funetrain_epoch,update_epoch,error):##基于预训练模型优化 微调
        kmeans = KMeans(n_clusters=self.cluster_num, init="k-means++")
        #self.latent_repre = np.nan_to_num(self.latent_repre)##使用0代替数组x中的nan元素，使用有限的数字代替inf元素(默认行为)
        self.kmeans_pred = kmeans.fit_predict(self.bank_current)
        self.last_pred = np.copy(self.kmeans_pred)
        self.sess.run(tf.assign(self.clusters, kmeans.cluster_centers_))##把A的值变为new_number
        self.kmeans_ARI = np.around(adjusted_rand_score(Y, self.kmeans_pred), 5)
        self.Y_pred_best = self.kmeans_pred
        print("kmean_ARI:",self.kmeans_ARI)
        self.ARI_max = self.kmeans_ARI
        self.best_epoch = -1
        self.best_ite = -1
        print("begin the funetraining")

        fune_index = 0
        for i in range(1, funetrain_epoch + 1):

              while True:##如果出现错误的话，可以继续循环
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
                        #self.bank = tf.multiply(moving_rate, tf.tensor_scatter_nd_update(self.bank, last_index_new,latent)) + tf.multiply(1 - moving_rate, self.bank)
                        #print("bank:",self.sess.run(self.bank)[-10:-1, :5])
                        #self.sess.run(self.bank)

                        self.Y_pred = np.argmin(latent_idx_cur, axis=1)
                        self.ARI = np.around(adjusted_rand_score(Y, self.Y_pred), 5)

                        #print("latent:", latent[:10, :10])
                        #print("bank:",bank[last_index[:10],:10])
                        # Best model save
                        if self.ARI > self.ARI_max:
                            self.ARI_max = self.ARI
                            self.Y_pred_best = self.Y_pred
                            self.best_epoch = i
                            self.best_ite = fune_index

                        if (i - 1) % 10 == 0:
                            print('epoch: ' + str(i) + '/' + str(funetrain_epoch) +
                                  ', iteration: ' + str(fune_index) +
                                  ', Current likelihood loss: ' + str(np.round(likelihood_loss, 8)) +
                                  ', Current mask loss: ' + str(np.round(mask_loss, 8)) +
                                  # ', Current reconstruct loss: ' + str(np.round(reconstruct_loss, 8)) +
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
                        #self.bank = tf.multiply(moving_rate, tf.tensor_scatter_nd_update(self.bank, batch_index_new,latent)) + tf.multiply(1 - moving_rate, self.bank)
                        #print("bank:",self.sess.run(self.bank)[-10:-1, :5]
                        #self.sess.run(self.bank)
                        #print("latent:",latent[:10,:10])
                        #print("bank:",bank[batch_index[:10],:10])
                        #print("bank_normal:",bank_normal[:10,:10])


                        self.Y_pred = np.argmin(latent_idx_cur, axis=1)
                        self.ARI = np.around(adjusted_rand_score(Y, self.Y_pred), 5)

                        #print("index", batch_index)
                        #print("bank", bank.shape, bank[-10:-1, :])
                        #print("bank_normal", bank_normal.shape, bank_normal[-1, :])
                        #print("x_tilde:",x_tilde_batch.shape,x_tilde_batch[:5, :50])
                        #print("latent:", latent.shape, latent[:5, :50])
                        #print("encoder_x_tilde:",encoder_x_tilde.shape,encoder_x_tilde[0,:])
                        #print("encoder_x_normal:", encoder_x_tilde_normal.shape, encoder_x_tilde_normal[0, :])
                        #print("encoder_x_tilde_normal:", encoder_x_tilde_normal.shape, encoder_x_tilde_normal[0, :])
                        #print("neighbor_k:",neighbor_k.shape,neighbor_k[0,:])
                        #print("distance_k",distance_k.shape,distance_k[:5,:50])
                        #print("num", num.shape, num)
                        #print("den0", den0.shape, den0[0, :])
                        #print("den1", den1.shape, den1[0,:])
                        #print("den2", den2.shape, den2)
                        #print("den", den.shape, den)
                        #print("loss1:",loss1)
                        #print("loss2:",loss2)
                        # Early stopping & Best model save
                        if self.ARI > self.ARI_max:
                            self.ARI_max = self.ARI
                            self.Y_pred_best = self.Y_pred
                            self.best_epoch = i
                            self.best_ite = fune_index
                        #print('epoch: ' + str(i) + '/' + str(funetrain_epoch) +
                        #       ', iteration: ' + str(fune_index) +
                        #      ', Current likelihood loss: ' + str(np.round(likelihood_loss, 8)) +
                        #      ', Current mask loss: ' + str(np.round(mask_loss, 8))+
                              #', Current reconstruct loss: ' + str(np.round(reconstruct_loss, 8)) +
                        #      ', Current neighbor loss: ' + str(np.round(neighbor_loss, 8)) +
                        #      ', Current kmeans loss: ' + str(np.round(kmeans_loss, 8)) +
                        #      ', Current total loss: ' + str(np.round(total_loss, 8)))
                        fune_index += 1
              if i % update_epoch == 0:
                    current_error=np.sum(self.Y_pred != self.last_pred) / len(self.last_pred)
                    print("current_error:",current_error)
                    if  current_error < error:
                        break
                    else:
                        self.last_pred = self.Y_pred
        #print('best performace is in epoch: ' + str(self.best_epoch) + ', iteration: '+ str(self.best_ite))
        print("best_ARI:",self.ARI_max)
        #np.savetxt('{}_latent_scNAME.csv'.format(dataname), bank, delimiter=',')
        #np.savetxt('{}_pred.csv'.format(dataname), self.Y_pred, delimiter=',')
        #self.Y_pred=self.Y_pred_best
        self.sess.close()
        return self.Y_pred
