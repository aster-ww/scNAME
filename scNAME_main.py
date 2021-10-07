from preprocess1 import *
from network import *
from utils1 import *
import argparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 5000)
#import inspect
def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size##测试condition，如为false，那么raise一个AssertionError
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w) ##01规划
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

if __name__ == "__main__":
    random_seed = [1111]#,2222,3333]#,2222,3333,4444, 5555]#, 6666, 7777, 8888, 9999, 10000]
    #random_seed = [8888, 9999, 10000]
    ##添加并管理参数
    parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataname", default = "Pollen", type = str)
    parser.add_argument('--p_m',help='corruption probability for self-supervised learning',default=0.05, type=float)
    parser.add_argument('--k',help='top k most similar features to be neighborhoods',default=10,type=int)#10
    parser.add_argument('--temperature',help='hyper-parameter in l3 loss', default=0.7,type=float)
    parser.add_argument("--dims", default = [1000, 256, 64,32])
    parser.add_argument("--highly_genes", default = 1000)
    parser.add_argument("--alpha",help='hyper-parameter to control the weights of l2 loss', default = 0.1,type = float)#0.1
    parser.add_argument("--beta",help='hyper-parameter to control the weights of l3 loss', default=0, type=float)#2.0
    parser.add_argument("--gamma",help='hyper-parameter to control the weights of l4 loss', default =0.01, type = float)#1.0
    parser.add_argument("--learning_rate", default = 0.0001, type = float)
    parser.add_argument("--random_seed", default = random_seed)
    parser.add_argument("--batch_size", default = 128, type = int)
    parser.add_argument("--pretrain_epoch", default = 500, type = int)#500
    parser.add_argument("--funetrain_epoch", default = 1000, type = int)#100
    parser.add_argument("--update_epoch", default=50, type=int)  # 100
    parser.add_argument("--noise_sd", default = 1.5)
    parser.add_argument("--error", default = 0.001, type = float)#0.001
    parser.add_argument("--gpu_option", default ="2,3" )#"0"

    args = parser.parse_args()
    X, Y = prepro(args.dataname)
    X = np.ceil(X).astype(int)
    count_X = X

    adata = sc.AnnData(X)
    adata.obs['Group'] = Y
    adata = normalize(adata, copy=True, highly_genes=args.highly_genes, size_factors=True,
                     normalize_input=True, logtrans_input=True)
    X = adata.X.astype(np.float32)
    Y = np.array(adata.obs["Group"])
    n = X.shape[0]
    high_variable = np.array(adata.var.highly_variable.index, dtype=np.int)  ##表达基因的序号
    count_X = count_X[:, high_variable]  ###count_X是计数矩阵，保留前500个
    size_factor = np.array(adata.obs.size_factors).reshape(-1, 1).astype(np.float32)
    cluster_number = int(max(Y) - min(Y) + 1)
    result = np.zeros([len(args.random_seed) + 3, 8])
    idx = 0
    for i in args.random_seed:
        np.random.seed([i])
        print("the dataset {} has {} samples".format(args.dataname, n))
        tf.reset_default_graph()  ##清除默认图形堆栈并重置全局默认图形
        scNAMEcluster = autoencoder(args.dataname, n, args.batch_size, args.k, args.temperature,
                                 args.dims, cluster_number,
                                 args.alpha, args.beta, args.gamma, args.learning_rate, args.noise_sd)
        scNAMEcluster.pretrain(X, count_X, args.p_m, size_factor, args.batch_size, args.pretrain_epoch,
                                     args.gpu_option)

        scNAMEcluster.funetrain(args.dataname,X, Y, count_X, args.p_m, size_factor, args.batch_size, args.funetrain_epoch,
                                      args.update_epoch,args.error)
        kmeans_accuracy = np.around(cluster_acc(Y, scNAMEcluster.kmeans_pred), 5)
        kmeans_ARI = np.around(adjusted_rand_score(Y, scNAMEcluster.kmeans_pred), 5)
        kmeans_NMI = np.around(normalized_mutual_info_score(Y, scNAMEcluster.kmeans_pred), 5)
        accuracy = np.around(cluster_acc(Y, scNAMEcluster.Y_pred), 5)
        ARI = np.around(adjusted_rand_score(Y, scNAMEcluster.Y_pred), 5)
        NMI = np.around(normalized_mutual_info_score(Y, scNAMEcluster.Y_pred), 5)
        best_epoch = scNAMEcluster.best_epoch
        result[idx, 1:] = [kmeans_accuracy, kmeans_ARI, kmeans_NMI, best_epoch, accuracy, ARI, NMI]
        print(result)
        idx+=1
    result[-3] = np.round(np.mean(result[0:len(args.random_seed), :], 0), 5)
    result[-2] = np.round(np.median(result[0:len(args.random_seed), :], 0), 5)
    result[-1] = np.round(np.std(result[0:len(args.random_seed), :], 0), 5)
    list = np.arange(1, len(args.random_seed) + 1)
    list = [str(i) for i in list]
    list.append('mean')
    list.append('median')
    list.append('std')
    output = np.array(result)
    output = pd.DataFrame(output,columns=["dataset", "kmeans accuracy", "kmeans ARI", "kmeans NMI", "best epoch",
                                       "accuracy", "ARI", "NMI"],index=list)

    output["dataset"] =args.dataname

    print(output)
