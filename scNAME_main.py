from scNAME_preprocess import *
from scNAME_network import *
from scNAME_utils import *
import argparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 5000)

def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w) 
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

if __name__ == "__main__":
    random_seed = [1111]

    parser = argparse.ArgumentParser(description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--dataname", default = "Muraro", type = str)
    parser.add_argument('--p_m',help='corruption probability for self-supervised learning',default=0.3, type=float)
    parser.add_argument('--k',help='top k most similar features to be neighborhoods',default=10,type=int)
    parser.add_argument('--temperature',help='hyper-parameter in l3 loss', default=0.7,type=float)
    parser.add_argument("--dims", default = [256, 64,32])
    parser.add_argument("--highly_genes", default = 1000, type = int)
    parser.add_argument("--alpha",help='hyper-parameter to control the weights of l2 loss', default = 1.0,type = float)
    parser.add_argument("--beta",help='hyper-parameter to control the weights of l3 loss', default=0.1, type=float)
    parser.add_argument("--gamma",help='hyper-parameter to control the weights of l4 loss', default =0.1, type = float)
    parser.add_argument("--learning_rate", default = 0.0001, type = float)
    parser.add_argument("--random_seed", default = random_seed)
    parser.add_argument("--batch_size", default = 128, type = int)
    parser.add_argument("--pretrain_epoch", default = 500, type = int)
    parser.add_argument("--funetrain_epoch", default = 1000, type = int)
    parser.add_argument("--update_epoch", default=50, type=int) 
    parser.add_argument("--noise_sd", default = 1.5)
    parser.add_argument("--error", default = 0.001, type = float)
    parser.add_argument("--gpu_option", default ="2,3" )

    args = parser.parse_args()
    X, Y = prepro(args.dataname)
    X = np.ceil(X).astype(int)
    count_X = X
    shuffle_ix = np.random.permutation(np.arange(len(X)))
    X = X[shuffle_ix]
    Y = Y[shuffle_ix]
    count_X = count_X[shuffle_ix]
    adata = sc.AnnData(X)
    adata.obs['Group'] = Y
    adata = normalize(adata, copy=True, highly_genes=args.highly_genes, size_factors=True,
                     normalize_input=True, logtrans_input=True)
    X = adata.X.astype(np.float32)
    Y = np.array(adata.obs["Group"])
    n = X.shape[0]
    high_variable = np.array(adata.var.highly_variable.index, dtype=np.int) 
    count_X = count_X[:, high_variable]
    size_factor = np.array(adata.obs.size_factors).reshape(-1, 1).astype(np.float32)
    args.dims.insert(0,args.highly_genes)
    cluster_number = int(max(Y) - min(Y) + 1)
    result = np.zeros([len(args.random_seed), 7])
    idx = 0
    for i in args.random_seed:
        np.random.seed([i])
        print("the dataset {} has {} samples".format(args.dataname, n))
        tf.reset_default_graph() 
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
        result[idx, 1:] = [kmeans_accuracy, kmeans_ARI, kmeans_NMI, accuracy, ARI, NMI]
        print(result)
        idx+=1
    list = np.arange(1, len(args.random_seed) + 1)
    list = [str(i) for i in list]
    output = np.array(result)
    output = pd.DataFrame(output,columns=["dataset", "kmeans accuracy", "kmeans ARI", "kmeans NMI","accuracy", "ARI", "NMI"],index=list)

    output["dataset"] =args.dataname

    print(output)
