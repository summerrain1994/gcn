import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix
from scipy.sparse.linalg.eigen.arpack import eigsh
import random

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):   
    cora_cites = pd.read_csv('/home/gcn/gcn/data_input/cora.cites', sep = '\s+' ,header = None)
    cora_content = pd.read_csv('/home/gcn/gcn/data_input/cora.content', sep = '\s+' ,header = None)
   
    import collections
    id_dict = collections.defaultdict(list)
    for i in range(2708):
        j = str(cora_content.iloc[i,0])
        id_dict[j].append(i)    
    
    #features
    X = cora_content.iloc[:,1:1434]
    X = np.array(X)
    features = lil_matrix(X)
    
    #labels
    Y = cora_content.iloc[:,1434]
    Y = np.array(pd.get_dummies(Y))
    labels = Y
    
    #adj matrix
    N = 2708
    A = np.zeros((N, N), dtype = int)
    for k in range(N):    
        id1 = str(cora_cites.iloc[k,0])
        id2 = str(cora_cites.iloc[k,1])
        i1 =  id_dict[id1]
        i2 =  id_dict[id2]
        A[i1,i2] = random.randint(1,10)
        A[i2,i1] = random.randint(1,10)
        #A[i1,i2] = 1
        #A[i2,i1] = 1
    adj = csr_matrix(A)
    
    #train_ratio = 
    #val_ratio = 
    #test_ratio =
    
    train_mask = np.zeros(N)
    train_mask[0:1000] = 1
    train_mask = np.array(train_mask, dtype=np.bool)
    
    val_mask = np.zeros(N)
    val_mask[1000:1500] = 1
    val_mask = np.array(val_mask, dtype=np.bool)
    
    test_mask = np.zeros(N)
    test_mask[1500:2700] = 1
    test_mask = np.array(test_mask, dtype=np.bool)
    
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = (1/rowsum).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)


def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)
