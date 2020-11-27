# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import time

import faiss
import numpy as np
from PIL import Image
from PIL import ImageFile
from scipy.sparse import csr_matrix, find
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

ImageFile.LOAD_TRUNCATED_IMAGES = True

__all__ = ['PIC', 'Kmeans', 'cluster_assign', 'arrange_clustering']

def preprocess_features(npdata, pca=256):
    """Preprocess an array of features.
    Args:
        npdata (np.array N * ndim): features to preprocess
        pca (int): dim of output
    Returns:
        np.array of dim N * pca: data PCA-reduced, whitened and L2-normalized
    """
    _, ndim = npdata.shape
    npdata =  npdata.astype('float32')

    # Apply PCA-whitening with Faiss
    mat = faiss.PCAMatrix (ndim, pca, eigen_power=-0.5)
    mat.train(npdata)
    assert mat.is_trained
    npdata = mat.apply_py(npdata)

    # L2 normalization
    row_sums = np.linalg.norm(npdata, axis=1)
    npdata = npdata / row_sums[:, np.newaxis]

    return npdata


def make_graph(xb, nnn):
    """Builds a graph of nearest neighbors.
    Args:
        xb (np.array): data
        nnn (int): number of nearest neighbors
    Returns:
        list: for each data the list of ids to its nnn nearest neighbors
        list: for each data the list of distances to its nnn NN
    """
    N, dim = xb.shape

    # we need only a StandardGpuResources per GPU
    res = faiss.StandardGpuResources()

    # L2
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.device = int(torch.cuda.device_count()) - 1
    index = faiss.GpuIndexFlatL2(res, dim, flat_config)
    index.add(xb)
    D, I = index.search(xb, nnn + 1)
    return I, D


def cluster_assign(images_lists):
    """Creates a dataset from clustering, with clusters as labels.
    Args:
        images_lists (list of list): for each cluster, the list of image indexes
                                    belonging to this cluster
        dataset (list): initial dataset
    Returns:
        ReassignedDataset(torch.utils.data.Dataset): a dataset with clusters as
                                                     labels
    """
    assert images_lists is not None
    #image_label
    pseudolabels = []
    #image_index
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))

    label_to_idx = {label: idx for idx, label in enumerate(set(pseudolabels))}
    #image_label_index
    #images_label_index = []
    images_dict = {}
    for j, idx in enumerate(image_indexes):
        pseudolabel = label_to_idx[pseudolabels[j]]
        #images_label_index.append(pseudolabel)
        images_dict[idx] = pseudolabel

    #return image_indexes, images_label_index
    #images_dict = sorted(images_dict.items(), key=lambda obj: obj[0])
    return images_dict


def run_kmeans(x, nmb_clusters, niter, verbose=False):
    """Runs kmeans on 1 GPU.
    Args:
        x: data
        nmb_clusters (int): number of clusters
    Returns:
        list: ids of data in each cluster
    """
    n_data, d = x.shape

    # faiss implementation of k-means
    clus = faiss.Clustering(d, nmb_clusters)

    # Change faiss seed at each k-means so that the randomly picked
    # initialization centroids do not correspond to the same feature ids
    # from an epoch to another.
    clus.seed = np.random.randint(1234)

    clus.niter = niter
    clus.max_points_per_centroid = 10000000
    res = faiss.StandardGpuResources()
    flat_config = faiss.GpuIndexFlatConfig()
    flat_config.useFloat16 = False
    flat_config.device = 0
    index = faiss.GpuIndexFlatL2(res, d, flat_config)

    # perform the training
    clus.train(x, index)
    _, I = index.search(x, 1)
    losses = faiss.vector_to_array(clus.obj)
    if verbose:
        print('k-means loss evolution: {0}'.format(losses))

    return [int(n[0]) for n in I], losses[-1]

def softmax(x):
    x_row_max = x.max(axis=-1)
    x_row_max = x_row_max.reshape(list(x.shape)[:-1]+[1])
    x = x - x_row_max
    x_exp = np.exp(x)
    x_exp_row_sum = x_exp.sum(axis=-1).reshape(list(x.shape)[:-1]+[1])
    softmax = x_exp / x_exp_row_sum
    return softmax


def arrange_clustering(images_lists):
    pseudolabels = []
    image_indexes = []
    for cluster, images in enumerate(images_lists):
        image_indexes.extend(images)
        pseudolabels.extend([cluster] * len(images))
    indexes = np.argsort(image_indexes)
    return np.asarray(pseudolabels)[indexes]


class Kmeans(object):
    def __init__(self, k):
        self.k = k

    def plot_embedding_mergelabel(self, data, label, view_number, epoch):
        label = np.array(label)
        """
        #sample_cluster = np.random.choice(21,6)
        sample_index = np.random.choice(99, 100)
        indexs=[]
        for i in range(6):
            l=len(index_list[i])
            if l<100:
                indexs.extend(index_list[i][:min([50, l])])
                continue
            indexs.extend([index_list[i][j] for j in sample_index])
        
        """

        x_min, x_max = np.min(data, 0), np.max(data, 0)
        data = (data - x_min) / (x_max - x_min)
        print("var_of_totaldata:",np.var(data))
        #row_sums = np.linalg.norm(data, axis=1)
        #data = data / row_sums[:, np.newaxis]
        index_dict = {}
        cluster_dict = {}
        for i in range(self.k):
            v_x=self.images_lists[i][:100]
            v=np.var(data[v_x,:])
            index_dict[v]=v_x
            cluster_dict[v]=i



        dict2 = sorted(cluster_dict, reverse=False)
        indexs = []

        """
        for i in range(self.k):
            v=dict2[i]
            v_x=[]
            for j in range(i, self.k):
                v_x.extend(index_dict[dict2[j]])
            compare_v=np.var(data[v_x,:])
            print("compare_v" + str(compare_v) + "<" + str(v))
            if (compare_v-v)<view_number:
                view_number=i
                print("view_number:" + str(i))
                break
        view_number+=1
        """
        #image_list_temp = [[] for i in range(view_number + 1)]
        for i in range(self.k):
            v=dict2[i]
            label[index_dict[v]]=i
            print(str(v)+"\t"+str(len(index_dict[v])))
            if i < view_number:
                indexs.extend(index_dict[v])
                ##image_list_temp[i]=self.images_lists[cluster_dict[v]]
            #else:
                #indexs.extend(index_dict[v][:10])
                #label[index_dict[v]] = view_number
                #image_list_temp[view_number].extend(self.images_lists[cluster_dict[v]])

        tsne = TSNE(n_components=3, init='pca', random_state=25)
        data=data[indexs, :]
        label = label[indexs]
        data = tsne.fit_transform(data)

        
        color=np.vstack((plt.cm.Set1([i for i in range(9)]),plt.cm.Set2([i for i in range(8)]),plt.cm.Set3([i for i in range(12)])))
        #color = plt.cm.Set3([i for i in range(view_number+1)])
        fig = plt.figure()
        ax = Axes3D(fig)
        ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=np.array([list(color[label[i]]) for i in range(len(label))]))
        #ax.set_xlabel('X')
        #ax.set_ylabel('Y')
        #ax.set_zlabel('Z')
        ax.grid(False)
        plt.axis('off')
        plt.savefig("data/saved_fig/cluster-" + str(self.k) + "-" + str(view_number) + "-" + "-" + str(epoch))
        #ax.view_init(45, 120)
        #plt.title(title)
        #plt.show()
        #return fig


    def plot_embedding(self, data, label):
        label = np.array(label)

        #x_min, x_max = np.min(data, 0), np.max(data, 0)
        #data = (data - x_min) / (x_max - x_min)
        print("var_of_totaldata:", np.var(data))
        index_dict = {}
        cluster_dict = {}
        for i in range(self.k):
            v_x = self.images_lists[i][:]
            v = np.var(data[v_x, :])
            index_dict[v] = v_x
            cluster_dict[v] = i

        sorted_var = sorted(cluster_dict, reverse=False)
        indexs = []


        image_list_temp = [[] for i in range(self.k)]
        for i in range(self.k):
            v = sorted_var[i]
            label[index_dict[v]] = i
            print(str(v) + "\t" + str(len(index_dict[v])))
            image_list_temp[i] = self.images_lists[cluster_dict[v]]

        self.images_lists = image_list_temp
        image_list_temp = []
        sorted_var = 1/np.array(sorted_var)
        print(softmax(sorted_var))
        return sorted_var


    def draw_cluster(self):
        np.random.seed(25)
        pubmed_class_label = ['biological_process_involves_gene_product', 'inheritance_type_of',
                              'is_normal_tissue_origin_of_disease', 'ingredient_of',
                              'is_primary_anatomic_site_of_disease', 'gene_found_in_organism', 'occurs_in',
                              'causative_agent_of', 'classified_as', 'gene_plays_role_in_process']

        file_name = "most_val_pubmed_features"

        val_fea = np.load(file_name + ".npy")

        sample_index = np.random.choice(99, 20)
        val_fea = val_fea[:, sample_index, :]

        val_fea = np.reshape(val_fea, [-1, 230])

        tsne_adv = TSNE(n_components=2, init="pca", random_state=25)
        val_fea_tsne = tsne_adv.fit_transform(val_fea)

        val_fea_tsne = val_fea_tsne.reshape(10, 20, 2)

        plt.xticks([])
        plt.yticks([])

        legend_record = []
        for i in range(10):
            leg = plt.scatter(val_fea_tsne[i, :, 0], val_fea_tsne[i, :, 1])
            legend_record.append(leg)

        plt.legend(handles=legend_record, labels=pubmed_class_label)
        #plt.show()
        plt.savefig(file_name + ".png", pad_inches=0.1, bbox_inches='tight')

    def cluster(self, data, view_number, cluster_layer, pca_dim, niter, epoch, verbose=False):
        """Performs k-means clustering.
            Args:
                x_data (np.array N * dim): data to cluster
        """
        end = time.time()

        # PCA-reducing, whitening and L2-normalization
        xb = preprocess_features(data, pca_dim)

        # cluster the data
        I, loss = run_kmeans(xb, self.k, niter, verbose)

        self.images_lists = [[] for i in range(self.k)]
        for i in range(len(data)):
            self.images_lists[I[i]].append(i)

        #draw
        """
        print("&&&&&Begin Drawing&&&&&")
        #var=self.plot_embedding(xb, I)
        self.plot_embedding_mergelabel(xb, I, 4, epoch)
        self.plot_embedding_mergelabel(xb, I, 5, epoch)
        self.plot_embedding_mergelabel(xb, I, 6, epoch)
        self.plot_embedding_mergelabel(xb, I, 7, epoch)
        self.plot_embedding_mergelabel(xb, I, 8, epoch)
        print("&&&&&End Drawing&&&&&")
        """

        var=[]
        #x_min, x_max=np.min(data,0), np.max(data,0)
        #data=(data-x_min) / (x_max-x_min)
        for i in range(self.k):
            var.append(np.var(data[self.images_lists[i], :]))
        print(var)
        
        var=1/np.array(var)

        print(softmax(var))




        for i in range(len(self.images_lists)):
            print("Number of Label-"+str(i)+": ", len(self.images_lists[i]))

        if verbose:
            print('k-means time: {0:.0f} s'.format(time.time() - end))

        return loss, var


def make_adjacencyW(I, D, sigma):
    """Create adjacency matrix with a Gaussian kernel.
    Args:
        I (numpy array): for each vertex the ids to its nnn linked vertices
                  + first column of identity.
        D (numpy array): for each data the l2 distances to its nnn linked vertices
                  + first column of zeros.
        sigma (float): Bandwidth of the Gaussian kernel.

    Returns:
        csr_matrix: affinity matrix of the graph.
    """
    V, k = I.shape
    k = k - 1
    indices = np.reshape(np.delete(I, 0, 1), (1, -1))
    indptr = np.multiply(k, np.arange(V + 1))

    def exp_ker(d):
        return np.exp(-d / sigma**2)

    exp_ker = np.vectorize(exp_ker)
    res_D = exp_ker(D)
    data = np.reshape(np.delete(res_D, 0, 1), (1, -1))
    adj_matrix = csr_matrix((data[0], indices[0], indptr), shape=(V, V))
    return adj_matrix


def run_pic(I, D, sigma, alpha):
    """Run PIC algorithm"""
    a = make_adjacencyW(I, D, sigma)
    graph = a + a.transpose()
    cgraph = graph
    nim = graph.shape[0]

    W = graph
    t0 = time.time()

    v0 = np.ones(nim) / nim

    # power iterations
    v = v0.astype('float32')

    t0 = time.time()
    dt = 0
    for i in range(200):
        vnext = np.zeros(nim, dtype='float32')

        vnext = vnext + W.transpose().dot(v)

        vnext = alpha * vnext + (1 - alpha) / nim
        # L1 normalize
        vnext /= vnext.sum()
        v = vnext

        if i == 200 - 1:
            clust = find_maxima_cluster(W, v)

    return [int(i) for i in clust]


def find_maxima_cluster(W, v):
    n, m = W.shape
    assert (n == m)
    assign = np.zeros(n)
    # for each node
    pointers = list(range(n))
    for i in range(n):
        best_vi = 0
        l0 = W.indptr[i]
        l1 = W.indptr[i + 1]
        for l in range(l0, l1):
            j = W.indices[l]
            vi = W.data[l] * (v[j] - v[i])
            if vi > best_vi:
                best_vi = vi
                pointers[i] = j
    n_clus = 0
    cluster_ids = -1 * np.ones(n)
    for i in range(n):
        if pointers[i] == i:
            cluster_ids[i] = n_clus
            n_clus = n_clus + 1
    for i in range(n):
        # go from pointers to pointers starting from i until reached a local optim
        current_node = i
        while pointers[current_node] != current_node:
            current_node = pointers[current_node]

        assign[i] = cluster_ids[current_node]
        assert (assign[i] >= 0)
    return assign


class PIC(object):
    """Class to perform Power Iteration Clustering on a graph of nearest neighbors.
        Args:
            args: for consistency with k-means init
            sigma (float): bandwidth of the Gaussian kernel (default 0.2)
            nnn (int): number of nearest neighbors (default 5)
            alpha (float): parameter in PIC (default 0.001)
            distribute_singletons (bool): If True, reassign each singleton to
                                      the cluster of its closest non
                                      singleton nearest neighbors (up to nnn
                                      nearest neighbors).
        Attributes:
            images_lists (list of list): for each cluster, the list of image indexes
                                         belonging to this cluster
    """

    def __init__(self, args=None, sigma=0.2, nnn=5, alpha=0.001, distribute_singletons=True):
        self.sigma = sigma
        self.alpha = alpha
        self.nnn = nnn
        self.distribute_singletons = distribute_singletons

    def cluster(self, data, verbose=False):
        end = time.time()

        # preprocess the data
        xb = preprocess_features(data)

        # construct nnn graph
        I, D = make_graph(xb, self.nnn)

        # run PIC
        clust = run_pic(I, D, self.sigma, self.alpha)
        images_lists = {}
        for h in set(clust):
            images_lists[h] = []
        for data, c in enumerate(clust):
            images_lists[c].append(data)

        # allocate singletons to clusters of their closest NN not singleton
        if self.distribute_singletons:
            clust_NN = {}
            for i in images_lists:
                # if singleton
                if len(images_lists[i]) == 1:
                    s = images_lists[i][0]
                    # for NN
                    for n in I[s, 1:]:
                        # if NN is not a singleton
                        if not len(images_lists[clust[n]]) == 1:
                            clust_NN[s] = n
                            break
            for s in clust_NN:
                del images_lists[clust[s]]
                clust[s] = clust[clust_NN[s]]
                images_lists[clust[s]].append(s)

        self.images_lists = []
        for c in images_lists:
            self.images_lists.append(images_lists[c])

        if verbose:
            print('pic time: {0:.0f} s'.format(time.time() - end))
        return 0

