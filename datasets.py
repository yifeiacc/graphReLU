import os
import os.path as osp
import shutil

import numpy as np
import scipy.sparse as sp
import torch
import torch_geometric.transforms as T
from torch_geometric.data import (Data, InMemoryDataset, download_url,
                                  extract_zip)
from torch_geometric.datasets import Planetoid


def index_to_mask(index, size):
    mask = torch.zeros(size, dtype=torch.bool, device=index.device)
    mask[index] = 1
    return mask



def random_planetoid_splits(data, num_classes):
    # Set new random planetoid splits:
    # * 20 * num_classes labels for training
    # * 500 labels for validation
    # * 1000 labels for testing

    indices = []
    for i in range(num_classes):
        index = (data.y == i).nonzero().view(-1)
        index = index[torch.randperm(index.size(0))]
        indices.append(index)

    train_index = torch.cat([i[:20] for i in indices], dim=0)

    rest_index = torch.cat([i[20:] for i in indices], dim=0)
    rest_index = rest_index[torch.randperm(rest_index.size(0))]

    data.train_mask = index_to_mask(train_index, size=data.num_nodes)
    data.val_mask = index_to_mask(rest_index[:500], size=data.num_nodes)
    data.test_mask = index_to_mask(rest_index[500:1500], size=data.num_nodes)

    return data


def get_planetoid_dataset(name, normalize_features=False, transform=None):

    if name == "flickr":
        path = osp.join(osp.dirname(osp.realpath(__file__)),
                        '..', 'data',name)
        dataset = FlickerDataSet(root=path)
    elif name == "BlogCatalog":
        path = osp.join(osp.dirname(osp.realpath(__file__)),
                        '..', 'data',name)
        dataset = BlogCatalogDataSet(root=path)
    else:
        path = osp.join(osp.dirname(osp.realpath(__file__)),
                        '..', 'data', name)
        dataset = Planetoid(path, name)

    if transform is not None and normalize_features:
        dataset.transform = T.Compose([T.NormalizeFeatures(), transform])
    elif normalize_features:
        dataset.transform = T.NormalizeFeatures()
    elif transform is not None:
        dataset.transform = transform

    return dataset


class FlickerDataSet(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(FlickerDataSet, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['flickr.edge', 'flickr.label', 'flickr.edge']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # shutil.rmtree(self.raw_dir)
        # path = download_url(
        #     "https://github.com/zhumeiqiBUPT/AM-GCN/blob/master/data/flickr.zip", self.root)
        # extract_zip(path, self.root)
        # os.rename(osp.join(self.root, 'flickr'), self.raw_dir)
        # os.unlink(path)
        pass

    def process(self):
        # Read data into huge `Data` list.
        g_path = self.raw_dir + "\\flickr.edge"
        f_path = self.raw_dir + "\\flickr.feature"
        l_path = self.raw_dir + "\\flickr.label"

        struct_edges = np.genfromtxt(g_path, dtype=np.int32)
        edges = np.array(list(struct_edges), dtype=np.int32).reshape(
            struct_edges.shape)
        adj = sp.coo_matrix((np.ones(
            edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(
            7575, 7575), dtype=np.float32)

        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        fea = np.loadtxt(f_path, dtype=float)
        lab = np.loadtxt(l_path, dtype=int)
        features = sp.csr_matrix(fea, dtype=np.float32)
        features = torch.FloatTensor(np.array(features.todense()))
        label = torch.LongTensor(np.array(lab))
        data = Data(x=features, edge_index=edge_index, y=label)
        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class BlogCatalogDataSet(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(BlogCatalogDataSet, self).__init__(
            root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['BlogCatalog.edge', 'BlogCatalog.label', 'BlogCatalog.edge']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        shutil.rmtree(self.raw_dir)
        path = download_url(
            "https://github.com/zhumeiqiBUPT/AM-GCN/blob/master/data/BlogCatalog.zip", self.root)
        extract_zip(path, self.root)
        os.rename(osp.join(self.root, 'BlogCatalog'), self.raw_dir)
        os.unlink(path)

    def process(self):
        # Read data into huge `Data` list.
        g_path = self.raw_dir + "/BlogCatalog.edge"
        f_path = self.raw_dir + "/BlogCatalog.feature"
        l_path = self.raw_dir + "/BlogCatalog.label"

        struct_edges = np.genfromtxt(g_path, dtype=np.int32)
        edges = np.array(list(struct_edges), dtype=np.int32).reshape(
            struct_edges.shape)
        adj = sp.coo_matrix((np.ones(
            edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(
            5196, 5196), dtype=np.float32)

        row = torch.from_numpy(adj.row).to(torch.long)
        col = torch.from_numpy(adj.col).to(torch.long)
        edge_index = torch.stack([row, col], dim=0)

        fea = np.loadtxt(f_path, dtype=float)
        lab = np.loadtxt(l_path, dtype=int)
        features = sp.csr_matrix(fea, dtype=np.float32)
        features = torch.FloatTensor(np.array(features.todense()))
        label = torch.LongTensor(np.array(lab))
        data = Data(x=features, edge_index=edge_index, y=label)

        data_list = [data]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
