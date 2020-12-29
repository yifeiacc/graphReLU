import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from drelu import DyReLUA, DyReLUB, DyReLUC, DyReLUE
from train_eval import experiment
from datasets import get_planetoid_dataset
from edgerelu import EdgeRelu, EdgeReluV2
import time
import numpy as np


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=str2bool, default=True)
parser.add_argument('--runs', type=int, default=30)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--hidden', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=str2bool, default=True)

args = parser.parse_args()
# print("asdasda")
# print(args.random_splits)


class Net(torch.nn.Module):
    def __init__(self, dataset, kind="None"):
        super(Net, self).__init__()
        self.conv1 = GCNConv(dataset.num_features, args.hidden)
        self.conv2 = GCNConv(args.hidden, dataset.num_classes)

        self.kind = kind
        self.dreluA = DyReLUA(args.hidden)
        self.dreluB = DyReLUB(args.hidden)
        self.dreluC = DyReLUC(args.hidden)
        self.dreluD = EdgeReluV2(args.hidden)
        self.relu = torch.nn.PReLU()
        torch.nn.ELU()

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        if self.kind == "A":
            x = self.dreluA(self.conv1(x, edge_index), edge_index)
        elif self.kind == "B":
            x = self.dreluB(self.conv1(x, edge_index), edge_index)
        elif self.kind == "C":
            x = self.dreluC(self.conv1(x, edge_index), edge_index)
        elif self.kind == "D":
            x = self.dreluD(self.conv1(x, edge_index), edge_index)
        elif self.kind == "ReLU":
            x = F.relu(self.conv1(x, edge_index))
        elif self.kind == "PReLU":
            x = F.prelu(self.conv1(x, edge_index), weight=0.25)
        elif self.kind == "ELU":
            x = F.elu(self.conv1(x, edge_index), alpha=1)
        elif self.kind == "LReLU":
            x = F.leaky_relu(self.conv1(x, edge_index), negative_slope=0.01)

        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        # if not self.training:
        #     t = time.time()
        #     if self.kind == "A":
        #         self.coefs = self.dreluA.coefs
        #     elif self.kind == "B":
        #         self.coefs = self.dreluB.coefs
        #     elif self.kind == "C":
        #         self.coefs = self.dreluC.coefs

        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
    experiment(args, Net, dataset, "GCN")
