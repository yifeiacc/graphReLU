import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import ChebConv

from drelu import DyReLUA, DyReLUB, DyReLUC
from train_eval import random_planetoid_splits, run, experiment
from datasets import get_planetoid_dataset
from edgerelu import EdgeReluV2

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=30)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=10)
parser.add_argument('--hidden', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--num_hops', type=int, default=3)
args = parser.parse_args()


class Net(torch.nn.Module):
    def __init__(self, dataset, kind="None"):
        super(Net, self).__init__()
        self.conv1 = ChebConv(dataset.num_features, args.hidden, args.num_hops)
        self.conv2 = ChebConv(args.hidden, dataset.num_classes, args.num_hops)
        self.kind = kind
        self.dreluA = DyReLUA(args.hidden)
        self.dreluB = DyReLUB(args.hidden)
        self.dreluC = DyReLUC(args.hidden)
        self.dreluD = EdgeReluV2(args.hidden)
        self.PReLU = torch.nn.PReLU()

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
            x = self.PReLU(self.conv1(x, edge_index))
        elif self.kind == "ELU":
            x = F.elu(self.conv1(x, edge_index), alpha=1)
        elif self.kind == "LReLU":
            x = F.leaky_relu(self.conv1(x, edge_index), negative_slope=0.01)
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
    experiment(args, Net, dataset, "Cheb")
