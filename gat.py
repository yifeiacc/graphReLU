import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from drelu import DyReLUA, DyReLUB, DyReLUC
from train_eval import random_planetoid_splits, run, experiment
from datasets import get_planetoid_dataset
from edgerelu import EdgeReluV2
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--random_splits', type=bool, default=False)
parser.add_argument('--runs', type=int, default=30)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.005)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--early_stopping', type=int, default=100)
parser.add_argument('--hidden', type=int, default=8)
parser.add_argument('--dropout', type=float, default=0.6)
parser.add_argument('--normalize_features', type=bool, default=True)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--output_heads', type=int, default=1)
args = parser.parse_args()


class Net(torch.nn.Module):
    def __init__(self, dataset, kind="None"):
        super(Net, self).__init__()
        self.conv1 = GATConv(
            dataset.num_features,
            args.hidden,
            heads=args.heads,
            dropout=args.dropout)
        self.conv2 = GATConv(
            args.hidden * args.heads,
            dataset.num_classes,
            heads=args.output_heads,
            concat=False,
            dropout=args.dropout)

        self.kind = kind
        self.dreluA = DyReLUA(args.hidden * args.heads)
        self.dreluB = DyReLUB(args.hidden * args.heads)
        self.dreluC = DyReLUC(args.hidden * args.heads)
        self.dreluD = EdgeReluV2(args.hidden * args.heads)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=args.dropout, training=self.training)
        if self.kind == "A":
            x = self.dreluA(self.conv1(x, edge_index), edge_index)
        elif self.kind == "B":
            x = self.dreluB(self.conv1(x, edge_index), edge_index)
        elif self.kind == "C":
            x = self.dreluC(self.conv1(x, edge_index), edge_index)
        elif self.kind == "D":
            x = self.dreluD(self.conv1(x, edge_index), edge_index)
        else:
            x = F.elu(self.conv1(x, edge_index))
        # x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
    experiment(args, Net, dataset, "GAT")
