import argparse
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from drelu import DyReLUA, DyReLUB, DyReLUC
from train_eval import random_planetoid_splits, run
from datasets import get_planetoid_dataset
from edgerelu import EdgeRelu, EdgeReluV2


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
parser.add_argument('--runs', type=int, default=10)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.001)
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
        else:
            x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=args.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)


def main():
    # print(args.normalize_features)
    print(args)
    dataset = get_planetoid_dataset(args.dataset, args.normalize_features)
    permute_masks = random_planetoid_splits if args.random_splits else None

    lst = ["D", "C", "B", "A", "None"]
    results = []
    for i in lst:
        print("The Model - {}".format(i))
        print(args)
        result = run(dataset, Net(dataset, kind=i), args.runs, args.epochs, args.lr, args.weight_decay,
                     args.early_stopping, permute_masks)
        results.append(result)

    print("Model: {}".format(__file__))
    for name, re in zip(lst, results):
        print("{} - {}".format(name, re))


if __name__ == "__main__":
    main()
