from __future__ import division
import pickle

import time

import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
import sys
import pandas as pd
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


def run(dataset, model, runs, epochs, lr, weight_decay, early_stopping,
        permute_masks=None, logger=None):

    val_losses, accs, durations = [], [], []
    for i in range(runs):
        data = dataset[0]
        if permute_masks is not None:
            data = permute_masks(data, dataset.num_classes)
        data = data.to(device)

        model.to(device).reset_parameters()
        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        best_val_loss = float('inf')
        test_acc = 0
        val_loss_history = []

        for epoch in range(1, epochs + 1):
            train(model, optimizer, data)
            eval_info = evaluate(model, data)
            eval_info['epoch'] = epoch

            if logger is not None:
                logger(eval_info)

            if eval_info['val_loss'] < best_val_loss:
                best_val_loss = eval_info['val_loss']
                test_acc = eval_info['test_acc']

            val_loss_history.append(eval_info['val_loss'])
            if early_stopping > 0 and epoch > epochs // 2:
                tmp = tensor(val_loss_history[-(early_stopping + 1):-1])
                if eval_info['val_loss'] > tmp.mean().item():
                    break
            # print(eval_info, file=sys.stderr)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()

        val_losses.append(best_val_loss)
        accs.append(test_acc)
        durations.append(t_end - t_start)

        # if model.kind != "D" and model.kind != "None":
        #     coefs = model.coefs.data.cpu().numpy()
        #     # print(coefs)
        #     with open("GCN-{}.pk".format(model.kind), "wb") as f:
        #         pickle.dump([coefs, data], f)
        #     np.save("GCN-{}.npy".format(model.kind), coefs)

    loss, acc, duration = tensor(val_losses), tensor(accs), tensor(durations)

    print('Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(loss.mean().item(),
                 acc.mean().item(),
                 acc.std().item(),
                 duration.mean().item()))
    re_acc = ",".join(["{:.3f}".format(i) for i in acc])
    re_loss = ",".join(["{:.3f}".format(i) for i in loss])
    return acc.mean().item(), acc.std().item(), acc.max().item(), re_acc, re_loss


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        logits = model(data)

    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = acc

    return outs


def experiment(args, Net, dataset, name):
    # print(args.normalize_features)
    print(args)
    permute_masks = random_planetoid_splits if args.random_splits else None

    lst = ["D", "C"]
    # lst = ["ReLU", "PReLU", "ELU", "LReLU"]
    # lst = ["D"]
    # lst = ["LReLU"]
    # lst = ["D", "C", "B", "ReLU", "PReLU", "ELU", "LReLU"]
    results = []
    best = []
    re_acc_lst = []
    re_loss_lst = []
    
    for i in lst:
        print("The Model - {}".format(i))
        print(args)
        if (args.dataset == 'BlogCatalog' or args.dataset == 'flickr') and i == "D":
            mean, std, mx, re_acc, re_loss = run(dataset, Net(dataset, kind=i), args.runs, args.epochs*3, args.lr, args.weight_decay,
                            args.early_stopping, permute_masks)
        else:
            mean, std, mx, re_acc, re_loss = run(dataset, Net(dataset, kind=i), args.runs, args.epochs, args.lr, args.weight_decay,
                            args.early_stopping, permute_masks)

        results.append("{:.3f} ± {:.3f}".format(mean, std))
        best.append("{:.3f}".format(mx))
        re_acc_lst.append(re_acc)
        re_loss_lst.append(re_loss)

    print("Model: {}".format(__file__))
    with open(".\\result\\{}-{}.csv".format(name, args.dataset), "w") as f:
        f.write("   ".join(lst) + "\n")
        f.write("   ".join(results) + "\n")
        f.write("   ".join(best) + "\n")
        f.write("   ".join(re_acc_lst) + "\n")
        f.write("   ".join(re_loss_lst) + "\n")
