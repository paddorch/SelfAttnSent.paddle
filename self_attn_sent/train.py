#!/usr/bin/env python3
import argparse
import datetime
import errno
import sys
import pickle
import os

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle import optimizer
from paddle.io import DataLoader

from model import SelfAttnSent
from data import NLIDataset
from utils import set_seed, print_f_score

set_seed(42)

parser = argparse.ArgumentParser(description='Character level CNN text classifier training')
# data
parser.add_argument('--train_path', metavar='DIR',
                    help='path to training data csv [default: ../data/preprocessed/train_data.pkl]',
                    default='../data/preprocessed/train_data.pkl')  # TODO
parser.add_argument('--val_path', metavar='DIR',
                    help='path to validation data csv [default: ../data/preprocessed/dev_data.pkl]',
                    default='../data/preprocessed/dev_data.pkl')
parser.add_argument('--embed_path', metavar='DIR',
                    help='path to embedding [default: ../data/preprocessed/embeddings_data.pkl]',
                    default='../data/preprocessed/embeddings.pkl')
# learning
learn = parser.add_argument_group('Learning options')
learn.add_argument('--lr', type=float, default=0.01, help='initial learning rate [default: 0.01]')
learn.add_argument('--epochs', type=int, default=200, help='number of epochs for train [default: 200]')
learn.add_argument('--batch_size', type=int, default=50, help='batch size for training [default: 50]')
learn.add_argument('--grad_clip', default=100, type=int, help='Norm cutoff to prevent explosion of gradients')
learn.add_argument('--optimizer', default='Adagrad',
                   help='Type of optimizer. Adagrad|Adam|AdamW are supported [default: Adagrad]')
learn.add_argument('--class_weight', default=None, action='store_true',
                   help='Weights should be a 1D Tensor assigning weight to each of the classes.')
# model
model_cfg = parser.add_argument_group('Model options')
model_cfg.add_argument('--lstm_hid_dim', type=int, default=300,
                       help='hidden size of LSTM [default: 300]')
model_cfg.add_argument('--shuffle', action='store_true', default=True, help='shuffle the data every epoch')
model_cfg.add_argument('--dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
model_cfg.add_argument('--use_penalty', action='store_true', default=False, help='whether enable penalty')
model_cfg.add_argument('--penalty_c', type=float, default=0.1, help='penalty term coefficient [default: 0.1]')
model_cfg.add_argument('--d_a', type=int, default=150,
                       help='d_a size [default: 150]')
model_cfg.add_argument('--r', type=int, default=30,
                       help='row size of sentence embedding [default: 30]')
model_cfg.add_argument('--output_hid_dim', type=int, default=4000,
                       help='mlp hidden size [default: 4000]')
# device
device = parser.add_argument_group('Device options')
device.add_argument('--num_workers', default=8, type=int, help='Number of workers used in data-loading')
device.add_argument('--cuda', action='store_true', default=True, help='enable the gpu')
device.add_argument('--gpu', type=int, default=None)
# experiment options
experiment = parser.add_argument_group('Experiment options')
experiment.add_argument('--verbose', dest='verbose', action='store_true', default=False,
                        help='Turn on progress tracking per iteration for debugging')
experiment.add_argument('--continue_from', default='', help='Continue from checkpoint model')
experiment.add_argument('--checkpoint', dest='checkpoint', default=True, action='store_true',
                        help='Enables checkpoint saving of model')
experiment.add_argument('--checkpoint_per_batch', default=10000, type=int,
                        help='Save checkpoint per batch. 0 means never save [default: 10000]')
experiment.add_argument('--save_folder', default='../output/',  # TODO
                        help='Location to save epoch models, training configurations and results.')
experiment.add_argument('--log_config', default=True, action='store_true', help='Store experiment configuration')
experiment.add_argument('--log_result', default=True, action='store_true', help='Store experiment result')
experiment.add_argument('--log_interval', type=int, default=20,
                        help='how many steps to wait before logging training status [default: 1]')
experiment.add_argument('--val_interval', type=int, default=1000,
                        help='how many steps to wait before vaidation [default: 400]')
experiment.add_argument('--save_interval', type=int, default=1,
                        help='how many epochs to wait before saving [default:1]')


def train(train_loader, dev_loader, model, args):
    # clip gradient
    if args.grad_clip:
        clip = paddle.nn.ClipGradByValue(max=args.grad_clip)
    else:
        clip = None
    # optimization scheme
    if args.optimizer == 'Adam':
        optim = optimizer.Adam(parameters=model.parameters(), learning_rate=args.lr, grad_clip=clip)
    elif args.optimizer == 'Adagrad':
        optim = optimizer.Adagrad(parameters=model.parameters(), learning_rate=args.lr, grad_clip=clip)
    elif args.optimizer == 'AdamW':
        optim = optimizer.AdamW(parameters=model.parameters(), learning_rate=args.lr, grad_clip=clip)

    # loss
    criterion = nn.CrossEntropyLoss()

    # continue training from checkpoint model
    if args.continue_from:
        print("=> loading checkpoint from '{}'".format(args.continue_from))
        assert os.path.isfile(args.continue_from), "=> no checkpoint found at '{}'".format(args.continue_from)
        checkpoint = paddle.load(args.continue_from)
        start_epoch = checkpoint['epoch']
        start_iter = checkpoint.get('iter', None)
        best_acc = checkpoint.get('best_acc', None)
        if start_iter is None:
            start_epoch += 1  # Assume that we saved a model after an epoch finished, so start at the next epoch.
            start_iter = 1
        else:
            start_iter += 1
        model.set_state_dict(checkpoint['state_dict'])
    else:
        start_epoch = 1
        start_iter = 1
        best_acc = None

    model.train()

    for epoch in range(start_epoch, args.epochs + 1):
        _i_batch = 0
        for i_batch, batch in enumerate(train_loader, start=start_iter):
            _i_batch = i_batch
            premises = paddle.to_tensor(batch["premise"])
            premises_lengths = paddle.to_tensor(batch["premise_length"])
            hypothesises = paddle.to_tensor(batch["hypothesis"])
            hypothesis_lengths = paddle.to_tensor(batch["hypothesis_length"])
            target = paddle.to_tensor(batch["label"])

            logit, penalty = model(premises, premises_lengths, hypothesises, hypothesis_lengths)
            loss = criterion(logit, target)
            if args.use_penalty:
                loss += args.penalty_c*penalty
            loss.backward()
            optim.step()
            optim.clear_grad()

            if args.verbose:
                print('\nTargets, Predicates')
                print(paddle.concat(
                    (target.unsqueeze(1), paddle.unsqueeze(paddle.argmax(logit, 1).reshape(target.shape), 1)), 1))
                print('\nLogit')
                print(logit)

            if i_batch % args.log_interval == 0:
                corrects = paddle.to_tensor((paddle.argmax(logit, 1) == target), dtype='int64').sum().numpy()[0]
                accuracy = 100.0 * corrects / args.batch_size
                print('Epoch[{}] Batch[{}] - loss: {:.5f}  lr: {:.5f}  acc: {:.2f}% {}/{}'.format(epoch,
                                                                                                  i_batch,
                                                                                                  loss.numpy()[0],
                                                                                                  optim._learning_rate,
                                                                                                  accuracy,
                                                                                                  corrects,
                                                                                                  args.batch_size,
                                                                                                  ))
            # validation
            if i_batch % args.val_interval == 0:
                val_loss, val_acc = eval(dev_loader, model, epoch, i_batch, optim, args)
                if best_acc is None or val_acc > best_acc:
                    file_path = '%s/SelfAttnSent_best.pth.tar' % (args.save_folder)
                    print("\r=> found better validated model, saving to %s" % file_path)
                    save_checkpoint(model,
                                    {'epoch': epoch,
                                     'optimizer': optim.state_dict(),
                                     'best_acc': best_acc},
                                    file_path)
                    best_acc = val_acc

        if args.checkpoint and epoch % args.save_interval == 0:
            file_path = '%s/SelfAttnSent_epoch_%d.pth.tar' % (args.save_folder, epoch)
            print("\r=> saving checkpoint model to %s" % file_path)
            save_checkpoint(model, {'epoch': epoch,
                                    'optimizer': optim.state_dict(),
                                    'best_acc': best_acc},
                            file_path)

        print('\n')


def eval(data_loader, model, epoch_train, batch_train, optim, args):
    model.eval()
    corrects, avg_loss, accumulated_loss, size = 0, 0, 0, 0
    predicates_all, target_all = [], []
    for i_batch, batch in enumerate(data_loader):
        premises = paddle.to_tensor(batch["premise"])
        premises_lengths = paddle.to_tensor(batch["premise_length"])
        hypothesises = paddle.to_tensor(batch["hypothesis"])
        hypothesis_lengths = paddle.to_tensor(batch["hypothesis_length"])
        target = paddle.to_tensor(batch["label"])

        size += len(target)
        target = target.squeeze()
        logit, penalty = model(premises, premises_lengths, hypothesises, hypothesis_lengths)
        predicates = paddle.argmax(logit, 1)
        accumulated_loss += F.cross_entropy(logit, target).numpy()[0]
        corrects += paddle.to_tensor((paddle.argmax(logit, 1) == target), dtype='int64').sum().numpy()[0]
        predicates_all += predicates.cpu().numpy().tolist()
        target_all += target.cpu().numpy().tolist()

    avg_loss = accumulated_loss / size
    accuracy = 100.0 * corrects / size
    model.train()
    print('\nEvaluation - loss: {:.5f}  lr: {:.5f}  acc: {:.2f} ({}/{}) error: {:.2f}'.format(avg_loss,
                                                                                              optim._learning_rate,
                                                                                              accuracy,
                                                                                              corrects,
                                                                                              size,
                                                                                              100.0 - accuracy))
    print_f_score(predicates_all, target_all)
    print('\n')
    if args.log_result:
        with open(os.path.join(args.save_folder, 'result.csv'), 'a') as r:
            r.write('\n{:d},{:d},{:.5f},{:.2f},{:f}'.format(epoch_train,
                                                            batch_train,
                                                            avg_loss,
                                                            accuracy,
                                                            optim._learning_rate))

    return avg_loss, accuracy


def save_checkpoint(model, state, filename):
    state['state_dict'] = model.state_dict()
    paddle.save(state, filename)


def make_data_loader(dataset_path, batch_size, num_workers, is_shuffle=False, data_augment=False):
    print("\nLoading data from {}".format(dataset_path))
    with open(dataset_path, "rb") as pkl:
        dataset = NLIDataset(pickle.load(pkl))

    dataset_loader = DataLoader(dataset, shuffle=is_shuffle, num_workers=num_workers, batch_size=batch_size, drop_last=True)
    return dataset, dataset_loader


def main():
    print(paddle.__version__)
    # parse arguments
    args = parser.parse_args()
    # gpu
    if args.cuda and args.gpu:
        paddle.set_device(f"gpu:{args.gpu}")

    # load train and dev data
    train_dataset, train_loader = make_data_loader(args.train_path, args.batch_size, args.num_workers, is_shuffle=True)
    dev_dataset, dev_loader = make_data_loader(args.val_path, args.batch_size, args.num_workers, is_shuffle=False)

    # load embeddings
    print("\nLoading embeddings from {}".format(args.embed_path))
    with open(args.embed_path, "rb") as pkl:
        embeddings = paddle.to_tensor(pickle.load(pkl), dtype='float64')

    # get class weights
    class_weight, num_class_train = train_dataset.get_class_weight()
    _, num_class_dev = dev_dataset.get_class_weight()


    print('\nNumber of training samples: {}'.format(str(train_dataset.__len__())))
    for i, c in enumerate(num_class_train):
        print("\tLabel {:d}:".format(i).ljust(15) + "{:d}".format(c).rjust(8))
    print('\nNumber of developing samples: {}'.format(str(dev_dataset.__len__())))
    for i, c in enumerate(num_class_dev):
        print("\tLabel {:d}:".format(i).ljust(15) + "{:d}".format(c).rjust(8))

    # make save folder
    try:
        os.makedirs(args.save_folder)
    except OSError as e:
        if e.errno == errno.EEXIST:
            print('Directory already exists.')
        else:
            raise

    # configuration
    print("\nConfiguration:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}:".format(attr.capitalize().replace('_', ' ')).ljust(25) + "{}".format(value))

    # log result
    if args.log_result:
        with open(os.path.join(args.save_folder, 'result.csv'), 'w') as r:
            r.write('{:s},{:s},{:s},{:s},{:s}'.format('epoch', 'batch', 'loss', 'acc', 'lr'))
    # model
    model = SelfAttnSent(
        batch_size=args.batch_size,
        lstm_hid_dim=args.lstm_hid_dim,
        embeddings=embeddings,
        vocab_size=embeddings.shape[0],
        emb_dim=embeddings.shape[1],
        d_a=args.d_a,
        r=args.r,
        output_hid_dim=args.output_hid_dim,
        dropout=args.dropout
    )
    print(model)

    # train
    train(train_loader, dev_loader, model, args)


if __name__ == '__main__':
    main()
