import os
import argparse
import datetime
import sys
import errno
import pickle
from tqdm import tqdm

import paddle
from paddle.io import DataLoader
import paddle.nn.functional as F

from model import SelfAttnSent
from data import NLIDataset
from utils import print_f_score

parser = argparse.ArgumentParser(description='Structured self-attention sentence embedding testing')
# model
parser.add_argument('--model_path', default='../output/SelfAttnSent_best.pth.tar',
                    help='Path to pre-trained acouctics model created by DeepSpeech training')
parser.add_argument('--lstm_hid_dim', type=int, default=300,
                       help='hidden size of LSTM [default: 300]')
parser.add_argument('--dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('--use_penalty', action='store_true', default=False, help='whether enable penalty')
parser.add_argument('--penalty_c', type=float, default=0.1, help='penalty term coefficient [default: 0.1]')
parser.add_argument('--d_a', type=int, default=150,
                       help='d_a size [default: 150]')
parser.add_argument('--r', type=int, default=30,
                       help='row size of sentence embedding [default: 30]')
parser.add_argument('--output_hid_dim', type=int, default=4000,
                       help='mlp hidden size [default: 4000]')
# data
parser.add_argument('--test-path', metavar='DIR',
                    help='path to testing data csv', default='../data/preprocessed/test_data.pkl')
parser.add_argument('--embed_path', metavar='DIR',
                    help='path to embedding [default: data/ag_news_csv/test.csv]',
                    default='../data/preprocessed/embeddings.pkl')
parser.add_argument('--batch-size', type=int, default=50, help='batch size for training [default: 128]')
# device
parser.add_argument('--num-workers', default=8, type=int, help='Number of workers used in data-loading')
parser.add_argument('--cuda', action='store_true', default=True, help='enable the gpu')
parser.add_argument('--device', type=str, default='gpu:0')
# logging options
parser.add_argument('--save-folder', default='Results/', help='Location to save epoch models')
args = parser.parse_args()


if __name__ == '__main__':
    paddle.set_device(args.device)

    # load testing data
    print("\nLoading testing data...")
    with open(args.test_path, "rb") as pkl:
        test_dataset = NLIDataset(pickle.load(pkl))
    print("Transferring testing data to iterator...")
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, drop_last=True)

    _, num_class_test = test_dataset.get_class_weight()
    print('\nNumber of testing samples: ' + str(test_dataset.__len__()))
    for i, c in enumerate(num_class_test):
        print("\tLabel {:d}:".format(i).ljust(15) + "{:d}".format(c).rjust(8))

    print("\nLoading embeddings from {}".format(args.embed_path))
    with open(args.embed_path, "rb") as pkl:
        embeddings = paddle.to_tensor(pickle.load(pkl), dtype='float64')

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
    print("=> loading weights from '{}'".format(args.model_path))
    assert os.path.isfile(args.model_path), "=> no checkpoint found at '{}'".format(args.model_path)
    checkpoint = paddle.load(args.model_path)
    model.set_state_dict(checkpoint['state_dict'])

    model.eval()
    corrects, avg_loss, accumulated_loss, size = 0, 0, 0, 0
    predicates_all, target_all = [], []
    print('\nTesting...')
    for i_batch, batch in enumerate(tqdm(test_loader)):
        premises = paddle.to_tensor(batch["premise"])
        premises_lengths = paddle.to_tensor(batch["premise_length"])
        hypothesises = paddle.to_tensor(batch["hypothesis"])
        hypothesis_lengths = paddle.to_tensor(batch["hypothesis_length"])
        target = paddle.to_tensor(batch["label"])
        size += len(target)

        logit, _ = model(premises, premises_lengths, hypothesises, hypothesis_lengths)
        predicates = paddle.argmax(logit, 1)
        accumulated_loss += F.cross_entropy(logit, target).numpy()[0]
        corrects += paddle.to_tensor((paddle.argmax(logit, 1) == target), dtype='int64').sum().numpy()[0]
        predicates_all += predicates.cpu().numpy().tolist()
        target_all += target.cpu().numpy().tolist()

    avg_loss = accumulated_loss / size
    accuracy = 100.0 * corrects / size
    print('\rEvaluation - loss: {:.6f}  acc: {:.2f}%({}/{}) error: {:.2f}'.format(avg_loss,
                                                                     accuracy,
                                                                     corrects,
                                                                     size,
                                                                     100 - accuracy))
    print_f_score(predicates_all, target_all)