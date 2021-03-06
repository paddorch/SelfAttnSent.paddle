# !/usr/bin/env python3
from termcolor import cprint, colored as c
import numpy as np
import random
import paddle


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def inc(d, label):
    if label in d:
        d[label] += 1
    else:
        d[label] = 1


def precision_recall(output, target):
    assert len(output) == len(target), "output len: {} != target len: {}".format(len(output), len(target))
    labels = set(target)
    TP = {}
    TP_plus_FN = {}
    TP_plus_FP = {}
    for i in range(len(output)):

        inc(TP_plus_FN, target[i])
        inc(TP_plus_FP, output[i])
        if target[i] == output[i]:
            inc(TP, output[i])

    for label in labels:
        if label not in TP_plus_FN:
            TP_plus_FN[label] = 0
        if label not in TP_plus_FP:
            TP_plus_FP[label] = 0

    precision = {label: 0. if TP_plus_FP[label] == 0 else ((TP[label] if label in TP else 0) / float(TP_plus_FP[label]))
                 for label in labels}
    recall = {label: 0. if TP_plus_FN[label] == 0 else ((TP[label] if label in TP else 0) / float(TP_plus_FN[label]))
              for label in labels}

    return precision, recall, TP, TP_plus_FN, TP_plus_FP


def F_score(p, r):
    f_scores = {
        label: None if p[label] == 0 and r[label] == 0 else (
            0 if p[label] == 0 or r[label] == 0 else 2 / (1 / p[label] + 1 / r[label]))
        for label in p
    }
    return f_scores


def print_f_score(output, target):
    """returns:
        p<recision>,
        r<ecall>,
        f<-score>,
        {"TP", "p", "TP_plus_FP"} """
    p, r, TP, TP_plus_FN, TP_plus_FP = precision_recall(output, target)
    f = F_score(p, r)

    # cprint("Label: " + c(("  " + str(10))[-5:], 'red') +
    #            "\tPrec: " + c("  {:.1f}".format(0.335448 * 100)[-5:], 'green') + '%' +
    #            " ({:d}/{:d})".format(1025, 1254).ljust(14) +
    #            "Recall: " + c("  {:.1f}".format(0.964 * 100)[-5:], 'green') + "%" +
    #            " ({:d}/{:d})".format(15, 154).ljust(14) +
    #            "F-Score: " +  (c("  {:.1f}".format(0.5 * 100)[-5:], "green") + "%")
    #            )

    for label in f.keys():
        cprint("Label: " + c(("  " + str(label))[-5:], 'red') +
               "\tPrec: " + c("  {:.1f}".format(p[label] * 100)[-5:], 'green') + '%' +
               " ({:d}/{:d})".format((TP[label] if label in TP else 0), TP_plus_FP[label]).ljust(14) +
               "Recall: " + c("  {:.1f}".format((r[label] if label in r else 0) * 100)[-5:], 'green') + "%" +
               " ({:d}/{:d})".format((TP[label] if label in TP else 0), TP_plus_FN[label]).ljust(14) +
               "F-Score: " + (
                   "  N/A" if f[label] is None else (c("  {:.1f}".format(f[label] * 100)[-5:], "green") + "%"))
               )
    # return p, r, f, _


if __name__ == '__main__':
    import paddle

    output = [1, 1, 1, 1, 1, 2, 0, 2, 2, 2, 2]
    output = paddle.to_tensor(output, dtype="int64")
    # target = [0,0,2,1,2,2,1,2,1,2,0]

    target = [1, 3, 2, 3, 3, 3, 3, 3, 0, 3, 3]

    target = paddle.to_tensor(target, dtype="int64")
    # output = autograd.Variable(output)
    # target = autograd.Variable(target)
    print('output')
    print(output.numpy().tolist())
    print('target')
    print(target.numpy().tolist())

    precision, recall, TP, TP_plus_FN, TP_plus_FP = precision_recall(output.numpy().tolist(), target.numpy().tolist())
    print('precision')
    print(precision)
    print('recall')
    print(recall)
    print('TP')
    print(TP)
    print('TP_plus_FN')
    print(TP_plus_FN)
    print('TP_plus_FP')
    print(TP_plus_FP)
    # print(dic)

    f_scores = F_score(precision, recall)
    print('f_scores')
    print(f_scores)
    # print(f_scores.keys())

    print('\r')
    print_f_score(output.numpy().tolist(), target.numpy().tolist())


def sort_by_seq_lens(batch, sequences_lengths, descending=True):
    """
    Sort a batch of padded variable length sequences by their length.

    Args:
        batch: A batch of padded variable length sequences. The batch should
            have the dimensions (batch_size x max_sequence_length x *).
        sequences_lengths: A tensor containing the lengths of the sequences in the
            input batch. The tensor should be of size (batch_size).
        descending: A boolean value indicating whether to sort the sequences
            by their lengths in descending order. Defaults to True.

    Returns:
        sorted_batch: A tensor containing the input batch reordered by
            sequences lengths.
        sorted_seq_lens: A tensor containing the sorted lengths of the
            sequences in the input batch.
        sorting_idx: A tensor containing the indices used to permute the input
            batch in order to get 'sorted_batch'.
        restoration_idx: A tensor containing the indices that can be used to
            restore the order of the sequences in 'sorted_batch' so that it
            matches the input batch.
    """
    sorted_seq_lens, sorting_index =\
        sequences_lengths.sort(0, descending=descending)

    sorted_batch = batch.index_select(0, sorting_index)

    idx_range =\
        sequences_lengths.new_tensor(paddle.arange(0, len(sequences_lengths)))
    _, reverse_mapping = sorting_index.sort(0, descending=False)
    restoration_index = idx_range.index_select(0, reverse_mapping)

    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index
