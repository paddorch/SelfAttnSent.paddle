# SelfAttnSent.paddle
A PaddlePaddle implementation of A Structured Self-attentive Sentence Embedding.

## Introduction

![](images/model.png)

论文: [A Structured Self-attentive Sentence Embedding](https://arxiv.org/pdf/1703.03130v1.pdf)

## Results

| Datasets | Paper accuracy | Our accuracy | abs. improv. |
| -------- | -------------- | ------------ | ------------ |
| SNLI     | 84.4           | 84.66        | 0.26         |

## Requirement

- Python >= 3
- PaddlePaddle >= 2.0.0
- see `requirements.txt`

## Usage

### Train
1. 下载数据集到 `data/raw` 文件夹，下载`GloVe`预训练向量，并在`config/snli_preprocess.json`中配置路径，运行`preprocess.py`
```shell
python preprocess.py
```

2. 开始训练
```shell
bash train.sh
```

### Download Trained model

[SNLI model](https://cowtransfer.com/s/16d77730356445)

将模型分别放置于 `output` 目录下，如下运行 `eval` bash 脚本即可测试模型。

### Test

```shell
bash eval.sh
```

可以得到如下结果：

![result](images/result.png)

## Detail

我们采用了与原文对应代码仓库基本一致的实验设置，不同之处在于我们首先在固定词嵌入层的情况下训练了 10 个 epoch，随后将词嵌入层加入训练。实验显示在本任务中这种训练方式能够有效提升模型性能。

对于原论文与其对应实现设置的冲突之处，我们选择参考后者。

## References
```bibtex
@misc{lin2017structured,
      title={A Structured Self-attentive Sentence Embedding}, 
      author={Zhouhan Lin and Minwei Feng and Cicero Nogueira dos Santos and Mo Yu and Bing Xiang and Bowen Zhou and Yoshua Bengio},
      year={2017},
      eprint={1703.03130},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

- https://github.com/hantek/SelfAttentiveSentEmbed