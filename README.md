# ConvBERT

## Introduction

In this repo, we introduce a new architecture **ConvBERT** for pre-training based language model. The code is tested on a V100 GPU. For detailed description and experimental results, please refer to our NeurIPS 2020 paper [ConvBERT: Improving BERT with Span-based Dynamic Convolution](https://arxiv.org/abs/2008.02496).

## Requirements
* Python 3
* tensorflow 1.15
* numpy
* scikit-learn

## Experiments


### Pre-training

These instructions pre-train a medium-small sized ConvBERT model (17M parameters)  using the [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) corpus.

To build the tf-record and pre-train the model, download the [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) corpus (12G) and **setup your data directory** in `build_data.sh` and `pretrain.sh`. Then run

```bash
bash build_data.sh
```

The processed data require roughly 30G of disk space. Then, to pre-train the model, run

```bash
bash pretrain.sh
```

See `configure_pretraining.py` for the details of the supported hyperparameters.

### Fine-tining

We gives the instruction to fine-tune a pre-trained medium-small sized ConvBERT model (17M parameters) on GLUE. You can refer to the Google Colab notebook for a [quick example](https://colab.research.google.com/drive/1WIu2Cc1C8E7ayZBzEmpfd5sXOhe7Ehhz?usp=sharing). See our paper for more details on model performance. Pre-trained model can be found [here](https://drive.google.com/drive/folders/1pSsPcQrGXyt1FB45clALUQf-WTNAbUQa?usp=sharing). (You can also download it from [baidu cloud](https://pan.baidu.com/s/1jPo0e94p2dB8UBz33QuMrQ) with extraction code m9d2.)

To evaluate the performance on GLUE, you can download the GLUE data by running
```bash
python3 download_glue_data.py
```
Set up the data by running `mv CoLA cola && mv MNLI mnli && mv MRPC mrpc && mv QNLI qnli && mv QQP qqp && mv RTE rte && mv SST-2 sst && mv STS-B sts && mv diagnostic/diagnostic.tsv mnli && mkdir -p $DATA_DIR/finetuning_data && mv * $DATA_DIR/finetuning_data`. After preparing the GLUE data, **setup your data directory** in `finetune.sh` and  run
```bash
bash finetune.sh
```
And you can test different tasks by changing configs in `finetune.sh`.

If you find this repo helpful, please consider cite
```bibtex
@article{Jiang2020ConvBERT,
  title={ConvBERT: Improving BERT with Span-based Dynamic Convolution},
  author={Zi-Hang Jiang and Weihao Yu and Daquan Zhou and Y. Chen and Jiashi Feng and S. Yan},
  journal={ArXiv},
  year={2020},
  volume={abs/2008.02496}
}
```
# References

Here are some great resources we benefit:

Codebase: Our codebase are based on [ELECTRA](https://github.com/google-research/electra).

Dynamic convolution: [Implementation](https://github.com/pytorch/fairseq/blob/265791b727b664d4d7da3abd918a3f6fb70d7337/fairseq/modules/lightconv_layer/lightconv_layer.py#L75) from [Pay Less Attention with Lightweight and Dynamic Convolutions](https://openreview.net/pdf?id=SkVhlh09tX)

Dataset: [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) from [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf)

