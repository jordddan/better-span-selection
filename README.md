## Title

This repository contains the code for our paper [](), which includes our proposed Span Selection method and datasets.



## Quick Links
  - [Overview](#overview)
  - [Datasets](#datasets)
    - [Real-world distantly supervised datasets](#real-world-distantly-supervised-datasets)
    - [Synthetic Datasets for UEP](#synthetic-datasets-for-uep)
  - [Quick Start](#quick-start)
  - [Main Results](#main-results)


## Overview

In this work, we explore the improvement of the Named-Entity-Recognition (NER) task under distant supervision.
As the labeled data constructed by distant supervision inevitably contains wrongly annotated samples, we first count the ratios of different wrong annotation types in a preliminary experiment and observe two significant problems, i.e., Unlabeled-Entity Problem (UEP) and Noisy Entity Problem (NEP).
Through further revisiting existing works with experimental trials and theoretical analysis in addressing the above two problems, we observe more insights into managing the UEP problem and propose a two-stage training strategy accordingly.
We also present an effective strategy tailored to the NEP problem. 

![](figure/TTA.png)

## Datasets

We conduct experiments on four real-world distantly supervised NER benchmarks and synthetic datasets constructed for UEP.

### Real-world distantly supervised datasets

| Dataset |  Origin   | DS version |
| :-----  | :-------: | :--------: |
| CoNLL   | [here](https://github.com/patverga/torch-ner-nlp-from-scratch/tree/master/data/conll2003/) | [here](https://github.com/cliang1453/BOND/tree/master/dataset/conll03_distant) |
| Webpage | [here](http://cogcomp.seas.upenn.edu/Data/NERWebpagesColumns.tgz) | [here](https://github.com/cliang1453/BOND/tree/master/dataset/webpage_distant) |
| Twitter | [here](https://github.com/aritter/twitter_nlp/tree/master/data/annotated/wnut16) | [here](https://github.com/cliang1453/BOND/tree/master/dataset/twitter_distant) |
| BC5CDR  | [here](https://github.com/kangISU/Conf-MPU-BERT-DS-NER/tree/master/data/BC5CDR_Fully) | [here](https://github.com/kangISU/Conf-MPU-BERT-DS-NER/tree/master/data/BC5CDR_Dict_1.0) |


### Synthetic Datasets for UEP

We randomly mask a certain proportion (from 0.4 to 0.9) of entity spans and treat them as non-entities to simulate the scenario of UEP.

## Quick Start

```bash
bash train.sh [PLM NAME] [DATASET NAME] [GPU ID] [SEED]
```

Our results are the average of 5 runs.

## Main Results


### real-word dataset

| Dataset | CoNLL | Webpage | Twitter | BC5CDR |
| :------ | :---: | :-----: | :-----: | :----: |
| ours    | **82.02** |  68.56  | **48.51**   | 74.09  |
| ours w/o NPE | 78.82 | **70.01** | 48.34 | **74.89** |

<!-- ### synthetic dataset -->
