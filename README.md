# Probabilistic Multi-Dimensional Classification
This repository contains the code for the paper [Probabilistic Multi-Dimensional Classification](https://openreview.net/forum?id=3WnTBadc-J) (*UAI 2023*)

**Authors:** Vu-Linh Nguyen*, Yang Yang*, Cassio de Campos

**TL;DR:** In this paper, we present a first attempt to learn probabilistic multi-dimensional classifiers which are interpretable, accurate, scalable and capable of handling mixed data.

## Requirements

The required packages are listed in `requirements.txt`.

## Data

Please check the experiment section of our paper for a collection of datasets used in the experiments.

## Usage

To replicate experiments, check the scripts in the `experiments` folder.

To run experiments on tabular data sets with only continuous features, you can call

```shell
python cont.py Rf1 --base lr --palim 3 --n-chains 10 --no-plot --n-folds 10
```

```shell
usage: cont.py [-h] [--base {lr,nb}] [--palim PALIM] [--n-chains N_CHAINS] [--plot | --no-plot] [--n-folds N_FOLDS] [--output OUTPUT]
               {Edm,Jura,Enb,Voice,Song,Flickr,Fera,WQplants,WQanimals,Rf1,Pain,Disfa,WaterQuality,Oes97,Oes10,Scm20d,Scm1d}

Tabular Continuous Features Experiments

positional arguments:
  {Edm,Jura,Enb,Voice,Song,Flickr,Fera,WQplants,WQanimals,Rf1,Pain,Disfa,WaterQuality,Oes97,Oes10,Scm20d,Scm1d}
                        Dataset

options:
  -h, --help            show this help message and exit
  --base {lr,nb}        Base classifier
  --palim PALIM         The maximum number of parents for each node
  --n-chains N_CHAINS   Number of randomly generated classifier chains
  --plot, --no-plot     Whether or not to plot the BN structure
  --n-folds N_FOLDS     Number of folds
  --output OUTPUT       Output path
```

To run experiments on tabular data sets with both continuous and discrete features, when using a ***logistic regression*** classifier as the base learner, you can call

```shell
python disc_lr.py Adult --palim 3 --disclim 3 --n-chains 10 --no-plot --n-folds 10
```

```shell
usage: disc_lr.py [-h] [--palim PALIM] [--disclim DISCLIM] [--n-chains N_CHAINS] [--plot | --no-plot] [--n-folds N_FOLDS] [--output OUTPUT] {Adult,Default,Thyroid}

Mixed Data Experiments with Logistic Regression

positional arguments:
  {Adult,Default,Thyroid}
                        Dataset

options:
  -h, --help            show this help message and exit
  --palim PALIM         The maximum number of parents for each node
  --disclim DISCLIM     The maximum number of discrete features for each node
  --n-chains N_CHAINS   Number of randomly generated classifier chains
  --plot, --no-plot     Whether or not to plot the BN structure
  --n-folds N_FOLDS     Number of folds
  --output OUTPUT       Output path
```

To run experiments on tabular data sets with both continuous and discrete features, when using a ***naive bayes*** classifier as the base learner, you can call

```shell
python3 disc_nb.py Adult --palim 3 --disclim 3 --n-chains 10 --no-plot --n-folds 10
```

```shell
usage: disc_nb.py [-h] [--palim PALIM] [--disclim DISCLIM] [--n-chains N_CHAINS] [--plot | --no-plot] [--n-folds N_FOLDS] [--output OUTPUT] {Adult,Default,Thyroid}

Mixed Data Experiments with Naive Bayes

positional arguments:
  {Adult,Default,Thyroid}
                        Dataset

options:
  -h, --help            show this help message and exit
  --palim PALIM         The maximum number of parents for each node
  --disclim DISCLIM     The maximum number of discrete features for each node
  --n-chains N_CHAINS   Number of randomly generated classifier chains
  --plot, --no-plot     Whether or not to plot the BN structure
  --n-folds N_FOLDS     Number of folds
  --output OUTPUT       Output path
```

To run experiments on image data sets, you can call

```shell
python3 img.py voc2007 --base resnet18 --batch-size 128 --lr 0.01 --n-epochs 20 --palim 3 --no-plot --mixup --calibrate
```

```shell
usage: img.py [-h] [--base {resnet18}] [--batch-size BATCH_SIZE] [--lr LR] [--n-epochs N_EPOCHS] [--palim PALIM] [--plot | --no-plot] [--mixup | --no-mixup]
              [--calibrate | --no-calibrate] [--output OUTPUT]
              {voc2007}

Image Experiments

positional arguments:
  {voc2007}             Dataset

options:
  -h, --help            show this help message and exit
  --base {resnet18}     Base classifier
  --batch-size BATCH_SIZE
                        Batch size
  --lr LR               Learning rate
  --n-epochs N_EPOCHS   Number of epochs
  --palim PALIM         The maximum number of parents for each node
  --plot, --no-plot     Whether or not to plot the BN structure
  --mixup, --no-mixup   Whether or not to use mixup training
  --calibrate, --no-calibrate
                        Whether or not to calibrate
  --output OUTPUT       Output path
```