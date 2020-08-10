
# Deep Lifetime Clustering

This repository is the official implementation of [Deep Lifetime Clustering](https://arxiv.org/abs/1910.00547).

> Abstract: The goal of lifetime clustering is to develop an inductive model that maps subjects into K clusters according to their underlying (unobserved) lifetime distribution. We introduce a neural-network based lifetime clustering model that can find cluster assignments by directly maximizing the divergence between the empirical lifetime distributions of the clusters. Accordingly, we define a novel clustering loss function over the lifetime distributions (of entire clusters) based on a tight upper bound of the two-sample Kuiper test p-value. The resultant model is robust to the modeling issues associated with the unobservability of termination signals, and does not assume proportional hazards. Our results in real and synthetic datasets show significantly better lifetime clusters (as evaluated by C-index, Brier Score, Logrank score and adjusted Rand index) as compared to competing approaches.

## Requirements

The necessary libraries are:
- [NumPy](https://www.numpy.org/)
- [PyTorch](https://pytorch.org/) 
- [Lifelines](https://lifelines.readthedocs.io/en/latest/)

Please refer to the respective documentations for installation instructions.

Datasets available for download (defined in datasets.py): 
1. Friendster dataset with ~1M users (processed as described in the paper)
2. Synthetic dataset with 3 clusters (see Figure 2(b) in the paper)

## Training

To train the model in the paper, run this command:

```train
# Lifetime Clustering on Synthetic dataset (download the dataset) with K = 2 (number of clusters). 
python main.py --dataset=synthetic --download --k=2 --epochs=10 --save

# Lifetime Clustering on Friendster dataset with K = 2, 1000 training samples and 1000 test samples. 
python main.py --dataset=friendster --Ntrain=1000 --Ntest=1000 --k=2 --epochs=10 --save
```

Main arguments to main.py are:
```
--dataset = friendster or synthetic
--k = integer (number of clusters)
--lossName = kuiper_ub or mmd (Loss for Lifetime clustering: Kuiper upper bound or MMD)
--eol = True or False (End-of-life signals learnt or not)

--Ntrain = integer (Number of training samples to choose from training fold)
--Ntest = integer (Number of test samples to choose from test fold. For faster debugging.)

--download (to download the dataset) 
--show (show result plots in Jupyter)
--save (save result plots in plot_train.pdf and plot_test.pdf)
```

Other arguments include ``--batchSize``, ``--epochs``, ``--lr``, ``--seed`` with the usual meanings.

## Questions
Please feel free to reach out to S Chandra Mouli (chandr at purdue.edu) if you have any questions.

## Citation
If you use the data or code from this repository in your own code, please cite
our paper:
```tex
@article{mouli2019deep,
  title={Deep Lifetime Clustering},
  author={Mouli, S Chandra and Teixeira, Leonardo and Ribeiro, Bruno and Neville, Jennifer},
  journal={arXiv preprint arXiv:1910.00547},
  year={2019}
}
```



