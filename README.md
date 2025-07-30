<div align=center>

# From Easy to Hard++: Promoting Differentially Private Image Synthesis Through Time-Frequency Curriculum
</div>

This is the official implementation of paper FETA-Pro. FETA-Pro proposes a spatial-frequency curriculum, which leverages frequency domain features to capture global structures and textures of images, complementing spatial domain features. To address the challenges of learning caused by heterogeneity between spatial and frequency domain features, FETA-Pro extracts frequency domain features using the Fourier Transformer under DP, and introduces an __auxiliary generator__ to produce images aligned the noisy features. Then, DP synthesizers use synthetic images for warm-up. 

We conduct experiments to show that across five sensitive image datasets, FETA-Pro achieves, on average, 24.5\% higher fidelity and 2.8\% greater utility compared to the state-of-the-art method under a privacy budget of $\epsilon = 1$.

<div align=center>
<img src="./plot/FETA-Pro-framework.png" width = "400" alt="The framework of FETA-Pro. During warm-up, FETA-Pro extracts spatial features to train the synthesizer. Then, FETA-Pro introduces auxiliary generators to generate images aligning the noisy frequency features, and training synthesizers on these synthetic images. Then, we train warm-up synthesizers on sensitive images using DP-SGD" align=center />
</div>

<p align="center">The framework of FETA-Pro. During warm-up, FETA-Pro extracts spatial features to train the synthesizer. Then, FETA-Pro introduces auxiliary generators to generate images aligning the noisy frequency features, and training synthesizers on these synthetic images. Then, we train warm-up synthesizers on sensitive images using DP-SGD.</p>

## 1. Contents
  - [1. Contents](#1-contents)
  - [2. Introduction](#2-introduction)
    - [2.1 Baselines](#21-baselines)
    - [2.2 Investigated-Datasets](#22-investigated-datasets)
  - [3. Repo Contents](#3-repo-contents)
  - [4. Quick Start](#4-quick-start)
    - [4.1 Install DPImageBench](#41-install-dpimagebench)
    - [4.2 Prepare Dataset](#42-prepare-dataset)
    - [4.3 Running](#43-running)
      - [4.3.1 Key hyper-parameter introductions](#431-key-hyper-parameter-introductions)
      - [4.3.2 How to run](#432-how-to-run)
      - [4.3.3 Training using checkpoints](#433-training-using-checkpoints)
    - [4.4 Results](#44-results)
      - [4.4.1 Results Structure](#441-results-structure)
      - [4.4.2 Results Explanation](#442-results-explanation)
    - [4.5 Results Visualization](#45-results-visualization)
  - [Acknowledgment](#acknowledgement)

## 2. Introduction

### 2.1 Baselines

We list baselines as follows.

  | Methods |  Link                                                         |
  | -------------- | ------------------------------------------------------------ |
  | DP-MERF            |  [\[AISTATS 2021\] DP-MERF: Differentially Private Mean Embeddings With Randomfeatures for Practical Privacy-Preserving Data Generation](https://proceedings.mlr.press/v130/harder21a.html) |
  | DP-NTK            |  [\[AISTATS 2021\] Differentially Private Neural Tangent Kernels (DP-NTK) for Privacy-Preserving Data Generation](https://arxiv.org/html/2303.01687v2) |
  | DP-Kernel        |  [\[NeuriPS 2023\] Functional Renyi Differential Privacy for Generative Modeling](https://proceedings.neurips.cc/paper_files/paper/2023/hash/2f9ee101e35b890d9eae79ee27bcd69a-Abstract-Conference.html) |
  | GS-WGAN            |  [\[NeuriPS 2020\] GS-WGAN: A Gradient-Sanitized Approach for Learning Differentially Private Generators](https://arxiv.org/pdf/2006.08265) |
  | DP-GAN            |  [\[arXiv 2020\] Differentially Private Generative Adversarial Network (arxiv.org)](https://arxiv.org/abs/1802.06739) |
  | DPDM          |  [\[TMLR 2023\] Differentially Private Diffusion Models](https://openreview.net/forum?id=ZPpQk7FJXF) |
  | DP-FETA      |  [\[S&P 2025\] From Easy to Hard: Building a Shortcut for Differentially Private Image Synthesis](https://www.computer.org/csdl/proceedings-article/sp/2025/223600d656/26hiVDu7y5W) |

### 2.2 Investigated Datasets
We list the studied datasets as follows in our paper, which include five sensitive datasets.
  | Usage |  Dataset  |
  | ------- | --------------------- |
  | Sensitive dataset | MNIST, FashionMNIST, CIFAR-10, CelebA, Camelyon |


This repository is built upon DPImageBench. Users can find more details about its functions in their [Repository](https://github.com/2019ChenGong/DPImageBench).

## 3. Repo Contents

Below is the directory structure of the DPImageBench project, which organizes its two core functionalities within the `models/` and `evaluation/` directories. To enhance user understanding and showcase the toolkit's ease of use, we offer a variety of example scripts located in the `scripts/` directory.


```plaintext
DPImageBench/
├── configs/                    # Configuration files for various DP image synthesis algorithms
├── data/                       # Data Preparation for Our Benchmark
├── dataset/                    # Datasets studied in the project
├── docker/                     # Docker file
├── exp/                        # The output of the training process and evaluation results.
├── evaluation/                 # Evaluation module of DPImageBench, including utility and fidelity
│   ├── classifier/             # Downstream tasks classification training algorithms
│   │   ├── densenet.py  
│   │   ├── resnet.py 
│   │   ├── resnext.py 
│   │   └── wrn.py 
│   ├── ema.py 
│   └── evaluator.py 
├── models/                     # Implementation framework for DP image synthesis algorithms
├── opacus/                     # Implementation of DPSGD
├── results_demo/               # Demo of experimental results
├── plot/                       # Figures and plots in our paper
│   ├── ablation.py                           # Plotting for Figure 5 in RQ2
│   ├── fid_curve.py                          # Plotting for Figure 4 in RQ1
│   ├── plot_eps_changes.py                   # Plotting for Figure 7 in RQ3
│   ├── plot_heatmap.py                       # Plotting for Figure 6 in RQ3
│   ├── visualization_tf.py                   # Plotting for Figure 2   
│   └── visualization.py                      # Plotting for Figure 3 in RQ1 
├── scripts/                                  # Scripts for using DPImageBench
│   └── script-feta-pro.sh                                                                    
├── utils/                      # Helper classes and functions supporting various operations
│   └── utils.py                    
├── README.md                   # Main project documentation
├── cal_privacy.py              # Calculate the privacy budget ratio in RDP
└── requirements.txt            # Dependencies required for the project
```

## 4. Quick Start

### 4.1 Install DPImageBench

Clone repo and setup the environment:

 ```
git clone git@github.com:2019ChenGong/Feta-Pro.git
sh install.sh
 ```

We also provide the [Docker](./docker/Dockerfile) file.

### 4.2 Prepare Dataset

 ```
sh scripts/data_preparation.sh
 ```

After running, we can found the folder `dataset`:

  ```plaintext
dataset/                                  
├── camelyon/       
├── celeba/ 
├── cifar10/ 
...
```

### 4.3 Running

The training and evaluatin codes are `run.py` and `eval.py`.

The core codes of `run.py` are present as follows.

```python
def main(config):

    initialize_environment(config)

    model, config = load_model(config)

    sensitive_train_loader, sensitive_val_loader, sensitive_test_loader, public_train_loader, config = load_data(config)

    model.pretrain(public_train_loader, config.pretrain)

    model.train(sensitive_train_loader, config.train)

    syn_data, syn_labels = model.generate(config.gen)

    evaluator = Evaluator(config)
    evaluator.eval(syn_data, syn_labels, sensitive_train_loader, sensitive_val_loader, sensitive_test_loader)
```

#### 4.3.1 Key hyper-parameter introductions

We list the key hyper-parameters below, including their explanations and available options.

- `--data_name` (`-dn`): means the sensitive dataset; the option is [`mnist_28`, `fmnist_28`, `cifar10_32`, `cifar100_32`, `eurosat_32`, `celeba_male_32`, `camelyon_32`].
- `--method` (`-m`): the method to train the DP image synthesizers; the option is [`DP-NTK`, `DP-Kernel`, `DP-MERF`, `DPGAN`, `DP-LDM-SD`, `DP-LDM`, `DP-LORA`, `DPDM`, `PE`, `GS-WGAN`, `PDP-Diffusion`, `PrivImage`, `DP-FETA-Pro`].
- `--epsilon` (`-e`): the privacy budget 10.0; the option is [`1.0`, `10.0`].
- `--exp_description` (`-ed`): the notes for the name of result folders.
- `setup.n_gpus_per_node`: means the number of GPUs to be used for training.
- `pretrain.cond`: specifies the mode of pretraining. The options are [`true`, `false`], where `true` indicates conditional pretraining and `false` indicates conditional pretraining.
- `public_data.name`: the name of pretraining dataset; the option is [`null`, `imagenet`, `places365`, `emnist`], which mean that without pretraining, using ImageNet dataset as pretraining dataset, and using Places365 as pretraining dataset. It is notice that DPImageBench uses ImageNet as default pretraining dataset. If users use Places365 as pretraining dataset, please add `public_data.n_classes=365 public_data.train_path=dataset/places365`.
- `eval.mode`: the mode of evaluations; the option is [`val`, `syn`] which means that using part of sensitive images and directly using the synthetic images as the validation set for model selection, respectively. The default setting is `val`.
- `setup.master_port`: a configuration parameter specifying the port number on the master node (or primary process) that other processes or nodes use to communicate within a distributed system.
- `pretrain.n_epochs`: the number of epoch for pretraining.
- `train.n_epochs`: the number of epoch for finetuning on sensitive datasets.
- `train.dp.n_split`: the number of gradient accumulations for saving GPU memory usage.

> [!Tip]
>
> Experiments such as pretraining or using DPSGD require significant computational resources. We recommend to use 4 NVIDIA GeForce A6000 Ada GPUs and 512GB of memory. Here are some tips to help users efficiently reduce computational resource usage and running time in an appropriate way:
> - Reduce `pretrain.n_epochs` and `train.n_epochs`: Reducing the number of pretraining and fine-tuning steps can decrease running time but may also impact the performance of synthetic images.
> - Increase `train.dp.n_split`: Increasing `train.dp.n_split` enables jobs to run even when GPU memory is insufficient. However, this adjustment will lead to longer running times.


> [!Warning]
>
> It is a common [issue](https://pytorch.org/docs/stable/distributed.html) that we can not run a distributed process under a `setup.master_port=6026`. If you intend to run multiple distributed processes on the same machine, please consider using a different `setup.master_port`, such as 6027.


#### 4.3.2 How to run

Users should first activate the conda environment.

```
conda activate dpimagebench
cd FETA-Pro
```
#### For the implementation of results reported in Table 3, Figure 3 and 4 (RQ1). 

We list an example as follows. Users can modify the configuration files in [configs](./configs) as their preference. 

The results reported in Table 3 were obtained by following the instructions provided. We provide an example of training a synthesizer using the DP-FETA-Pro method with 4 GPUs. 

```
python run.py setup.n_gpus_per_node=4 --method DP-FETA-Pro --data_name mnist_28 -e 1.0 eval.mode=val
```

We provide more examples in the `scripts/script-feta-pro.sh`, please refer to [scrips](scripts/script-feta-pro.sh).

Besides, if users want to directly evaluate the synthetic images,
```
python eval.py --method DP-FETA-Pro --data_name mnist_28 --epsilon 10.0 --exp_path exp/dp-feta-pro/<the-name-of-file>
```
The results are recorded in `exp/pdp-diffusion/<the-name-of-file>/stdout.txt`.

For Figure3 and Figure4, please refer to `plot/visualization.py` and `plot/fid_curve.py` and change the `log_files` in codes.

For baselines, readers can select the options `-m`: [`DP-NTK`, `DP-Kernel`, `DP-MERF`, `DPGAN`, `DP-LDM-SD`, `DP-LDM`, `DP-LORA`, `DPDM`, `PE`, `GS-WGAN`, `PDP-Diffusion`, `PrivImage`, `DP-FETA-Pro`], refering to implementations in [DPImageBench](https://github.com/2019ChenGong/DPImageBench).
```
python run.py setup.n_gpus_per_node=4 setup.master_port=6662 eval.mode=val -m DPDM -dn mnist_28 -e 1.0 -ed dpdm
```

#### For the implementation of the results reported in Table 4 and Figures 5 (RQ2).

In RQ2, to investigate the benifits of frequency features, in Figure 5, we compare the performance of DP-FETA-Pro with three invariants,

- `FETA-Pro_{ft}' denotes a synthesizer that prioritizes learning spatial domain knowledge from sensitive datasets before acquiring frequency domain knowledge.
- `FETA-Pro_{mix}' means the synthesizers simultaneously learn the spatial and frequency domain knowledge.
- `FETA-Pro_{f}' means the synthesizers solely learn the frequency domain knowledge for warm-up.

Users can set the `pretrain.mode`=[`freq_time`, `mix`, `freq`] to choose the invariants. We provide example as follows, 

```
python run.py setup.n_gpus_per_node=4 setup.master_port=6662 eval.mode=val pretrain.mode=mix -m DP-FETA-Pro -dn mnist_28 -e 1.0 -ed mix
```

To investigate the benifits of auxiliary generator, in Table 4, we compare the performance of DP-FETA-Pro with two invariants,

- `FETA-Pro-No-Auxiliary' means directly warming up synthesizers on frequency domain features. 
- `FETA-Pro-DM-Auxiliary' means using diffusion models as the auxiliary generator.

Users can set the `pretrain.mode`=[`freq_time`, `mix`] to choose the invariants. We provide example as follows, 

```
python run.py setup.n_gpus_per_node=4 setup.master_port=6662 eval.mode=val pretrain.mode=mix -m DP-FETA-Pro -dn mnist_28 -e 1.0 -ed mix
```


#### For the implementation of the results reported in RQ3.

Users can set the `pretrain.cond` and `public_data.name` to choose between conditional and unconditional pretraining or to enable or disable pretraining. `public_data.name=null` indicates that pretraining is excluded. If users wish to use Places365 or a pretraining dataset, please take note of the following key parameters.

- `public_data.n_classes`: the number of categories for pretraining dataset (e.g., 365 for Places365).
- `public_data.name`: [`null`, `imagenet`, `places365`].
- `public_data.train_path`: the path to pretraining dataset.

We use ImageNet as the default pretraining dataset, and these parameters are configured accordingly. We provide more implementation examples in the [scripts](./scripts/rq3.sh).

For example, 

(1) Using ImageNet to pretrain DPGAN using conditional pretraining.
```
python run.py setup.n_gpus_per_node=4 --method DPGAN --data_name mnist_28 \
 --epsilon 10.0 eval.mode=val \
 public_data.name=imagenet \
 pretrain.cond=true \
 --exp_description pretrain_imagenet_conditional 
```

(2) Using Places365 to pretrain DPGAN using conditional pretraining.
```
python run.py setup.n_gpus_per_node=4 --method DPGAN --data_name mnist_28 \ 
 --epsilon 10.0 eval.mode=val \
 public_data.name=places365 public_data.n_classes=365 public_data.train_path=dataset/places365 \
 pretrain.cond=true \
 --exp_description pretrain_places365_conditional 
```

(3) Using Places365 to pretrain DPGAN using unconditional pretraining.
```shell
python run.py setup.n_gpus_per_node=4 --method DPGAN --data_name mnist_28 \ 
 --epsilon 10.0 eval.mode=val \
 public_data.name=places365 public_data.n_classes=365 public_data.train_path=dataset/places365 \
 pretrain.cond=False \
 --exp_description pretrain_places365_unconditional 
```

If users wish to use select the public dataset for pretraining like PrivImage in Table 12, you should set the `public_data.selective.ratio`.

For example, use 5% ImageNet to pretrain DPDM.
```
python run.py setup.n_gpus_per_node=4 --method DPDM --data_name cifar10_32 \
 --epsilon 10.0 eval.mode=val \
 public_data.name=imagenet \
 public_data.selective.ratio=0.05 \
 pretrain.cond=true \
 --exp_description pretrain_imagenet_5perc 
```

> [!Note]
>
> If users wish to combine warm-up training in DP-FETA with other methods, you should set the `public_data.name=central_mean`.

#### 4.3.3 Training Using Checkpoints

DPImageBench also supports training synthesizers from the checkpoints. As mentioned in the [results structure](#451-results-structure), we provide `snapshot_checkpoint.pth` to store the synthesizer's parameters at the current epoch after each iteration. If users wish to finetune the synthesizers using pretrained models, they should: (1) set `public_data.name=null`, and (2) load the pretrained synthesizers through `model.ckpt`. For example, the pretrained synthesizer can be sourced from other algorithms. Readers can refer to the [file structure](./exp/README.md) for more details about loading pretrained models like

```
python run.py setup.n_gpus_per_node=3 public_data.name=null eval.mode=val \
 model.ckpt=./exp/pdp-diffusion/<the-name-of-scripts>/pretrain/checkpoints/snapshot_checkpoint.pth \
 --method PDP-Diffusion --data_name fmnist_28 --epsilon 10.0 --exp_description <any-notes>
```

If users wish to continue the pretraining using checkpoints, you just need to set the `public_data.name` as usual like

```
python run.py setup.n_gpus_per_node=3 public_data.name=imagenet eval.mode=val \
 model.ckpt=./exp/pdp-diffusion/<the-name-of-scripts>/pretrain/checkpoints/snapshot_checkpoint.pth \
 --method PDP-Diffusion --data_name fmnist_28 --epsilon 10.0 --exp_description <any-notes>
```

Currently, only diffuisn-based methods are supported, because GAN-based methods usually do not benefit from pretraining and their training is fast.


### 4.4 Results
We can find the `stdout.txt` files in the result folder, which record the training and evaluation processes. The results for utility and fidelity evaluations are available in `stdout.txt`. The result folder name consists of `<data_name>_eps<epsilon><notes>-<starting-time>`, e.g., `mnist_28_eps1.0-2024-10-25-23-09-18`.


#### 4.4.1 Results Structure

We outline the structure of the results files as follows. The training and evaluations results are recorded in the file `exp`. For example, if users leverage the DP-FETA-Pro method to generate synthetic images for the MNIST dataset under a privacy budget of `eps=1.0`, the structure of the folder is as follows:

```plaintext
exp/                                  
├── dp-feta-pro/ 
│   └── mnist_28_eps1.0-2025-07-29-13-35-18/  
│           ├── gen  
│           │   ├── gen.npz 
│           │   └── sample.png 
│           ├── gen_freq  
│           │   ├── gen.npz 
│           │   ├── pc.pth 
│           │   └── sample.png 
│           ├── pretrain  
│           │   ├── checkpoints  
│           │   │   ├── final_checkpoint.pth  
│           │   │   └── snapshot_checkpoint.pth  
│           │   └── samples 
│           │       ├── iter_2000 
│           │       └── ... 
│           ├── pretrain_freq  
│           │   ├── checkpoints  
│           │   │   ├── final_checkpoint.pth  
│           │   │   └── snapshot_checkpoint.pth  
│           │   └── samples 
│           │       ├── iter_2000 
│           │       └── ... 
│           ├── train
│           │   ├── checkooints  
│           │   │   ├── final_checkpoint.pth  
│           │   │   └── snapshot_checkpoint.pth    
│           │   └── samples 
│           │       ├── iter_2000 
│           │       └── ... 
│           ├── train_freq
│           │   ├── checkooints  
│           │   │   ├── final_checkpoint.pth  
│           │   │   └── snapshot_checkpoint.pth    
│           │   └── samples 
│           │       ├── iter_2000 
│           │       └── ... 
│           └──stdout.txt   
├── pe/ 
└── privimage/  
```

We introduce the files as follows,

- `./gen/gen.npz`: the synthetic images.
- `./gen/sample.png`: the samples of synthetic images.
- `./gen_freq/gen.npz`: the synthetic images from the auxiliary generator using frequency features.
- `./gen_freq/sample.png`: the sample of synthetic images from the auxiliary generator.
- `./pretrain/checkpoints/final_checkpoint.pth`: the parameters of synthsizer at the final epochs.
- `./pretrain/checkpoints/snapshot_checkpoint.pth`: we store the synthesizer's parameters at the current epoch after each iteration, deleting the previous parameters to manage storage efficiently.
- `./pretrain/samples/iter_2000`: the synthetic images under 2000 iterations for pretraining on public datasets.
- `./train/checkpoints/final_checkpoint.pth`: the parameters of synthsizer at the final epochs.
- `./train/checkpoints/snapshot_checkpoint.pth`: we store the synthesizer's parameters at the current epoch after each iteration, deleting the previous parameters to manage storage efficiently.
- `./train/samples/iter_2000`: the synthetic images under 2000 iterations for training on sensitive datasets.
- `./stdout.txt`: the file used to record the training and evaluation results.

#### 4.4.2 Results Explanation

In utility evaluation, after each classifier training, we can find,

```
INFO - evaluator.py - 2024-11-12 05:54:26,463 - The best acc of synthetic images on sensitive val and the corresponding acc on test dataset from wrn is 64.75999999999999 and 63.99
INFO - evaluator.py - 2024-11-12 05:54:26,463 - The best acc of synthetic images on noisy sensitive val and the corresponding acc on test dataset from wrn is 64.75999999999999 and 63.87
INFO - evaluator.py - 2024-11-12 05:54:26,463 - The best acc test dataset from wrn is 64.12
```
These results represent the best accuracy achieved by: (1) using the sensitive validation set (63.99%), (2) adding noise to the validation results of the sensitive dataset (`model.eval = val`), and the accuracy is 63.87%, and (3) using the sensitive test set for classifier selection (64.12%). 

If synthetic images are used as the validation set (`model.eval = syn`), the results after each classifier training would be:
```
INFO - evaluator.py - 2024-10-24 06:45:11,042 - The best acc of synthetic images on val (synthetic images) and the corresponding acc on test dataset from wrn is 63.175 and 56.22
INFO - evaluator.py - 2024-10-24 06:45:11,042 - The best acc test dataset from wrn is 64.22
```
These results present that the best accuracy achieved by: (1) using the synthetic images for validation set (56.22%) and (2) using the sensitive test set for classifier selection (64.22%).

The following results can be found at the end of the log file:
``` 
INFO - evaluator.py - 2024-11-13 21:19:44,813 - The best acc of accuracy (adding noise to the results on the sensitive set of validation set) of synthetic images from resnet, wrn, and resnext are [61.6, 64.36, 59.31999999999999].
INFO - evaluator.py - 2024-11-13 21:19:44,813 - The average and std of accuracy of synthetic images are 61.76 and 2.06
INFO - evaluator.py - 2024-11-13 21:50:27,195 - The FID of synthetic images is 21.644407353392182
INFO - evaluator.py - 2024-11-13 21:50:27,200 - The Inception Score of synthetic images is 7.621163845062256
INFO - evaluator.py - 2024-11-13 21:50:27,200 - The Precision and Recall of synthetic images is 0.5463906526565552 and 0.555840015411377
INFO - evaluator.py - 2024-11-13 21:50:27,200 - The FLD of synthetic images is 7.258963584899902
```
The first line shows the accuracy of the downstream task when noise is added to the validation results of the sensitive dataset for classifier selection (`model.eval = val`), across three studied classification outcomes. 

If synthetic images are used as the validation set (`model.eval = syn`), the first line would be:
```
INFO - evaluator.py - 2024-11-12 09:06:18,148 - The best acc of accuracy (using synthetic images as the validation set) of synthetic images from resnet, wrn, and resnext are [59.48, 63.99, 59.53].
```
The synthetic images can be found at the `./exp/<algorithm_name>/<file_name>/gen/gen.npz`.

### 4.5 Results Visualization

We provide the plotting codes for results visualization in the folder `plot`.

- `ablation.py`: plotting for Figure 5 in RQ2.
- `fid_curve.py`: plotting for Figure 4 in RQ1.
- `plot_eps_changes.py`: plotting for Figure 7 in RQ3.
- `plot_heatmap.py`: plotting for Figure 6 in RQ3.
- `visualization_tf.py`: plotting for Figure 2.   
- `visualization.py`: plotting for Figure 3 in RQ1. 


## Acknowledgement
 
Part of code is borrowed from [DPImageBench](https://github.com/2019ChenGong/DPImageBench). We sincerely thank them for their contributions to the community.
