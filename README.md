# Moment Localization with two different approaches using Residual Connection 

As various types of unstructured data, such as life logging, increase with the development of networks and smart devices, research on multimodal learning through vision and language is drawing more attention. In particular, we noted the moment localization that determines the temporal moment corresponding to the natural language query, where many studies are being conducted. Accordingly, two improvement measures are proposed by analyzing the (2D-TAN) model that proposed the 2D thermal map. Based on the application of the residual block and the concept of DenseNet, we experimented with a model that combines the hidden layers of all previous blocks as input, and for the Charades-STA dataset, performance improvement of up to 5 points compared to the performance of the previous model was confirmed.



## Method
![image](https://user-images.githubusercontent.com/57340671/147183105-d1483789-2567-49ce-8f74-ea3825e6e3fd.png)

### 1) MoL + R : Moment localization with Residual block

We constructed a Temporal adjective network by reflecting the Residual block. Additionally adding only residual information to the learned function before transferring the block-by-block parameters. Therefore, learning became easier than learning the whole. In addition, in the case of the previous method, since all weight layers are separated when learning the whole, the difficulty of convergence increased by learning for each layer, and convergence became easier by using residual blocks. By directly inserting information about the previous input x each block, the information on the original video and the natural language query can be consistently maintained.


![image](https://user-images.githubusercontent.com/57340671/147183175-fa99a671-d796-458f-8f60-a58a1067e134.png)

### 2) MoL + D : Moment localization with Dense layer
We applied the Dense layer by advancing one step further from the network of Moment localization with residual block. In the Dense layer, like Residual blocks, one block consists of two convolution networks. Unlike the Residual block, which adds the outputs of the previous block, F(x) and x, it adds the input of all passed blocks. It represented as 


$H(x)=F(x_{n-1})+\sum_{i=1}^{n-1}x_{i}$


A temporal adjective network is configured by replacing the convolution layer of the previous model with 4 Dense layers. Dense Layer concatenates the feature map of the previous layer to the feature map of all layers that appear thereafter. Through this configuration, the effect of regularization can also be seen because it prevents loss of information, such as alleviating the vanishing gradient problem, and learns by connecting feature maps of various layers.



## Main Results

#### Main results on Charades-STA
| Method | Rank1@0.5 | Rank1@0.7 | Rank5@0.5 | Rank5@0.7 |
| ---- |:-------------:| :-----:|:-----:|:-----:|
| 2D-TAN | 39.70 | 23.31 | 80.32 | 51.26 |
| MoL + R | 41.02 | 23.33 | 84.33 | 50.03 |
| MoL + D | 42.04 | 24.46 | 85.94 | 52.18 |

#### Main results on ActivityNet Captions 
| Method | Rank1@0.3 | Rank1@0.5 | Rank1@0.7 | Rank5@0.3 | Rank5@0.5 | Rank5@0.7 |
| ---- |:-------------:| :-----:|:-----:|:-----:|:-----:|:-----:|
| 2D-TAN | 59.45 | 44.51 | 26.54 | 85.53 | 77.13 | 61.96 |
| MoL + R | 60.71 | 44.08 | 26.05 | 85.20 | 76.50 | 60.61 |
| MoL + D | 60.63 | 44.33 | 26.43 | 85.23 | 76.40 | 60.83 |

#### Main results on TACoS
| Method | Rank1@0.1 | Rank1@0.3 | Rank1@0.5 | Rank5@0.1 | Rank5@0.3 | Rank5@0.5 |
| ---- |:-------------:| :-----:|:-----:|:-----:|:-----:|:-----:|
| 2D-TAN | 47.59 | 37.29 | 25.32 | 70.31 | 57.81 | 45.04 |
| MoL + R | 47.86 | 38.09 | 26.27 | 72.48 | 60.93 | 47.54 |
| MoL + D | 48.09 | 36.49 | 25.12 | 73.11 | 57.79 | 45.51 |



## Prerequisites
- pytorch 1.1.0
- python 3.7
- torchtext
- easydict
- terminaltables




#### Training
Use the following commands for training:
```
# Evaluate "Pool" in Table 1
python moment_localization/train.py --cfg experiments/charades/2D-TAN-16x16-K5L8-pool.yaml --verbose

# Evaluate "Pool" in Table 2
python moment_localization/train.py --cfg experiments/activitynet/2D-TAN-64x64-K9L4-pool.yaml --verbose

# Evaluate "Pool" in Table 3
python moment_localization/train.py --cfg experiments/tacos/2D-TAN-128x128-K5L8-pool.yaml --verbose

```

#### Testing
Our trained model are provided in [box drive](https://rochester.box.com/s/5cfp7a5snvl9uky30bu7mn1cb381w91v). Please download them to the `checkpoints` folder.

Then, run the following commands for evaluation: 
```
# Evaluate "Pool" in Table 1
python moment_localization/test.py --cfg experiments/charades/2D-TAN-16x16-K5L8-pool.yaml --verbose --split test

# Evaluate "Pool" in Table 2
python moment_localization/test.py --cfg experiments/activitynet/2D-TAN-64x64-K9L4-pool.yaml --verbose --split test

# Evaluate "Pool" in Table 3
python moment_localization/test.py --cfg experiments/tacos/2D-TAN-128x128-K5L8-pool.yaml --verbose --split test

```

## Citation
```
@InProceedings{2DTAN_2020_AAAI,
author = {Zhang, Songyang and Peng, Houwen and Fu, Jianlong and Luo, Jiebo},
title = {Learning 2D Temporal Adjacent Networks forMoment Localization with Natural Language},
booktitle = {AAAI},
year = {2020}
} 
```
