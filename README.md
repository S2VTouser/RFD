# RFD

# Enhancing Data-Free Class-Incremental Learning via Image-Centric Dual Distillation
This is the *Pytorch Implementation* for the paper Enhancing Data-Free Class-Incremental Learning via Image-Centric Dual Distillation.

## Framework


## Abstract
In Data-free Class-Incremental Learning (DFCIL), catastrophic forgetting is a significant challenge due to the lack of access to previous task image data. Recent approaches using model inversion have made progress in addressing this issue, yet the suboptimal application of knowledge distillation hampers new task learning, limiting overall model performance. To overcome this, we propose a novel method incorporating image-centric dual distillation, designed to retain more old knowledge while facilitating new knowledge acquisition, thus enhancing DFCIL performance. Specifically, we first introduce a weak-constraint relation distillation strategy to preserve old knowledge while promoting the assimilation of new knowledge by learning the relationships among intra-class samples. Then, to further enhance the preservation of old knowledge and refine the integration of new knowledge, we introduce a low-level feature distillation strategy to retain foundational general knowledge by leveraging semantic information from shallow network layers. Extensive experiments show the effectiveness of our method.

### Contributions
* We propose a novel method incorporating image-centric dual distillation strategies to retain more old knowledge while facilitating new knowledge acquisition thereby enhancing the model performance under the DFCIL setting.
* We introduce a weak-constraint relation distillation strategy to retain more useful old knowledge while promoting the assimilation of new knowledge by learning the correlated relationships among intra-class image samples. Furthermore, to enhance the preservation of old knowledge and refine the integration of new knowledge, we introduce a low-level feature distillation strategy to dig deep and retain fundamental general knowledge by leveraging semantic information from shallow network layers.
* Extensive experiments on multiple benchmarks demonstrate that our method significantly outperforms the state-of-the-arts under the DFCIL setting.

## Setup
           

## Datasets
* Sequential CIFAR-10 
* Sequential CIFAR-100 
* Sequential Tiny ImageNet
* Sequential ImageNet-100
* Sequential ImageNet200-R

## Citation
If you found the provided code useful, please cite our work.
