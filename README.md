# Improving the Convergence Speed of Deep Networks with Biased Sampling

## Abstract

Deep Neural Networks (DNNs) are usually trained using a stochastic gradient descent algorithm or its variants. 
This involves the gradient being computed on small batches of samples. The decision of which samples will be selected is an important step in the training process, as it can greatly affect the DNN's performance.
In this paper, we propose a novel method of sample selection based on replacing easy examples with difficult ones every epoch. 
This leads to an acceleration of the training process of deep networks. The resulting algorithm was tested on four datasets, representing both regression (Boston Housing) and image classification (MNIST, CIFAR10, CIFAR100) problems. We show that our approach, called Batch Selection with Biased Sampling (\(BSBS\)), yields faster convergence rates. Experiments indicate that the model can be trained in up to 50\% less epochs.
