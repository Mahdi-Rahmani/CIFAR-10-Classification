# Neural Networks : Classification task
One of the applications of neural networks is classification. In this project, we aim to venture into image classification, specifically hand gesture recognition. We intend to build a model using various neural networks, particularly fully connected networks, which take images as input and classify each image as a specific hand gesture. After implementing the fully connected neural network, we can also optionally explore more advanced architectures, such as CNNs, and learn about their enhancements. 
## Dataset
In this project, we want to categorize [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset images. This dataset is a smaller version of the CIFAR 100 dataset and is limited to images from 10 different categories. The images of this dataset can be seen in the figure below. The dimensions of the images in this dataset are equal to 32 (length) by 32 (width) by 3 (RGB color channels).
<p align="center">
  <a href="https://github.com/Mahdi-Rahmani/CIFAR-10-Classification/blob/main/pics/cifar10.png">
    <img src="https://github.com/Mahdi-Rahmani/CIFAR-10-Classification/blob/main/pics/cifar10.png" alt="cifar Image" width="400" height="300">
  </a>
</p>

## Learning Process
The learning process of our neural network according to the Stochastic Gradient Descent method is as follows. The idea of this method is that instead of coming and training with the entire set of data at each stage of model learning, we can divide the data into mini-batch sections in each scan, the gradient corresponding to each sample of that mini-batch get the batch, and finally get their average and then apply the changes. This will reduce the number of calculations per scan and increase the speed of convergence.
<p align="center">
  <a href="https://github.com/Mahdi-Rahmani/CIFAR-10-Classification/blob/main/pics/psudocode.png">
    <img src="https://github.com/Mahdi-Rahmani/CIFAR-10-Classification/blob/main/pics/psudocode.png" alt="psudo code Image" width="400" height="300">
  </a>
</p>

## Implementing learning process
In First part of this project, a simple feed forward fully connected nueral network is implemented to classify first four classes of this dataset.  
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images. 
This Prroject has 5 major parts:  
 1. Loading dataset
 2. Feedforward implementation
    - we can use below formula:
    $$ a^{(L+1)} = \sigma{(W^{(L+1)}\times{a^{(L)}} + b^{(L+1)}) }$$  

 3. Backpropagation (non-vectorized and with loops) 
    - In non-vectorixed backpropagation we should calculate grad matrixes with for loops. In MLP class, with backpropagation function we do this job. We used SSE cost-function. the formula is:  
    $$ Cost = \sum_{j = 0}^{n_L -1 }{{(a_j^{(L)} - y_j)}^2 }$$  
    - We want to minimum the cost. We can use Gradient Descent for this. we should calculate partial derivatives of the Cost function with respect to other parameters. for calculating partial derivatives we can use chain rule. according to chane rule we have:  
    $$\displaystyle \frac{\partial Cost}{\partial w_{jk}^{(L)}} = \displaystyle \frac{\partial Cost}{\partial a_{j}^{(L)}} \times \displaystyle \frac{\partial a_{j}^{(L)}}{\partial z_{j}^{(L)}} \times \displaystyle \frac{\partial z_{j}^{(L)}}{\partial w_{jk}^{(L)}} $$  
    $$\displaystyle \frac{\partial Cost}{\partial b_{j}^{(L)}} = \displaystyle \frac{\partial Cost}{\partial a_{j}^{(L)}} \times \displaystyle \frac{\partial a_{j}^{(L)}}{\partial z_{j}^{(L)}} \times \displaystyle \frac{\partial z_{j}^{(L)}}{\partial b_{j}^{(L)}} $$  
    If we simpilify we can have the below formulas. Also we need derivative according to Activations and they calculated like others.

    ### (Last layer):
    #### Derivative according to Weights:
    $$\displaystyle \frac{\partial Cost}{\partial w_{jk}^{(L)}} = 2(a_{j}^{(L)} - y_{j}) \times \sigma^{'}(z_{j}^{(L)})\times a_{k}^{(L-1)}$$
    #### Derivative according to Biases:
    $$\displaystyle \frac{\partial Cost}{\partial b_{j}^{(L)}} = 2(a_{j}^{(L)} - y_{j}) \times \sigma^{'}(z_{j}^{(L)})\times 1$$
    #### Derivative according to Activations:
    $$\displaystyle \frac{\partial Cost}{\partial a_{k}^{(L-1)}} = \sum_{j=0}^{n_{(L-1)}} \displaystyle \frac{\partial Cost}{\partial a_{j}^{(L)}} \times \displaystyle \frac{\partial a_{j}^{(L)}}{\partial z_{j}^{(L)}} \times \displaystyle \frac{\partial z_{j}^{(L)}}{\partial a_{k}^{(L-1)}} $$  
    ### (Other layers):
    #### Derivative according to Weights:  
    $$\displaystyle \frac{\partial Cost}{\partial w_{km}^{(L-1)}} =  \displaystyle \frac{\partial Cost}{\partial a_{k}^{(L-1)}} \times \sigma^{'}(z_{k}^{(L-1)})\times a_{m}^{(L-2)}$$
    #### Derivative according to Biases:  
    $$\displaystyle \frac{\partial Cost}{\partial b_{k}^{(L-1)}} = \displaystyle \frac{\partial Cost}{\partial a_{k}^{(L-1)}} \times \sigma^{'}(z_{k}^{(L-1)})\times 1$$
    #### Derivative according to Activations:  
    $$\displaystyle \frac{\partial Cost}{\partial a_{m}^{(L-2)}} = \sum_{k=0}^{n_{(L-2)}} (\displaystyle \frac{\partial Cost}{\partial a_{k}^{(L-1)}} \times \sigma^{'}(z_{k}^{(L-1)})\times w_{km}^{(L-1)}) $$  
    The above formula is implemented in **backpropagation function in MLP class**.
 4. Backpropagation (vectorized form)
    - Code of this section is implemented in **vectorized_backpropagation in MLP class**. we dont use for loop for calculates derivatives with respect to each element of vector or matrix. we can use matrix multiplications. also for doing this we get help from project explanation and codes like this:  
    grad_w3 += (2 * sigmoid_deriv(z3) * (a3 - y)) @ (np.transpose(a2))
 5. Test final model  
  
For training the network Stochastic Gradient Descent was used, implemented in tow forms of iterative backpropagation and vectorized backpropagation. 
The model loss after implementing backpropagation phase in vectorized and non-vectorized forms, like below:
vectorized    |  non-vectorized  |
:-------------------------:|:-------------------------:|
![](https://github.com/Mahdi-Rahmani/CIFAR-10-Classification/blob/main/pics/model_loss_vec.png)  |  ![](https://github.com/Mahdi-Rahmani/CIFAR-10-Classification/blob/main/pics/model_loss_nonvec.png)  |

Code of this section is available at this [link](https://github.com/Mahdi-Rahmani/CIFAR-10-Classification/blob/main/Project1.ipynb).

## Additional part
In the main part, we saw how to implement a fully connected neural network, but with the existence of 10 different classes, it was not acceptable to classify the performance of this network.
Convolutional neural networks work exactly like the image processing system in living organisms. Contrary to the old history, these types of networks were not used until the last few years due to the weakness of the hardware's ability to implement them. But in ImageNet Large Scale Recognition Challenge This type of network attracted everyone's attention and was able to classify 1000 different classes with an error of 16%, while the best method reached an error of 26%.
Now we want to use a convolutional neural network on the CIFAR-10 dataset and compare the results with the networks you implemented in the main part of the project.

We implemented the following architecture. For the convolutional layers, we set the kernel size to 3 and set the padding to the same mode.
<p align="center">
  <a href="https://github.com/Mahdi-Rahmani/CIFAR-10-Classification/blob/main/pics/architectur.png">
    <img src="https://github.com/Mahdi-Rahmani/CIFAR-10-Classification/blob/main/pics/architectur.png" alt="CNN Image" width="400" height="300">
  </a>
</p>

Code of this section is available at this [link](https://github.com/Mahdi-Rahmani/CIFAR-10-Classification/blob/main/Project1.ipynb) too.
