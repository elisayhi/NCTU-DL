Lab1 Back-Propogation
=====================
## Introduction
Backpropagation is a method to calculate the gradient that is used to update weights and biases in the neural network. Backpropagation applies the chain rule iteratively to calculate gradient of weights and biases of each layer.<br>

Backpropagation is a special case of a more general technique called automatic differentiation, and is commonly used by the gradient descent optimization algorithm to adjust to weights of neurons. The parameters of the network have the relationship with the error the net produces, and when the parameters change, the error does, too.<br>

There are 2 phases in backpropagation, that is, forward phase and backward phase.<br>

In the forward phase, the input will be passed from input layer to output layer, and will do a linear transformation (that is, WTX+b) and pass an activation function (sigmoid function is used in this lab) in each layer of neuron network. After the input pass through the network, an outcome will be produced.<br>

In the backward phase, all the hyperparameters in the network are updated by the algorithm called gradient descent, which is useful for finding a minimum of a function.<br>

## Experiment Setup
### Sigmoid function
The definition of sigmoid function: S(x) = 1/(1+exp(-x))<br><br>
Sigmoid function is a monotonic function, and also a special case of the logistic function. It can return all real numbers to the range from 0 to 1. Besides, it is differentiable, having a non-negative or non-positive first derivative, one local minimum and one local maximum, and a simple form which is computationally easy to perform, that is S(x)(1-S(x)).<br><br>
Sigmoid functions are often used in neural networks to introduce nonlinearity in the model<br><br>
The weakness of sigmoid function is that, for extremely big or small value, it do not differ in the degree.<br><br>
### Neural network
The neural network includes forward phase and backward phase, inputs are passed to the output layer in the forward phase, and hyperparamters are updated in the backward phase. A neural network shows below.<br>
<img src="https://i.imgur.com/3hq6bJJ.png" width="500"/>
Inputs of each hidden layer are linear combination of outputs of previous layer, and the outputs of each hidden layer are sigmoid of input of that neuron.<br>
Parameters in the derivation:
- z: linear combination of outputs of previous layer
- a: value of z after passing activation function
- w<sub>ij</sub><sup>k</sup>: weights of neurons
- i: to which neuron
- j: from which neuron
- k: at which layer
- L: loss function
- delta: partial derivative of L with respect to w

Parameters in the program:
- out: same as a parameters in the derivation
- net: same as z parameters in the derivation
- other parameters are same as parameters in the derivation
The loss function is MSE in this program: 2(y - ŷ)2
The activation function is sigmoid in this program

## Backpropagation
Derive general matrix form<br>
<img src="https://i.imgur.com/XW2RKUl.png" width="500"/>
<img src="https://i.imgur.com/kXP4nCt.png" width="500"/>
<img src="https://i.imgur.com/ItBGJ3f.png" width="500"/>
<img src="https://i.imgur.com/jOcT9U1.png" width="500"/>
<img src="https://i.imgur.com/8iDIiAQ.png" width="500"/>
The derivative sigmoid function is not same as the function TA provide. My derivative sigmoid function is dsig(x)=sig(x)(1-sig(x)).<br>
