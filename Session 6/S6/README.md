# Session 6
<br>

# PART - 1 

# BACKPROPAGATION

Backpropagation is a widely used algorithm for training artificial neural networks, particularly in the context of deep learning. It enables neural networks to learn from labeled training data and make predictions or classify new examples.

The goal of backpropagation is to adjust the weights and biases of a neural network's connections to minimize the difference between its predicted output and the desired output. This process involves two key steps: forward propagation and backward propagation.

The process of forward propagation followed by backward propagation is iteratively repeated for multiple training examples until the network converges to a state where the loss is minimized and the predictions are accurate. Backpropagation allows neural networks to automatically learn the appropriate weights and biases through a supervised learning process.

<br>

# Neural Network

![SimpleNeuralNetwork.png](../../Results/Session%206/NeuralNetwork.png)

<br>

# Forward pass

![ForwardPass.png](../../Results/Session%206/ForwardPass.png)

<br>

# Calculating Gradients w.r.t w5

![w5.png](../../Results/Session%206/w5.png)

<br>

# Calculating Gradients in layer 2

![l2.png](../../Results/Session%206/l2.png)

<br>

# Calculating gradients intermediate step for layer 1

![gl1.png](../../Results/Session%206/gl1.png)

<br>

# Calculating Gradients in layer 1

![l10.png](../../Results/Session%206/l10.png)

<br>

# Calculating Gradients in layer 1

![l11.png](../../Results/Session%206/l11.png)

<br>

# Neural Network Architecture

![BackPropagation.png](../../Results/Session%206/Backpropagation.png)

<br>

# Major Steps

Below are the defined major steps in this exercise :     
1. **Initialization** - Weights of the neural network are initialized. (Inputs, Targets, Initial set of weights and Hidden Layer weights)
<br>

2. **Utility functions** - Sigmoid Activation function  : maps the input to a value between 0 and 1.
<br>

3. **Forward propagation** - Given the weights and inputs, this function calculates the predicted output of the network.
<br>

4. **Error Calculation** - Calculate ```0.5 * Squared Error``` between predicted output and target values.
<br>

5. **Gradient functions for each weights of the network** - These functions calculate the gradients of Error with respect to each weights in the network. This determines the direction and size of step we could take in the direction of minima. Two gradient functions are defined one for each layer. ```gradient_layer1``` function updates the weights that connect the input layer to the hidden layer. ```gradient_layer2``` function updates the weights that connect the hidden layer to output layer.
<br>

6. **Updation of weights** - We have incorporated updation of weights for each iteration in a "for loop". Each weight is updated by taking only a fraction of step size. The fraction here is defined using learning rate. Higher the learning rate greater the step we take. As a common practice learning rates are in the range between 0 to 1.
<br>

7. All the above steps are run for different learning rates in a for loop.   
<br>

# Variation of Losses w.r.t Learning Rates

![Losses.png](../../Results/Session%206/Losses.png)

<br>

# Error graphs

<div style="margin-bottom: 15px;">
    <img src="../../Results/Session%206/lr0.1.png" alt="LR 0.1" width="350" style="margin-right: 15px;"/>
    <img src="../../Results/Session%206/lr0.2.png" alt="LR 0.2" width="350"/>
</div>

<div style="margin-bottom: 15px;">
    <img src="../../Results/Session%206/lr0.5.png" alt="LR 0.5" width="350"style="margin-right: 15px;"/>
    <img src="../../Results/Session%206/lr0.8.png" alt="LR 0.8" width="350"/>
</div>

<div style="margin-bottom: 15px;">
    <img src="../../Results/Session%206/lr1.0.png" alt="LR 1.0" width="350" style="margin-right: 15px;"/>
    <img src="../../Results/Session%206/lr2.0.png" alt="LR 2.0" width="350"/>
</div>

<br>

## Note:
- With higher learning rate, we are reaching global minima for the weights faster. 

<br>