# Neural-Network-Implementation-from-scratch-Python
This repository contains two implementations for a feedforward neural network capable to clasify the three iris flowers classes from the iris data set. For a better understanding of the neural network, writing the algorithms is a big help.

The data set used for training and testing is the iris data set, both of them are capable to clasify the three classes of iris flowers. These implementations are from scratch coded in Python using only numpy for mathematical operations.

The Arhitecture of the neural network consits in two layers, the first layer has 5 perceptrons and this layer receive the input, and the second layer is the output layer and has 3 perceptrons, the perceptron with the max output value set the class, each perceptron from the output layer is capable to tell if a flower is in it's class or not. Both layers uses the LogSig activation function.
Both of them use:
- Backpropagation algorithm for the training of the neural network
- Gradient Descent for the update of weights and bias.
- The mean squared error (MSE)
- Criterion function is sum of squared error (SSE)

The main difference is the optimization strategy for Optimizing the Gradient Descent, the first one uses the Momentum Strategy, and the second one uses Adam Strategy.
