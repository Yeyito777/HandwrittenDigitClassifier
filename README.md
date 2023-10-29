# DigitClassifier
A multi-layered neural network for classifying MNIST handwritten 28x28 digits.

The architecture has been coded from the ground-up, I make no use of externally available optimizers or libraries.
The goal of this project was to learn how neural networks work at the most basic level.

This architecture uses a Sigmoid activation function, easily-alterable layers both in depth and width and input normalization.
For optimizing we use the Mean Squared Error and Stochastic Gradient Descent with a momentum term.

It is able to achieve a 97.5% on the MNIST handwritten digits test set with a network of dimension: [784,64,64,64,64,10] (!),
Increasing the network width or depth further does not yield any substantial improvement, suggesting the saturation of a Feed-Forward,
Sigmoid neural network on the raw, unedited, 60K training images given by MNIST.

The 97.5% score was attained using the following args:
  1. NeuronArrangement: [784,64,64,64,64,10]
  2. LearningRateDecayPerEpoch: 1.15
  3. BatchSize: 20
  4. Epochs: 300
  5. Dampening: 0.1
  6. BatchSizeIncreasePerEpoch: 1 (No increase)

 Despite a large number of epochs the network doesn't overfit to the data because of it's reduced size.

 Feel free to do whatever you wish with this code!
 I do not recommend using this code for anything other than learning for yourself.

 P.S. The main function in the C++ code starts with code on how to re-test a network.

 # Usage
  ./a.out
No arguments makes it start training a new Neural Network from scratch with the specifications in the code, it will be saved to trainedNetworks

  ./a.out [NETWORK_PATH]
Lets you re-test a network of your choice. Does not save the network again.

# Credits
This work was based off of the book "An Introduction to Neural Networks" by Patrick van der Smagt & B. J.A. Krose
Feel free to contact me through discord: yeyito777
