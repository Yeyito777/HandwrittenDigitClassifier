# HandwrittenDigitClassifier
A multi-layered neural network for classifying MNIST handwritten 28x28 digits.

This branch is an implementation of the C++ code in the main branch into Pytorch.
There are some functional changes that were made leveraging the ease-of-use of the Pytorch framework:
  1. ReLU activation functions for all layers except the last two layers
  2. SiLU activation function for the last layer before the output layer
  3. Output layer features no activation function as it is implicit in the criterion
  4. Cross Entropy Loss criterion instead of Mean Squared Error
  5. AdamW optimizer instead of Stochastic Gradient Descent

Saved networks can be found inside the "Networks" folder and can be loaded automatically using ```net = Net(); net.load_state_dict(torch.load(model_path))``` in Pytorch.
**I didn't save any of the Net()'s config for the trained networks so don't bother loading them. Their sole purpose is showcasing test scores.**
If you wish to train your own networks and load them later, **make sure to save their Net() class!**.

The 98.16% Was achieved using the following architecture: [784,1024,512,256,128,64,10]. In my tests I was also able to achieve a 98.35% using the same architecture and training for 10 to 15 epochs.
The variance when training a network can be quite high as the algorithm converges to different local minima. I've noticed you may get up to a 1.5% increase or decrease in training runs with the same configuration.

# Setup
First, create a local environment, I've decided to use Python's built in virtual environment but you can use other tools such as conda.
```python -m venv Environment```

Next, install all the necessary dependencies.
```pip install torch torchvision torchaudio```
You may also pick your installation from Pytorch's official website: https://pytorch.org/get-started/locally/
In this case I went with the CPU only installation.

You're all set!

# Credits
Feel free to use this code in any way shape or form.
This was developed entirely by Yeyito777.
  discord: yeyito777
  twitter: @Yeyito777x
