# Annotation
The program for training and validation a neural network. The program can be used as a library and set the parameters manually, or you can do it through the console. The program also uses several libraries, most of which are written manually, and the rest is installed separately.
The goal of the project is to provide the ability to quickly implement deep learning algorithms for image recognition and solution validation.

# Technical task
The user can set 4 training parameters:

* Number of epochs
* Batch size
* Architecture
* Learning rate
* Optimizer

The number of training epochs is a positive integer value, the batch size is also a positive integer value. The learning rate is a floating point number that limits the learning rate of the model. Two optimizers are currently available: "momentum" and "sgd". The architecture is given by a sequence of positive integer values ​​separated by a dash, for example, **784-128-10** will represent a neural network with the number of input neurons 784(=28*28), then a hidden layer of 128 neurons and at the end the number of neurons is equal classes, which is 10 in our case, because we classify numbers from 0 to 9:

![](https://raw.githubusercontent.com/aamini/introtodeeplearning/master/lab2/img/mnist_2layers_arch.png)

# Algorithm
A simple neural network without convolutional layers was chosen as the recognition algorithm. Other machine learning algorithms, for example, decision trees, also do a good job of solving the problem of recognizing the MNIST dataset, because it is a fairly simple dataset, but the choice of a neural network seemed to me the most interesting, because it is more complicated, and I see more possibilites for optimization here for project extensions. The principle of a neural network is quite simple: the image is converted into a vector, and then this vector is multiplied with each weight vector in each neuron, as we would do in linear regression. After we have received a vector of new values, the number of values ​​is equal to the number of hidden neurons. After this, the activation function is applied, in my project I used the hyperbolic tangent. Then we do all the previous steps with each new hidden layer until we get to the last one. In the last layer, instead of the activation function, we use a function that transforms the linear part of the last layer into a probability distribution. Then the loss function is calculated, in my case it is cross-entropy, and then for each model parameter its derivative with respect to the loss function is calculated to update the model parameter (backpropagation).

Of course, there are a number of other algorithms for training a neural network, but I chose this one because it is the most popular and fairly reliable.

# Program structure
The program is divided into several logical parts:
* Loading data from [source](http://yann.lecun.com/exdb/mnist/) and wrapping it into training and test datasets, dividing datasets into batches
* Image processing, that is, converting them to two-dimensional arrays and normalizing
* Math operations
* Model, its training and testing
* Model parameters
* Optimizers

At the beginning, the data is loaded from [source](http://yann.lecun.com/exdb/mnist/) and placed in the current working directory. Then each image is processed and put into a dataset and batch, which takes quite a long time, because this procedure is carried out for each of the 70,000 images. The process usually takes 1.5-2 minutes. In the class for the dataset, the "Batches" method is represented as an iterator, at least we could make an array with batches. This is done because usually we don't want to have access to an arbitrary batch, but we iterate through the batches sequentially.

The code base for image processing was taken from this project https://github.com/guylangston/MNIST.IO, but not all parts of the code were suitable for my project, some parts of the code were completely rewritten, for example, data structures were changed, image normalization was added , that is, bringing the values ​​of the pixels of the image to the range from zero to one, some parts of the code were completely removed.

To perform mathematical operations, the vector, matrix, and tensor classes have been implemented. A class was also added to generate a matrix, vectors and tensors. All operations are performed according to the rules of linear algebra.

The model is implemented very simply: the model receives the architecture and, according to it, creates its own configuration. The model has methods of prediction, training, testing on test data. To facilitate learning and prediction, a separate class was implemented for fully connected layers.

A separate file for model parameters was created first of all in order to work with the model and the program as a whole through the command line.

Libraries are closely related to each other and interact with each other, for example, Models.cs uses Optimization file and Datasets file, Program.cs uses Models.cs and so on. The model uses fully connected layers, which can be found in Layers.cs. The optimization of the weights in these layers can be found in the Optimization.cs file. All mathematics is described in the Math.cs file: there are three main ways of storing structured information in mathematics: a vector, a matrix, and a three-dimensional tensor. In each of these structures, mathematical operations are carried out differently, therefore, they require separate implementation. ModelOptions.cs contains a set of options for working through the console. Program.cs performs several functions: parses parameters from the console if necessary, initializes the dataset, initializes the project, and tests the quality on each of the 10 classes.

# Representation of input data and their preparation
The fact is that the input to the neural network is possible for the vector 784, that is, 28 * 28. The current processing of the data corresponds to how the data from [source](http://yann.lecun.com/exdb/mnist/) was processed:
>The original black and white (bilevel) images from MNIST were size normalized to fit in a 20x20 pixel box while preserving their aspect ratio. The resulting images contain grey levels as a result of the anti-aliasing technique used by the normalization algorithm. the images were centered in a 28x28 image by computing the center of mass of the pixels, and translating the image so as to position this point at the center of the 28x28 field.

The project was written specifically in such a way that it could be worked with separately through the command line. To enter the command, you need to go from the main project folder to "bin/Debug/net7.0". The command below executes the program and changes the default parameters:
```
./mlpsharp --batch_size=64 --epochs=20 --architecture=784-50-50-50-20-10 --learning_rate=0.01 --optimizer=sgd
```
The results are displayed to the user in the console.
I would like to draw your attention to the fact that in the indication of the architecture, the first and last numbers (784 and 10) must always be the same, only the numbers between them change!!!

The input data to the neural network has already been described in the previous parts. To work in console mode, the parameters are resistant to incorrect format (the format itself is described above) and in the case of incorrect input, the default options specified in the ModelOptions.cs file will be used, but not for architecture parameter, the format is that we indicate a sequence of natural numbers through a dash, with the first and last number being fixed.

# Representation of output data and their interpretation
When the model has made predictions, it does not output a discrete number that some class is correct, it only outputs a probability distribution. Next, we take the number with the highest probability. The user can see in his console the number of correct predictions and their ratio to all predictions. It would also be possible to fix the resistance of the neural network architecture parameter to incorrect input.
architecture parameter to incorrect input.
Example:
```
>>> ./mlpsharp --batch_size=64 --epochs=5 --architecture=784-50-20-10 --learning_rate=0.01 --optimizer=sgd 
Train dataset preparation...
Test dataset preparation...
Epoch: 1, Accuracy: 0,7522, Corrects: 7522
Epoch: 2, Accuracy: 0,8492, Corrects: 8492
Epoch: 3, Accuracy: 0,8795, Corrects: 8795
Epoch: 4, Accuracy: 0,8938, Corrects: 8938
Epoch: 5, Accuracy: 0,9015, Corrects: 9015
Number 0: 96%
Number 1: 97%
Number 2: 88%
Number 3: 89%
Number 4: 91%
Number 5: 80%
Number 6: 93%
Number 7: 91%
Number 8: 82%
Number 9: 87%
```

# Tests
At the end of the training, you can look at the accuracy of the neural network on individual classes. The result is output to the terminal. The test data is in the "bin/Debug/net7.0/MNIST-JPG-testing" folder relative to the project. Data in "jpg" format. Program.cs contains a function that takes the path to an image and converts it to a 2D float array. Then a separate prediction is made on each image and an answer is given. Further, all correct answers are summed up and the accuracy is calculated, which is displayed on the screen.

# Improvements
I used SGD and Momentum to train the network parameters, although this optimizer is more common in academic environments than in serious projects. Instead of SGD and Momentum one could use RMS-Prop, Adam or its modifications. Another way to increase the accuracy of the prediction would be to add convolutional layers or make an ensemble of networks and average the predictions.

# Problems
Unfortunately, the project is not ideal and there are flaws in it.

### Training Speed
Training of one epoch for a good architecture takes 3-4 minutes, which is quite slow, but not catastrophic, because neural networks are characterized by long training due to the large number of mathematical operations. It may seem strange, but when I rewrote my code in Python (not all, but only the model) and started the training, it was faster than in C#. To be fair, I did not do it manually, but on the tensorflow framework, which most likely uses various tricks in its implementation to get around the "slowness" of Python.

### Speed ​​of some mathematical operations
This point is closely related to the point above, because if we compare the speed of some mathematical operations on tensorflow and in my implementation, for example, einsum, then the difference in speed is enormous. I used a naive approach to doing math, so I think there's a lot of possibilities for optimization here.

### Poor recognition of some digits
Some numbers are systematically poorly recognized, and I see two reasons for this:
* Insufficient training
* Some numbers in the training dataset are represented mainly by one way of writing

All these problems are solvable if enough time is given to them, so the project expects a lot of improvements.

### External dependencies
* MatthiWare.CommandLineParser [NuGet]
* Magick.NET-Q16-AnyCPU [NuGet]