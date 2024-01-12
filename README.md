[//]: # (TODO make this markdown)

# ZUtilLib Quick Guide
## Note
This library is not specifically intended to be useful really to anyone besides myself, so why I am making a guide is just for fun. And testing __***formatting***__. Do note that this library is, and will *always* be a work in progress, though most of it will be finished to a usable state eventually (of course). Documentation is available for *most* features.

## Basic Utilities
Under the `ZUtilLib` namespace directly is where the most generic and uncategorizable methods (some extension) are. Basically stuff that's useful for, well, whatever they say they do.

## Maths Stuff
Under the namespace `ZUtilLib.ZMath` a whole bunch of classes, an interface, and methods under those classes can be found. These are all a part of my "**Object Oriented Algebraic Calculator**" system that I came up with that can do trivial algebraic stuff like substituting variables, or random calculus things like differentiation and integration of equations. As of writing this, integration and *division* (coming later), have not been implemented. Do note, future self and other readers, that this part of the library is currently entirely undocumented.

## AI (Neural Networks)
So instead of using someone else's obscure AI library, I decided to make my *own* obscure AI library. Stuff for specifically training networks is not provided, but the generating, initializing, deriving, mutating, calculating, and data structures for neural networks is all provided. This is all under the `ZUtilLib.ZAI` namespace, and will be relatively well documented when it is at a usable state. Activation functions and other stuff is included and utilized internally, so it should be quite easy and effective to customize and use. Below I'll write up a quick tutorial so that I can remember what do to, then wonder *why* I made it that way.

## Basic Neural Network Usage Tutorial
All of these steps are required for the most basic usage of neural networks.

1. Instantite a new NN: `NeuralNetwork testnet = new NeuralNetwork(3, 3, 5, 2, NDNodeActivFunc.ReLU);`. Intellisense will show you what the parameter names are, and the documentation assigned to them. Calling the constructor essentially creates a new NN instance of the specified sizes.

2. Initialize the NN: `testNet.InitializeThis();`. What this does is generate all of the nodes and links with randomized weights and biases. An optional parameter is a float, for amplifying the randomness.

3. Calculate the result by passing in the inputs: `float[] result = testNet.PerformCalculations(0.2f, 0.3fm 0.4f);`. This method takes a params array of the values of the input nodes.

4. So now you've gotten your results, and scored the neural network or whatever you wish to do. To clone the network you'll have to instantiate a new network with identical specifications: `NeuralNetwork secondTestNet = new NeuralNetwork(3, 3, 5, 2, NDNodeActivFunc.ReLU);`.

5. Then clone it via an overload of the InitializeThis method: `secondTestNet.InitializeThis(testNet, 1, 1);`. The two floats following the network to be cloned determine mutation chance and learning rate for it to be cloned by. There's also an optional bool for if you want the changes to be relative (default off).

Packaging networks for JSON serialization can be done using the `PackagedNeuralNetwork` struct in `ZAI.Saving`, by passing the `NeuralNetwork` into the constructor. Unpack the `PackagedNeuralNetwork` by passing it into the `NeuralNetwork` constructor.

## Convolutional Neural Network Usage
TODO - Add this sometime.