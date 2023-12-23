/*
This software is released under the MIT License.
Copyright (c) 2023 Ethan Henry
*/


//First specify the computational environment. Other options are: AC_USE_GPU
#define AC_USE_CPU

//Then specify the library of use. Other options are: AC_WITH_VIENNACL (GPU), AC_WITH_CUDA (GPU)
#define AC_WITH_EIGEN

//Then include the AlphaCore library, after macros are defined.
#include "AlphaCore.h"

#include <iostream>
#include <vector>

int main(int argc, const char * argv[]) {
    // insert code here...
    std::cout << "Hello, World!\n";
    
    //Instantiate a neural network as an object. The first parameter is an array describing the layers of the neural network. The second parameter describes the initialization of weights.
    AlphaCore::CPUFeedForwardNeuralNetwork XOR({2, 5, 2, 2}, AlphaCore::Arguments::GLOROT_INIT);
    //...to load a saved neural network from a file, simply pass in the file location as a std::string as the constructor parameter. Saving a network to a file will be demonstrated at the end.
    
    //Set activations by first passing a number corresponding to the layer, and then a function from the AlphaCore library corresponding to the activation function. Layer numbers for this function start at 0, with zero indicating the first hidden layer. Custom activation functions can be set by passing in a lambda or function pointer to a custom function. The same goes for the derivative fucntions of each layer. Note that, by default, all activation functions are Sigmoid and all derivative functions are the derivatives of sigmoid.
    XOR.setActivations(0, AlphaCore::sigmoid);
    XOR.setDerivatives(0, AlphaCore::sigmoidder);
    XOR.setActivations(1, AlphaCore::sigmoid);
    XOR.setDerivatives(1, AlphaCore::sigmoidder);
    XOR.setActivations(2, AlphaCore::sigmoid);
    XOR.setDerivatives(2, AlphaCore::sigmoidder);
    
    
    //Set Skip Connections by passing in the layer number it begins at, then the layer number it connects to. Skip connections are calculated element-wise.
    XOR.addElementwiseSkipConnection(0, 2);
    
    
    //Choose a data set with input-target pairs
    std::vector<std::vector<float>> inputs = {{0, 1}, {1,0}, {0,0}, {1, 1}};
    std::vector<std::vector<float>> targets = {{1, 0}, {1,0}, {0,1}, {0, 1}};
    
    
    int epochs = 100;
    
    //Train network using the .backpropagate() function. For most functions that require a vector of information as a parameter, the vector should be passed in as an r-value reference, done by either type casting or using the C++11 provided macro std::move();
    for (int e = 0; e < epochs; e++ ) {
        for (int i = 0; i < inputs.size(); i++) {
            XOR.backpropagate(std::move(inputs[i]), std::move(targets[i]));
        }
    }
    //...or use the .trainSet() function to automate this process. The line --> XOR.trainSet(std::move(inputs), std::move(targets), epochs); <-- does exactly the same thing as the code above.
    
    
    //The .feedforward() accepts a vector of information as its only parameter, feeds that information through the network, and then returns the output layer of the network as a single-dimension vector.
    std::vector<float> input = {0, 1};
    std::vector<float> outputs = XOR.feedforward(input);
    if (outputs[0] > outputs[1]) {
        std::cout << "1" << std::endl;
    } else {
        std::cout << "0" << std::endl;
    }
    
    //To save a neural network to a file, simply pass in the location and desired name for the folder that the information of the network is to be stored in.
    XOR.save("Some/Random/Filepath/TrainedNetwork");
    //...note that, in the example above, a new folder with the name "TrainedNetwork" will be created at "Some/Random/Filepath/"
    //This means that, to load this network if it had already been saved, "Some/Random/Filepath/TrainedNetwork" would have been the file name passed into the network constructor.
    return 0;
}

//The AlphaCore library contains a plethora of other rich features that have not been demonstrated here. What is shown here is a simple, rudimentary example of how the library can be used, illustrating only the most critical functionality of the library.

/*
MIT License

Copyright (c) 2023 Ethan Henry

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

1. The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/
