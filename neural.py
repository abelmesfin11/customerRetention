#!/usr/bin/env python

import numpy as np
import pandas as pd
import random
import math
from sklearn.model_selection import train_test_split
from missforest import MissForest
from sklearn.metrics import classification_report

class Value:

    def __init__(self, data=None, label="", _parents=set(), _operation=""):
        """
        Constructor for a value node that holds data + gradient + how the value was produced
        """

        # initialize data randomly between -1 and 1 if no data was provided
        self.data = random.uniform(-1, 1) if data is None else data

        # initialize a gradient and label used for printing the value
        self.grad = 0
        self.label = label

        # initialize how the value was created (what operation using what inputs)
        # initialize how the backward pass for this operation executes
        self._parents = set(_parents)
        self._operation = _operation
        self._backward = lambda: None

    def __repr__(self):
        """
        Helper class used for printing the Value
        """
        return f"Value(data={self.data}, label={self.label})"

    def __add__(self, other):
        """
        Implementing the + operator
        """

        # perform the operation
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, _parents=(self, other), _operation='+')

        # assign how the gradients are propagated backwards
        def _backward():
            self.grad += 1*out.grad
            other.grad += 1*out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        """
        Implementing the * operator
        """

        # perform the operation
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, _parents=(self, other), _operation='*')

        # assign how the gradients are propagated backwards
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out


    def __pow__(self, other):
        """
        Implementation of the power operator **
        """

        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, _parents=(self,), _operation="**" + str(other))

        def _backward():
            self.grad += (other * (self.data**(other-1))) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        """
        Implementation of ReLU activation function
        """

        out = Value(0 if self.data < 0 else self.data, _parents=(self,), _operation=("ReLU"))

        def _backward():
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def backward(self):
        """
        Calls the _backward() method for each node in reverse topological order
        """

        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._parents:
                    build_topo(parent)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

    def exp(self):
        """
        Implementing exponent
        """
        out = Value(math.exp(self.data), _parents=(self,), _operation="exp")
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def log(self):
        """
        Implementing log
        """
        out = Value(math.log(self.data), _parents=(self,), _operation="log")
        def _backward():
            self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out


    def __float__(self): return float(self.data)
    def __radd__(self, other): return self + other
    def __rmul__(self, other): return self * other
    def __neg__(self): return self * -1
    def __sub__(self, other):  return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1



def accuracy(Y, Yhat):
    """
    Function for computing accuracy

    Y is a vector of the true labels and Yhat is a vector of estimated 0/1 labels
    """

    acc = 0
    for y, yhat in zip(Y, Yhat):

        if y == yhat: acc += 1

    return acc/len(Y) * 100


def sigmoid(value, scale=0.5):
    """
    Returns sig(value) using the scale parameter
    """
    scaled = value * scale
    return Value(1) / ((-scaled).exp() + Value(1))

def negative_loglikelihood(y, pY1):
    """
    Return negative loglikelihood for a single example based on the value of Y and p(Y=1 | ...)
    """
    if y == 1:
        return (pY1 + 1e-10).log() * -1
    else:
        return ((Value(1) - pY1) + 1e-10).log() * -1


class Neuron:
    """
    Class that represents a single neuron in a neural network.
    A neuron first computes a linear combination of its inputs + an intercept,
    and then (potentially) passes it through a non-linear activation function
    """

    def __init__(self, n_inputs):
        """
        Constructor which initializes a parameter for each input, and one parameter for an intercept
        """
        self.theta = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.intercept = Value(random.uniform(-1, 1))

    def __call__(self, x, relu=False, dropout_proba=0.1, train_mode=False):
        """
        Implementing the function call operator ()
        """
        p = 1 - dropout_proba

        if train_mode:
            out = Value(0)
            for i in range(len(self.theta)):
                mask = 1 if random.random() < p else 0
                out = out + self.theta[i] * x[i] * mask
            out = out + self.intercept
        else:
            out = sum([self.theta[i] * x[i] * p for i in range(len(self.theta))]) + self.intercept

        # activate using ReLU based on boolean flag
        if relu:
            return out.relu()
        return sigmoid(out)

    def parameters(self):
        """
        Method that returns a list of all parameters of the neuron
        """
        return self.theta + [self.intercept]

class Layer:
    """
    Class for implementing a single layer of a neural network
    """

    def __init__(self, n_inputs, n_outputs):
        """
        Constructor initializes the layer with neurons that ta
        """
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x, relu=True, dropout_proba=0.1, train_mode=False):
        """
        Implementing the function call operator ()
        """

        # produces a list of outputs for each neuron
        outputs = [n(x, relu, dropout_proba, train_mode=train_mode) for n in self.neurons]
        return outputs[0] if len(outputs) == 1 else outputs

    def parameters(self):
        """
        Method that returns a list of all parameters of the layer
        """
        return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
    """
    Class for implementing a multilayer perceptron
    """

    def __init__(self, n_features, layer_sizes, learning_rate=0.01, dropout_proba=0.1):
        """
        Constructor that initializes layers of appropriate width
        """

        layer_sizes = [n_features] + layer_sizes
        self.layers = [Layer(layer_sizes[i], layer_sizes[i+1]) for i in range(len(layer_sizes)-1)]
        self.dropout_proba = dropout_proba
        self.learning_rate = learning_rate
        # boolean flag that determines whether we are currently training or testing
        # helpful for controlling how dropout is used
        self.train_mode = True

    def __call__(self, x):
        """
        Impelementing the call () operator which simply calls each layer in the net
        sequentially using outputs of previous layers
        """

        # use ReLU activation for all layers except the last one
        out = x
        for layer in self.layers[0:len(self.layers)-1]:
            out = layer(out, relu=True, dropout_proba =self.dropout_proba, train_mode=self.train_mode)
        return self.layers[-1](out, relu=False)

    def parameters(self):
        """
        Method that returns a list of all parameters of the neural network
        """
        return [p for layer in self.layers for p in layer.parameters()]

    def _zero_grad(self):
        """
        Method that sets the gradients of all parameters to zero
        """
        for p in self.parameters():
            p.grad = 0

    def fit(self, Xmat_train, Y_train, Xmat_val=None, Y_val=None, max_epochs=100, verbose=False):
        """
        Fit parameters of the neural network to given data using SGD.
        SGD ends after reaching a maximum number of epochs.

        Can optionally take in validation inputs as well to test generalization error.
        """

        # initialize all parameters randomly
        for p in self.parameters():
            p.data = random.uniform(-1, 1)

        # iterate over epochs
        for e in range(max_epochs):

            # TODO implement SGD
            for x, y in zip(Xmat_train, Y_train):
                self._zero_grad()
                out = self(x)
                loss = negative_loglikelihood(y, out)
                loss.backward()
                for p in self.parameters():
                    p.data -= self.learning_rate * p.grad


            # use the verbose flag with True when debugging your outputs
            if verbose:
                self.train_mode = False

                train_acc = accuracy(Y_train, self.predict(Xmat_train))

                if Xmat_val is not None:
                    val_acc = accuracy(Y_val, self.predict(Xmat_val))
                    print(f"Epoch {e}: Training accuracy {train_acc:.0f}%, Validation accuracy {val_acc:.0f}%")
                else:
                    print(f"Epoch {e}: Training accuracy {train_acc:.0f}%")

                self.train_mode = True

        # at the end of training set train mode to be False
        self.train_mode = False

    def predict(self, Xmat):
        """
        Predict method which returns a list of 0/1 labels for the given inputs
        """

        return [int(self(x).data > 0.5) for x in Xmat]


def load_customer_data():
    """
    Cleaning
    """
    # Load and clean raw data
    data = pd.read_csv("CustomerChurn.csv")
    data = data.drop(columns=['Customer ID', 'LoyaltyID'], errors='ignore')
    data["Total Charges"] = pd.to_numeric(data["Total Charges"], errors="coerce")
    data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})

    # Separate Total Charges for imputation
    data_encoded = pd.get_dummies(data.drop(columns=["Total Charges"]))
    data_encoded["Total Charges"] = data["Total Charges"]
    data_encoded = data_encoded.astype(np.float64)

    # Impute missing values using MissForest
    imputer = MissForest()
    data_imputed_array = imputer.fit_transform(data_encoded)
    data_imputed = pd.DataFrame(data_imputed_array, columns=data_encoded.columns)

    # Re-attach target
    data_imputed["Churn"] = data["Churn"].values

    # Final feature matrix and label vector
    feature_names = list(data_imputed.columns)
    feature_names.remove("Churn")
    X = data_imputed[feature_names].to_numpy(dtype=np.float64)
    Y = data_imputed["Churn"].to_numpy(dtype=np.float64)

    # Split: Train/Val/Test with 80-10-10 ratio
    X_train, X_test, Y_train, Y_test  = train_test_split(X, Y, test_size=0.20, random_state=42)
    X_train, X_val, Y_train, Y_val  = train_test_split(X_train, Y_train, test_size=0.50, random_state=42)


    # Standardize using training set statistics
    means = X_train.mean(axis=0)
    stds = X_train.std(axis=0)
    X_train = (X_train - means) / stds
    X_val = (X_val - means) / stds
    X_test = (X_test - means) / stds


    d = X_train.shape[1]
    model = MLP(n_features=d, layer_sizes=[16, 8, 4, 1], learning_rate=0.05, dropout_proba=0.5)
    model.fit(X_train, Y_train, X_val, Y_val, max_epochs=25, verbose=True)

    return model, X_test, Y_test


def main():
    """
    Do not edit this function. This function is used for grading purposes only."
    """

    random.seed(42)
    model, X_test, Y_test = load_customer_data()

    # test final model
    # test_acc = accuracy(Y_test, model.predict(X_test))
    # print(f"Customer test accuracy {test_acc:.0f}%")

    # testing final model using sklearn
    print(classification_report(Y_test, model.predict(X_test)))

if __name__ == "__main__":
    main()
