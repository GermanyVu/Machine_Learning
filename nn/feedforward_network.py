import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib as mpl
import copy


class FFNN:
    def __init__(self):
        # size is a list ex:[2,3,1] which tells you the number of neurons in
        # in each layer including the input layer
        # this is a neural network 2 layers deep input layer is not counted

        self.size_list = []
        self.layer_idx_list = []
        self.bias_list = []
        self.weight_list = []
        self.activations = []
        self.z_list = []
        self.sigmoid_primes = []
        self.eta = 3.0

    def __repr__(self):
        return f"FFNN(sizes={self.size_list}, biases={len(self.bias_list)},weights={len(self.weight_list)})"

    def add_layer(self, number_neurons):
        """adds a single neuron layer and caches the updated size and index the layers"""
        self.size_list.append(number_neurons)
        self.layer_idx_list = [i for i in range(len(self.size_list))]

    def make_weights(self):
        self.bias_list = [
            np.zeros((y, 1)) for y in self.size_list[1:]
        ]  # creating column vectors for each layer
        # except input layer
        # ex zeros(3,1) and zeros(1,1)  numpy array
        self.weight_list = [
            np.random.randn(y, x) * 0.01
            for x, y in zip(self.size_list[:-1], self.size_list[1:])
        ]  # creates random weights

    def _ReLU(self, z):
        """function that will allow for faster convergence than sigmoid"""
        return np.maximum(0, z)

    def _ReLU_prime(self, z):
        return 1 * (z > 0)

    def _sigmoid(self, z):
        """The sigmoid function."""
        return 1.0 / (1.0 + np.exp(-z))

    def _sigmoid_prime(self, z):
        """Derivative of the sigmoid function."""
        return self._sigmoid(z) * (1 - self._sigmoid(z))

    def _cost(
        self, a, label
    ):  # a and label are both vectors. Each row represents a different
        # sample
        return 0.5 * np.sum((a - label) ** 2)

    def _cost_ce(
        self, a, label
    ):  # a and label are both vectors. Each row represents a different
        # sample
        """cross entropy cost function"""
        return np.sum(
            ((-np.log(a)) * label + (-np.log(1 - a)) * (1 - label))
        )  # compute cost

    def _cost_mult_sigmoid_prime(self, a, label):
        return a - label  # c'(a^L)*g'(z^L)

    def _cost_prime_ce(self, a, label):
        """cross entropy prime"""
        return label / a - (1 - label) / (1 - a)

    def _forward_pass(self, a, weight_list, bias_list):
        """
        The point of this function is to use weights
        and biases to create a predicted final layer output
        """

        for b, w in zip(
            bias_list, weight_list
        ):  # by looping the weifhts and biases like this we don;t have to worry about
            # which layer to extract them from
            # this is a first pass throughout the entire Network
            # can be used to calculate gradient checks

            a = self._sigmoid(np.dot(w, a) + b)
        #
        return a

    def _feedforward(self, a):
        self.activations = [a]  # input layer
        self.z_list = []

        for b, w in zip(self.bias_list, self.weight_list):
            z = np.dot(w, a) + b
            a = self._sigmoid(z)  # in the first layer

            self.z_list.append(z)
            self.activations.append(a)

    def _alter_biases(self, eps, row, layer):
        """helper function for finite difference
        creates a small change in a single element in the bias
        matrix"""
        biases_p = copy.deepcopy(self.bias_list)
        biases_m = copy.deepcopy(self.bias_list)
        biases_p[layer][row] += eps
        biases_m[layer][row] -= eps
        return biases_p, biases_m

    def _alter_weights(self, eps, row, col, layer):
        """helper fn for finite finite_difference
        creates a small change in the weight for a singel element"""
        weights_p = copy.deepcopy(self.weight_list)
        weights_m = copy.deepcopy(self.weight_list)
        weights_p[layer][row][col] += eps
        weights_m[layer][row][col] -= eps

        return weights_p, weights_m

    def finite_difference(self, cost_fn, input_neurons, label, row, col, layer):
        """computes the finite difference of cost function"""
        eps = 10**-6
        weight_list = copy.deepcopy(self.weight_list)
        bias_list = copy.deepcopy(self.bias_list)
        weights_p, weights_m = self._alter_weights(eps, row, col, layer)
        biases_p, biases_m = self._alter_biases(eps, row, layer)

        output_weight_p = self._forward_pass(
            input_neurons, weights_p, bias_list
        )  # weight + eps
        output_weight_m = self._forward_pass(
            input_neurons, weights_m, bias_list
        )  # weight - eps

        output_bias_p = self._forward_pass(
            input_neurons, weight_list, biases_p
        )  # bias + eps
        output_bias_m = self._forward_pass(
            input_neurons, weight_list, biases_m
        )  # bias - eps

        dc_dw = (cost_fn(output_weight_p, label) - cost_fn(output_weight_m, label)) / (
            2 * eps
        )
        dc_db = (cost_fn(output_bias_p, label) - cost_fn(output_bias_m, label)) / (
            2 * eps
        )
        return dc_dw, dc_db

    def _backpropagation(self, label):
        gradient_biases = []
        gradient_weights = []

        """ deepest layer delta
            delta^L = c'(a^L)*g'(z^L)
            or delta^L = a - label if using sigmoid
        """
        delta_list = [(self._cost_mult_sigmoid_prime(self.activations[-1], label))]

        """ more shallow layer deltas
            delta_j^l = Sum_k(delta_k^(l+1) @ W_jk^(l+1) * g'(z_j)^l )
            delta_j^L-1 = Sum_k(delta_k^(L))@W_jk^(L) * g'(z_j)^L-1
        """

        for weight, delta, z in zip(
            self.weight_list[1:][::-1], delta_list, self.z_list[0:-1][::-1]
        ):
            delta_list.append(np.dot(weight.T, delta) * self._sigmoid_prime(z))

        """ gradients for biases and weights for all layers
            dc/db_j^l = delta_j^l
            dc/dw_ij^l = delta_j^l+1 * a_i^l"""
        for (
            delta,
            a,
        ) in zip(
            delta_list[::-1], self.activations[0:-1]
        ):  # reversed the deltas list and sliced the input neurons
            # out of the activation list
            # they should be the same size and going from first to last
            # layer

            gradient_weights.append(delta * a.T)
            gradient_biases.append(delta)

        #         ipy_exit()
        return gradient_weights, gradient_biases

    def _gradient_descent(
        self, gradient_weights_tot, gradient_biases_tot, mini_batch_size
    ):
        for idx in self.layer_idx_list[:-1]:  # cut off the end
            self.weight_list[idx] -= (
                self.eta / mini_batch_size
            ) * gradient_weights_tot[idx]
            self.bias_list[idx] -= (self.eta / mini_batch_size) * gradient_biases_tot[
                idx
            ]

    def _update_mini_batch(self, mini_batch, mini_batch_size):
        """goes through batch and calls ff and backprop to product a tot grad weights and bias
        for batch"""
        gradient_biases_tot = [np.zeros(b.shape) for b in self.bias_list]
        gradient_weights_tot = [np.zeros(w.shape) for w in self.weight_list]

        for training_input, label in mini_batch:
            # run feedforward
            self._feedforward(training_input)

            # run backprop which will update the gradients
            gradient_weights, gradient_biases = self._backpropagation(label)

            # created a tot gradient which is the sum of all gradients in batch
            for grad_b, idx in zip(gradient_biases, self.layer_idx_list[0:-1]):
                gradient_biases_tot[idx] += grad_b
            for grad_w, idx in zip(gradient_weights, self.layer_idx_list[0:-1]):
                gradient_weights_tot[idx] += grad_w

        self._gradient_descent(
            gradient_weights_tot, gradient_biases_tot, mini_batch_size
        )

    def predict(self, test_inputs, test_labels):
        """shows how accurate the model is
        only used for binary results! 0 or 1"""
        A2 = [
            (self._forward_pass(inputs, self.weight_list, self.bias_list))
            for inputs in test_inputs
        ]

        predictions = [
            A2[i][0][0] <= test_labels[i][0] + 0.5
            and A2[i][0][0] >= test_labels[i][0] - 0.5
            for i in range(len(test_labels))
        ]
        return predictions

    def _evaluate(self, test_data):
        """Return the number of test inputs for which the neural
        network outputs the correct result. Note that the neural
        network's output is assumed to be the index of whichever
        neuron in the final layer has the highest activation."""
        test_results = []

        for a, label in test_data:
            test_results.append(
                (
                    np.argmax(self._forward_pass(a, self.weight_list, self.bias_list)),
                    label,
                )
            )

        return sum(int(x == y) for (x, y) in test_results)

    def stochastic_gradient(
        self, training_data, epochs, mini_batch_size, test_data=None, test_inputs=None
    ):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs.  The other non-optional parameters are
        self-explanatory.  If ``test_data`` is provided then the
        network will be evaluated against the test data after each
        epoch, and partial progress printed out.  This is useful for
        tracking progress, but slows things down substantially."""
        if test_data:
            n_test = len(test_data)
        number_samples = len(training_data)  # number of samples

        for j in range(epochs):
            np.random.shuffle(training_data)

            # creates mini batches from data
            mini_batches = [
                training_data[k : k + mini_batch_size]
                for k in range(0, number_samples, mini_batch_size)
            ]

            for mini_batch in mini_batches:
                self._update_mini_batch(mini_batch, mini_batch_size)

            if (test_data and j % 5 == 0) or (test_data and j == epochs - 1):
                print("Epoch {0}: {1} ".format(j, self._evaluate(test_data) / n_test))
