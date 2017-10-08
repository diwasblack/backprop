import copy
import logging

import numpy as np
import matplotlib.pyplot as plt

logging.basicConfig(
    format='%(asctime)s %(levelname)s %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def sigmoid_function(x):
    return 1.0 / (1.0 + np.exp(-x))


def sigmoid_function_derivative(x):
    return x * (1.0 - x)


def bipolar_sigmoid_function(x):
    return 2.0 / (1.0 + np.exp(-x)) - 1.0


def bipolar_sigmoid_function_derivative(x):
    return (1.0 - np.square(x)) / 2.0


def hyperbolic_tangent(x):
    return np.tanh(x)


def hyperbolic_tangent_derivative(x):
    return 1.0 - np.square(x)


class NN():
    """
    Class for neural network
    """

    def __init__(self,
                 neuron_layers,
                 lr=0.01,
                 print_frequency=10000,
                 momentum_coefficient=0,
                 activation_function="sigmoid"):
        self.neuron_layers = neuron_layers
        self.number_of_layers = len(neuron_layers)

        # Store information of last weights and biases
        self.weights = []
        self.previous_weights = []
        self.biases = []
        self.previous_biases = []

        # Hyper parameters
        self.learning_rate = lr
        self.momentum_coefficient = momentum_coefficient

        # Store cost during training
        self.training_cost = []

        self.print_frequency = print_frequency

        # Initialize weights
        self.initialize_weights()

        # Set activation function
        self.set_activation_function(activation_function)

    def set_activation_function(self, activation_function):
        if activation_function == "tanh":
            self.activation_function = hyperbolic_tangent
            self.activation_function_derivative = hyperbolic_tangent_derivative
        elif activation_function == "sigmoid":
            self.activation_function = sigmoid_function
            self.activation_function_derivative = sigmoid_function_derivative
        elif activation_function == "bipolar_sigmoid":
            self.activation_function = bipolar_sigmoid_function
            self.activation_function_derivative = bipolar_sigmoid_function_derivative
        else:
            raise Exception("Activation function not found")

    def backup_weights(self):
        """
        Make a deep copy of the current weights and biases
        """

        self.previous_weights.append(copy.deepcopy(self.weights))
        self.previous_biases.append(copy.deepcopy(self.biases))

    def initialize_weights(self):
        """
        Initial weights
        """
        # Randomly initialize weights
        for i in range(1, self.number_of_layers):
            current_layer_neurons = self.neuron_layers[i]
            previous_layer_neurons = self.neuron_layers[i - 1]

            # Create a numpy array to hold weights
            # Randomly initialize weights using uniform distribution in the
            # range [-0.5, 0.5]
            layer_weights = np.random.rand(previous_layer_neurons,
                                           current_layer_neurons) - 0.5

            # Store layer weights
            self.weights.append(layer_weights)

            # Randomly initialize biases in range [-0.5, 0.5]
            random_biases = np.random.rand(current_layer_neurons, 1) - 0.5
            self.biases.append(random_biases)

        # Backup the weights and biases.
        # Must be done twice
        self.backup_weights()
        self.backup_weights()

    def predict(self, x):
        """
        Feed forward the values to obtain the predicted value
        """
        # Set layer_activation to x initially
        layer_activation = x
        for layer in range(0, self.number_of_layers - 1):

            # Obtain weights and biases
            weights = self.weights[layer]
            biases = self.biases[layer]

            # Assuming layer_activation is a row vector with dimension 1*m
            # Perform matrix multiplication and add bias
            z = np.dot(layer_activation, weights) + biases.T

            # Use the activation function
            layer_activation = self.activation_function(z)

        # Return activation obtained from final layer
        return layer_activation

    def cost_function(self, y_pred, y_target):
        """
        Calculate the cost function
        """

        y_pred = y_pred.reshape(len(y_target), 1)

        # Total squared error
        squared_error = np.linalg.norm(y_target - y_pred)

        # Mean squared error
        return squared_error / len(y_target)

    def train(self, input_array, output_array):
        """
        Train the neural network
        """
        logger.info("Training neural network")
        logger.info(
            "Using momentum_coefficient={}".format(self.momentum_coefficient))

        # Keep training until stopping criteria is met
        iteration = 0
        while (True):
            predicted_output_list = np.array([])
            for input_sample, output_sample in zip(input_array, output_array):
                activations = [input_sample.reshape(1, len(input_sample))]

                # Feed forward through the layers
                for layer in range(0, self.number_of_layers - 1):

                    # Obtain weights and biases
                    weights = self.weights[layer]
                    biases = self.biases[layer]

                    # Assuming layer_activation is a row vector with dimension
                    # 1*m
                    # Perform matrix multiplication and add bias
                    z = np.dot(activations[layer], weights) + biases.T

                    # Use the activation function
                    layer_activation = self.activation_function(z)

                    # Append this layers activation
                    activations.append(layer_activation)

                predicted_output = activations[-1]
                predicted_output_list = np.append(predicted_output_list,
                                                  predicted_output)
                delta = (output_sample - predicted_output) * \
                    self.activation_function_derivative(predicted_output)

                previous_weights = self.previous_weights.pop(0)
                previous_biases = self.previous_biases.pop(0)

                logger.debug("\n{}".format(
                    str(self.weights[-1] - previous_weights[-1])))

                # Update weights using SGD and momentum
                self.weights[-1] += self.learning_rate * \
                    activations[-2].T.dot(delta) + self.momentum_coefficient *\
                    (self.weights[-1] - previous_weights[-1])
                self.biases[-1] += self.learning_rate * delta + \
                    self.momentum_coefficient *\
                    (self.biases[-1] - previous_biases[-1])

                # Backpropagate delta through layers
                for layer in range(self.number_of_layers - 3, -1, -1):

                    next_layer_activation = activations[layer + 1].T

                    delta = np.dot(self.weights[
                        layer
                        + 1], delta) * self.activation_function_derivative(
                            next_layer_activation)

                    self.weights[layer] += self.learning_rate * np.dot(
                        activations[layer].T,
                        delta.T) + self.momentum_coefficient * (
                            self.weights[layer] - previous_weights[layer])
                    self.biases[
                        layer] += self.learning_rate * delta + self.momentum_coefficient * (
                            self.biases[layer] - previous_biases[layer])

                # Store weights
                self.backup_weights()

            cost = self.cost_function(predicted_output_list, output_array)
            self.training_cost.append(cost)

            logger.debug("Iteration: {}, Cost: {}".format(iteration, cost))

            if iteration % self.print_frequency == 0:
                logger.info("Iteration: {}, Cost: {}".format(iteration, cost))
            if cost <= 0.01:
                break

            if iteration >= 1000000:
                break
            iteration += 1

    def plot_training_cost(self):
        """
        Plot a graph of training_cost vs number of iterations
        """
        training_costs = self.training_cost[:]

        plt.plot(list(range(len(training_costs))), training_costs)
        plt.xlabel("Number of iterations")
        plt.ylabel("Training cost")
        plt.savefig("cost.png")


def main():
    # Create a neural network model
    model = NN([2, 4, 1], activation_function="tanh", momentum_coefficient=0.5)
    x = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Train model
    model.train(x, y)

    # Plot and save training cost graph
    model.plot_training_cost()

    # Predict output
    logger.info("Predicted values:\n{}".format(model.predict(x)))


if __name__ == "__main__":
    main()
