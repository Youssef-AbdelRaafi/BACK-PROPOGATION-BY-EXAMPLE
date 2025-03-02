import random

class NeuralNetwork:
    def __init__(self):
        # Constants
        self.LEARNING_RATE = 0.5
        self.INPUT_NEURONS = 2
        self.HIDDEN_NEURONS = 2
        self.OUTPUT_NEURONS = 2

        self.hidden_layer_weights = [[random.uniform(-1, 1) for _ in range(self.HIDDEN_NEURONS)] for _ in range(self.INPUT_NEURONS)]
        self.output_layer_weights = [[random.uniform(-1, 1) for _ in range(self.OUTPUT_NEURONS)] for _ in range(self.HIDDEN_NEURONS)]
        self.hidden_layer_biases = [random.uniform(-1, 1) for _ in range(self.HIDDEN_NEURONS)]
        self.output_layer_biases = [random.uniform(-1, 1) for _ in range(self.OUTPUT_NEURONS)]

    def sigmoid(self, x):
        """Compute the sigmoid activation function."""
        return 1 / (1 + 2.71828 ** (-x))

    def sigmoid_derivative(self, x):
        """Compute the derivative of the sigmoid function."""
        return x * (1 - x)

    def feedforward(self, inputs):
        """Perform the feedforward pass through the network."""

        # Hidden layer calculations
        self.hidden_layer_input = [sum(inputs[i] * self.hidden_layer_weights[i][j] for i in range(self.INPUT_NEURONS)) + self.hidden_layer_biases[j]
                                   for j in range(self.HIDDEN_NEURONS)]
        self.hidden_layer_output = [self.sigmoid(x) for x in self.hidden_layer_input]

        # Output layer calculations
        self.output_layer_input = [sum(self.hidden_layer_output[i] * self.output_layer_weights[i][j] for i in range(self.HIDDEN_NEURONS)) + self.output_layer_biases[j]
                                   for j in range(self.OUTPUT_NEURONS)]
        self.output_layer_output = [self.sigmoid(x) for x in self.output_layer_input]
        return self.output_layer_output

    def backpropagate(self, inputs, targets):
        """Perform backpropagation to update weights and biases."""
        output_error = [targets[i] - self.output_layer_output[i] for i in range(self.OUTPUT_NEURONS)]
        output_delta = [output_error[i] * self.sigmoid_derivative(self.output_layer_output[i]) for i in range(self.OUTPUT_NEURONS)]

        hidden_error = [sum(output_delta[j] * self.output_layer_weights[i][j] for j in range(self.OUTPUT_NEURONS)) for i in range(self.HIDDEN_NEURONS)]
        hidden_delta = [hidden_error[i] * self.sigmoid_derivative(self.hidden_layer_output[i]) for i in range(self.HIDDEN_NEURONS)]

        for i in range(self.HIDDEN_NEURONS):
            for j in range(self.OUTPUT_NEURONS):
                self.output_layer_weights[i][j] += self.LEARNING_RATE * output_delta[j] * self.hidden_layer_output[i]

        for j in range(self.OUTPUT_NEURONS):
            self.output_layer_biases[j] += self.LEARNING_RATE * output_delta[j]

        for i in range(self.INPUT_NEURONS):
            for j in range(self.HIDDEN_NEURONS):
                self.hidden_layer_weights[i][j] += self.LEARNING_RATE * hidden_delta[j] * inputs[i]

        for j in range(self.HIDDEN_NEURONS):
            self.hidden_layer_biases[j] += self.LEARNING_RATE * hidden_delta[j]

    def train(self, inputs, targets, epochs):
        """Train the neural network for a given number of epochs."""
        for epoch in range(epochs):
            self.feedforward(inputs)
            self.backpropagate(inputs, targets)
            if epoch % 1000 == 0:
                print(f"Epoch {epoch}, Output: {self.output_layer_output}")

if __name__ == "__main__":
    inputs = [0.05, 0.10]
    targets = [0.01, 0.99]

    nn = NeuralNetwork()
    nn.train(inputs, targets, epochs=10000)

    print("Final Output:", nn.feedforward(inputs))