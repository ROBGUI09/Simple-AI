import numpy
import random
import pickle

def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

def sigmoid_derivative(x):
    """Производная сигмоиды"""
    sig = sigmoid(x)
    return sig * (1 - sig)

class Layer:
    rels = None
    output = None
    input = None

    def __init__(self, count, next_count):
        self.rels = numpy.random.rand(count, next_count)

    def count(self):
        return len(self.rels)

    def next(self):
        return len(self.rels[0])

    def activate(self, indat):
        """Выполняет активацию слоя и сохраняет вход/выход"""
        self.input = indat
        out = numpy.dot(self.rels.T, indat)
        out_activated = sigmoid(out)
        self.output = out_activated
        return out_activated

    def backpropagate(self, delta_next_layer, learning_rate):
        delta = numpy.dot(self.rels, delta_next_layer) * sigmoid_derivative(numpy.dot(self.rels.T, self.input))

        delta_rels = learning_rate * numpy.outer(self.input, delta_next_layer)
        self.rels -= delta_rels

        return delta


class NeuralNetwork:
    def __init__(self, layers_config):
        self.layers = []
        for i in range(len(layers_config) - 1):
            self.layers.append(Layer(layers_config[i], layers_config[i+1]))

    def forward_pass(self, input_data):
        output = input_data
        for layer in self.layers:
            output = layer.activate(output)
        return output

    def predict(self, input_data):
        input_arr = numpy.array(input_data) if isinstance(input_data, list) else numpy.array([input_data])
        output = self.forward_pass(input_arr)
        return output

    def train_step(self, x, y, learning_rate):
            output_layer = self.forward_pass(x)
            loss = self.mse(output_layer, y)

            error_delta = self.mse_derivative(output_layer, y) * sigmoid_derivative(numpy.dot(self.layers[-1].rels.T, self.layers[-1].input))
            delta = error_delta
            for layer in reversed(self.layers):
                 delta = layer.backpropagate(delta, learning_rate)

            return loss

    def train(self, training_data, learning_rate, epochs, callback = None):
        loss_history = []
        for epoch in range(epochs):
             total_loss = 0
             for data in training_data:
                 x = data[0] if isinstance(data[0],list) else numpy.array([data[0]])
                 y = data[1] if isinstance(data[1],list) else numpy.array([data[1]])
                 loss = self.train_step(x,y,learning_rate)
                 total_loss += loss
             avg_loss = total_loss/len(training_data)
             loss_history.append(avg_loss)

             if callback is not None:
                callback(epoch, avg_loss)
        return loss_history

    def mse(self, output, target):
        return numpy.mean((output - target) ** 2)

    def mse_derivative(self, output, target):
        return 2*(output-target)

    def save_weights(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump([layer.rels for layer in self.layers], f)

    def load_weights(self, filename):
         with open(filename, 'rb') as f:
            weights = pickle.load(f)
            for i, layer in enumerate(self.layers):
               layer.rels = weights[i]

