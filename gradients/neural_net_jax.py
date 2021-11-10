#!/usr/bin/env python3

import jax.numpy as jnp
import jax.random as jrn
import jax.profiler as jpr
import random
from jax import grad, vmap
import sys
from time import time

input_dimensions = 2
output_dimensions = 1
learning_rate = 0.15
number_of_samples = 100
number_of_epochs = 1


def activation_func(x: float) -> float:
    return 1.0 / (1.0 + jnp.exp(-x))


def neuron_output(weight: jnp.ndarray, inp: jnp.ndarray) -> float:
    return jnp.sum(jnp.append(inp, 1) * jnp.transpose(weight))


def loss_func(expected_output: float, weight: jnp.ndarray, inp: jnp.ndarray) -> float:
    activated_neuron_output: float = activation_func(neuron_output(weight, inp))
    loss: float = pow((expected_output - activated_neuron_output), 2)
    return loss


class Network:
    def __init__(self, input_neurons: int, output_neurons: int, learning_rate: float):
        assert (input_neurons > 0)
        assert (output_neurons > 0)
        self.input_dim: int = input_neurons
        self.output_dim: int = output_neurons
        self.lr: float = learning_rate

        self.output_values: jnp.ndarray = jnp.array([])
        self.input_values: jnp.ndarray = jnp.array([])

        self.error_backpropagate = grad(loss_func, argnums=1)
        self.af_backpropagate = grad(activation_func, argnums=0)

        self.weights: jnp.ndarray = jrn.uniform(jrn.PRNGKey(0), shape=(self.output_dim, self.input_dim + 1))

    def feed_forward(self, inp: jnp.ndarray) -> jnp.ndarray:
        assert (self.input_dim == len(inp))
        self.output_values = []
        self.input_values = inp
        for weight in self.weights:
            self.output_values.append(activation_func(neuron_output(weight, inp)))
        return jnp.asarray(self.output_values)

    def calculate_weights(self, exp_value, local_weights):
        activation_backprop = self.af_backpropagate(exp_value)
        network_backprop = self.error_backpropagate(exp_value, local_weights, self.input_values)
        return local_weights - self.lr * network_backprop * activation_backprop

    def back_propagate(self, exp_values: jnp.ndarray):
        assert (len(exp_values) == self.output_dim)
        self.weights = vmap(self.calculate_weights)(exp_values, self.weights)

    def train(self, train_data: jnp.ndarray, nr_epochs: int = 10000):
        for i in range(nr_epochs):
            ind: int = random.randint(0, len(train_data))
            output: jnp.ndarray = self.feed_forward(jnp.asarray(train_data[ind][:self.input_dim]))
            self.back_propagate(jnp.asarray(train_data[ind][self.input_dim:]))
            if i % 10 == 9:
                print(f"{i + 1} iteration: weight update: {self.weights[0][0]}")

    def test(self, train_data: jnp.ndarray):
        acc: float = 0.0
        print(f"after weights: {self.weights[0][0]}")
        for data in train_data:
            output: jnp.ndarray = self.feed_forward(jnp.asarray(data[:self.input_dim]).reshape(-1, 1))
            correct_pred = False
            if self.output_dim == 1:
                correct_pred = int(round(output[0])) == int(round(data[self.input_dim]))
            else:
                correct_pred = jnp.where(output == jnp.amax(output)) == jnp.where(data[self.input_dim:] == jnp.amax(data[self.input_dim:]))
            acc += 1 if correct_pred else 0
        print(f"Accuracy: {acc / len(train_data)}")


def main():
    start = time()
    # jpr.start_trace("/tmp/tensorboard")
    global input_dimensions, output_dimensions, number_of_epochs, number_of_samples
    if len(sys.argv) == 5:
        input_dimensions = int(sys.argv[1])
        output_dimensions = int(sys.argv[2])
        number_of_epochs = int(sys.argv[3])
        number_of_samples = int(sys.argv[4])
    network: Network = Network(input_dimensions, output_dimensions, learning_rate)

    # x1 = jrn.uniform(jrn.PRNGKey(0), shape=(number_of_samples // 2, 1))
    # x2 = (-(5 / 7) * x1) + jrn.uniform(jrn.PRNGKey(0), shape=(number_of_samples // 2, 1)) + 0.0001
    # x3 = jnp.ones((50, 1))
    #
    # train_data = jnp.concatenate([x1, x2, x3], axis=1)
    #
    # x1 = jrn.uniform(jrn.PRNGKey(0), shape=(50, 1))
    # x2 = (-(5 / 7) * x1) - jrn.uniform(jrn.PRNGKey(0), shape=(50, 1)) - 0.0001
    # x3 = jnp.zeros((50, 1))
    #
    # x = jnp.concatenate([x1, x2, x3], axis=1)
    # train_data = jnp.concatenate([train_data, x], axis=0)
    train_data = jnp.asarray([])
    train_data = jrn.uniform(jrn.PRNGKey(0), shape=(number_of_samples, input_dimensions))
    output = jnp.asarray([])
    if output_dimensions == 1:
        output = jnp.round(jrn.uniform(jrn.PRNGKey(0), shape=(number_of_samples, output_dimensions)))
    else:
        output = jrn.uniform(jrn.PRNGKey(0), shape=(number_of_samples, output_dimensions))
        output = (output == output.max(axis=1)[:, None]).astype(float)

    train_data = jnp.concatenate([train_data, output], axis=1)


    # print(train_data)

    network.train(train_data, number_of_epochs)

    network.test(train_data)
    print(round(time() - start, 5))

    # jpr.stop_trace()

if __name__ == "__main__":
    main()
