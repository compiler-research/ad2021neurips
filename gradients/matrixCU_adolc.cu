#ifndef __MATRIXCUDAADOLC_H__
#define __MATRIXCUDAADOLC_H__
#include <adolc/adoublecuda.h>
#include <assert.h>
#include <cmath>
#include <iostream>
#include <vector>
using namespace std;

// define network activation; can be Liniar/Step/Ramp/Hyperbolic/Tangent,
// Gaussian Function, here
template <typename T> inline __device__ T activationFunc(T x) {
  return (1.0 / (1.0 + exp(-x)));
}

template <typename T> inline __host__ T activationFunc(T x) {
  return (1.0 / (1.0 + exp(-x)));
}

template <typename T>
inline __host__ __device__ T neuronOutput(T *weight, T *input, int input_dim) {
  T neuron_output = 0.0;
  for (int i = 0; i < input_dim; i++) {
    neuron_output = neuron_output + weight[i] * input[i];
  }
  neuron_output = neuron_output + weight[input_dim]; // account for the bias
  return neuron_output;
}

// most famous cost functions are: Quadratic Cost (Root Mean Square), Cross
// Entropy, Exponential
template <typename T>
inline __device__ T lossFunc(double expected_output, T *weights, double *input,
                             int input_dim) {
  T neuron_output = 0.0;
  for (int i = 0; i < input_dim; i++)
    neuron_output += weights[i] * input[i];
  neuron_output += weights[input_dim]; // account for the bias

  neuron_output = (1.0 / (1.0 + exp(-1 * neuron_output)));

  T loss = -1 * (expected_output * log(neuron_output)) +
           (1 - expected_output) * log(1 - neuron_output);
  return loss;
}

class Network {
private:
  // nr of input/output sneurons
  std::size_t input_dim, output_dim, sizeArray;
  double lr; // learning rate
  double *output_values;
  double *input_values;
  bool are_weights_allocated = false;

public:
  double *weights;
  Network()
      : input_dim(2), output_dim(1), sizeArray((input_dim + 1) * output_dim),
        lr(0.15) {
    cudaError_t err = cudaMallocManaged(&weights, sizeArray * sizeof(double));
    if (err != cudaSuccess)
      printf("1-Memory allocation failed: %d\n", err);
    are_weights_allocated = true;
  }

  Network(const size_t input_neurons, const size_t output_neurons,
          double learning_rate)
      : input_dim(input_neurons), output_dim(output_neurons),
        sizeArray((input_dim + 1) * output_dim), lr(learning_rate) {
    cudaError_t err = cudaMallocManaged(&weights, sizeArray * sizeof(double));
    if (err != cudaSuccess)
      printf("2-Memory allocation failed: %d\n", err);
//    srand(time(0));
    size_t i, j, l;
    for (i = 0; i < output_dim; i++) {
      // the extra weight needs to account for the bias
      for (j = 0; j < input_dim + 1; j++) {
        l = i * (input_dim + 1) + j;
        weights[l] = ((double)rand() / RAND_MAX);
      }
    }
    are_weights_allocated = true;
  }

  Network(size_t input_neurons, size_t output_neurons, double learning_rate,
          double *trained_weights)
      : input_dim(input_neurons), output_dim(output_neurons),
        sizeArray((input_dim + 1) * output_dim), lr(learning_rate) {
    weights = trained_weights;
    are_weights_allocated = false;
  }

  ~Network() {
    if (are_weights_allocated)
      cudaFree(weights);
  }

  __host__ __device__ double *feedForward(double *input, int input_size) {
    assert(input_size == input_dim);
    output_values = new double[output_dim];
    // double input_vals[input_values.size()];
    // std::copy(input_values.begin(), input_values.end(), input_vals);

    size_t i, j, l;
    for (i = 0; i < output_dim; i++) {
      // the extra weight needs to account for the bias
      double *weight = new double[input_dim + 1];
      for (j = 0; j < input_dim + 1; j++) {
        l = i * (input_dim + 1) + j;
        weight[j] = weights[l];
      }
      output_values[i] =
          activationFunc(neuronOutput(weight, input, input_dim));
    }
    return output_values;
  }

  __device__ void backPropagate(double *input, int inp_dim, double *exp_values, int exp_dim) {
    assert(inp_dim == input_dim);
    assert(exp_dim == output_dim);
    // use the direction of the steepest descent, learning rate and activation
    // function to update $
    size_t out, j, l;
    for (out = 0; out < exp_dim; out++) {
      adtlc::adouble output_value, activation_value;
      output_value = output_values[out];
      output_value.setADValue(1);

      activation_value = activationFunc(output_value);

      double activation_backprop = activation_value.getADValue();

      adtlc::adouble *weight = new adtlc::adouble[input_dim + 1], loss;
      for (j = 0; j < input_dim + 1; j++) {
        l = out * (input_dim + 1) + j;
        weight[j] = weights[l];
      }
      double *network_backprop =  new double [input_dim + 1]();

      for (int w = 0; w < input_dim + 1; w++) {
        weight[w].setADValue(1);

        loss = lossFunc(exp_values[out], weight, input, input_dim);
        network_backprop[w] = loss.getADValue();

        weights[out * (input_dim + 1) + w] -=
            (lr * network_backprop[w] * activation_backprop);
        weight[w].setADValue(0);
      }
//      printf("i: %g %g; w: %g %g %g\n", input[0], input[1], network_backprop[0], network_backprop[1], network_backprop[2]);
    }
  }

  __host__ __device__ double *train(double *train_data, int data_sample_dim) {
    assert(data_sample_dim == input_dim + output_dim);
    feedForward(train_data, input_dim);
    backPropagate(train_data, input_dim, train_data + input_dim, output_dim);
    return weights;
  }

  __host__ __device__ void test(double *train_data, int nr_samples,
                                int sample_input_dim) {
    assert(sample_input_dim == input_dim + output_dim);
    // printf("prior weights: %f \n", weights[0]);
    // printf("prior weights: %f \n", trained_weights[0]);
    // assert(sizeof(weights) == sizeof(trained_weights));
    // weights = trained_weights;
    printf("after weights: %f \n", weights[0]);
    double acc = 0.0;
    for (int i = 0; i < nr_samples; i++) {
      feedForward(train_data + i * sample_input_dim, input_dim);
      if (output_dim == 1) {
        if (output_values[0] > 0.5 &&
            train_data[i * sample_input_dim + input_dim] == 1.0)
          acc++;
        else if (output_values[0] < 0.5 &&
                 train_data[i * sample_input_dim + input_dim] == 0.0)
          acc++;
        printf("%0.6g\n", output_values[0]);
      } else {
        int max_output_idx = 0, max_data_idx = 0;
        for (int j = 1; j < output_dim; j++) {
          if (output_values[max_output_idx] < output_values[j])
            max_output_idx = j;
          if (train_data[i * sample_input_dim + input_dim + max_data_idx] < train_data[i * sample_input_dim + input_dim + j])
            max_data_idx = j;
        }
        acc += (max_output_idx == max_data_idx);
      }
    }
    printf("Accuracy: %f \n", acc / nr_samples);
  }

  // __host__ __device__ void assignValue(size_t l, double value){
  //     double activation_backprop[1] = {};
  //     activationFunc_grad(2.0, activation_backprop);
  //     // af_backpropagate.execute(2.0, activation_backprop);

  //     // weights[l] = activation_backprop[0];
  //     weights[l] = value;

  // }

  __host__ __device__ double *returnWeights() { return weights; }

  __host__ __device__ void displayArray() {
    size_t i, j, l;
    for (i = 0; i < (input_dim + 1); ++i) {
      for (j = 0; j < output_dim; ++j) {
        l = i * output_dim + j;
        printf("%f\t", weights[l]);
      }
      // cout<<endl;
      printf("\n");
    }
  }
};
#endif