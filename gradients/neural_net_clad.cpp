#include "matrixCU_clad.cu"
#include <iostream>
#include <time.h>

int input_dimensions = 2;
int output_dimensions = 1;
int N = 100; // nr of samples
int nr_epochs = 3;

__global__ void kTrain(Network *network, double *train_data, int nr_samples,
                       int data_sample_dim, int no_of_samples) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < no_of_samples) {
    network->train(train_data + i * (data_sample_dim), data_sample_dim);
  }
}

int main(int argc, char *argv[]) {
  if (argc == 5) {
    input_dimensions = atoi(argv[1]);
    output_dimensions = atoi(argv[2]);
    nr_epochs = atoi(argv[3]);
    N = atoi(argv[4]);
  }

  auto lossFunc_gradient = clad::gradient(lossFunc, "weights, input");
  auto activationFunc_gradient = clad::gradient(activationFunc);

  const int input_neurons = input_dimensions;
  const int output_neurons = output_dimensions;
  double learning_rate = 0.15;
  Network network(input_neurons, output_neurons, learning_rate);
  Network *network_device;
  cudaMalloc(&network_device, sizeof(Network));
  cudaMemcpy(network_device, &network, sizeof(Network), cudaMemcpyHostToDevice);

  int input_dim = input_neurons + 1; // the third one is the label
  const int nr_samples = N;
  int size = (input_neurons + output_neurons) * nr_samples * sizeof(double);
  int size_weights = input_dim * output_neurons * sizeof(double);

  int *input_dim_device, *nr_samples_device;
  cudaMalloc((void **)&input_dim_device, sizeof(int));
  cudaMalloc((void **)&nr_samples_device, sizeof(int));
  cudaMemcpy(input_dim_device, &input_dim, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(nr_samples_device, &nr_samples, sizeof(int),
             cudaMemcpyHostToDevice);

  double *train_data, *train_data_device, *trained_weights,
      *trained_weights_device;

  // Allocate memory on the CPU
  train_data = new double[size]();

  // Generate training_data points -- 5x_1 + 7x_2 = 0
  srand(3);
  int i = 0;
  //  while (i < nr_samples) {
  //    double x_1 = (double)rand() / RAND_MAX;
  //    train_data[i * input_dim] = x_1;
  //    if (i < nr_samples / 2) {
  //      train_data[i * input_dim + 1] =
  //          (-(5.0 / 7.0) * x_1) + ((double)rand() / RAND_MAX) + 0.0001;
  //      train_data[i * input_dim + 2] = 1.0;
  //
  //    } else {
  //      train_data[i * input_dim + 1] =
  //          (-(5.0 / 7.0) * x_1) - ((double)rand() / RAND_MAX) - 0.0001;
  //      train_data[i * input_dim + 2] = 0.0;
  //    }
  //    i++;
  //  }
  while (i < nr_samples) {
    for (int j = 0; j < input_neurons; j++) {
      double x = (double)rand() / RAND_MAX;
      train_data[i * (input_neurons + output_neurons) + j] = x;
    }

    if (output_neurons == 1) {
      int y = rand();
      if (y > RAND_MAX / 2)
        train_data[i * (input_neurons + output_neurons) + input_neurons] = 1;
    } else {
      int y = rand() % output_neurons;
      train_data[i * (input_neurons + output_neurons) + input_neurons + y] = 1;
    }
    i++;
  }
  printf("First input: %g %g\n\n", train_data[0], train_data[1]);
  //  for (int q = 0; q < nr_samples; q++) {
  //    for (int w = 0; w < input_neurons + output_neurons; w++)
  //      printf("%g\t", train_data[q*w]);
  //    printf("\n");
  //  }

  // Allocate memory on the GPU for training set
  cudaMalloc((void **)&train_data_device, size);
  cudaMemcpy(train_data_device, train_data, size, cudaMemcpyHostToDevice);

  // Initialise weights
  trained_weights = network.returnWeights();
  printf("inital weights are: %f \n", trained_weights[0]);

  for (int i = 0; i < nr_epochs; i++) {
    kTrain<<<(N + 255) / 256, (N > 256 ? 256 : N)>>>(
        network_device, train_data_device, nr_samples,
        input_neurons + output_neurons, N);
    cudaDeviceSynchronize();
//    if (i % 10 == 9)
      printf("%d iteration: weights are = %g \n", i + 1, trained_weights[0]);
  }

  //  cudaDeviceSynchronize();

  printf("Training over!\n");

  Network network_test(input_neurons, output_neurons, learning_rate,
                       trained_weights);
  network_test.test(train_data, nr_samples, input_neurons + output_neurons);

  printf("Testing over!\n");

  // Network *network_test_device;
  // cudaMallocManaged(&network_test_device,sizeof(Network));
  // *network_test_device = network_test;

  // kTest<<<(N+255)/256, 256>>>(network_test_device, train_data_device,
  // input_dim);
  return 0;
}