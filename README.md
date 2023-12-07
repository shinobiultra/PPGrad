# PPGrad
C++ &amp; OpenMPI &amp; OpenMP toy framework for Neural Nets (a la [Micrograd](https://github.com/karpathy/micrograd)) created in approx. 90 hours as a semestral project

# Initial Plan

The goal is to create distributed (OpenMPI) and parallelized (OpenMP) C++ framework (called "PPGrad") that would provide modular building blocks of Neural Networks (such as Conv2D Layer, Dense Layer, AdamW Optimizer, ReLU activation) and would allow to run the training in a Data distributed fashion.

We will use Eigen to utilize optimized matrix multiplications and we'll use OpenMP to paralellize run as many samples from the microbatch as possible.

The OpenMPI will then be used to provide the Data Parallel training.

The training loop provided by the framework should look something like:
1. Distribute the model to all the MPI machines
2. Scatter the batch so that each machine has part of the batch
3. Run the gradient calculations on each machine using OpenMP to parallelize each sample in the given microbatch
4. Synchronize the gradients using Allreduce so that all the machines can update the model.
5. Repeat.

We'd also like for the framework to use operator overloading to provide simple Autograd feature like PyTorch or Tensorflow (or Karpathy's "famous" Micrograd).

Stretch goal is to also support running the matrix multiplication on the GPU using custom CUDA kernels (to learn CUDA development and have some *fun* with it.)
