
/** @file */

#pragma once

#include "Tensor/TensorBase.hpp"
#include "NN/Model.hpp"
#include "NN/Optimizer.hpp"
#include "NN/Loss.hpp"
#include <vector>
#include <memory>
#include <iostream>
#include <omp.h>
#include <mpi.h>
#include <algorithm>

namespace PPNN
{

    /// @brief Distributed (OpenMPI) and Parallel (OpenMP) Trainer for PPNN neural network model with given optimizer and loss function.
    /// @tparam DT Data type of the model.
    /// @tparam Dim Dimension of the model's tensors.
    template <int Dim, typename DT>
    class DPTrainer
    {
    private:
        std::shared_ptr<Model<Dim, DT>> model;
        std::shared_ptr<Optimizer<Dim, DT>> optimizer;
        std::shared_ptr<Loss<Dim, DT>> loss;
        std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> params;

        int32_t gradSyncFreq;
        bool gradientAccumulation = false;

    public:
        DPTrainer(
            std::shared_ptr<Model<Dim, DT>> model,
            std::shared_ptr<Optimizer<Dim, DT>> optimizer,
            std::shared_ptr<Loss<Dim, DT>> loss,
            int32_t gradSyncFreq = 16, // very conservative default value (see DiLoCo paper)
            bool gradientAccumulation = false)
        {
            this->model = model;
            this->optimizer = optimizer;
            this->loss = loss;
            this->params = model->getParams(); // TODO: Return by reference
            this->gradSyncFreq = gradSyncFreq;
            this->gradientAccumulation = gradientAccumulation;
        }

        void train(std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> inputs,
                   std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> targets,
                   size_t epochs,
                   size_t batchSize,
                   bool verbose = false)
        {

            int worldSize, worldRank;
            MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
            MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

            const size_t localDataStart = 0;
            const size_t localDataEnd = inputs.size();

            if (verbose)
            {
                std::cout << "[Rank: " << worldRank << "] "
                          << "Training on " << localDataEnd - localDataStart << " samples." << std::endl;
            }

            // Train the model
            int32_t gradSyncCounter = 0;
            for (size_t epoch = 0; epoch < epochs; epoch++)
            {
                std::vector<DT> epochLosses;
                for (size_t batchStart = localDataStart; batchStart < localDataEnd; batchStart += batchSize)
                {

                    std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> batchPredictions(batchSize);

#pragma omp parallel for default(shared)
                    for (size_t batchIdx = batchStart; batchIdx < batchStart + batchSize; batchIdx++)
                    {
                        batchPredictions[batchIdx - batchStart] = model->forward(inputs[batchIdx]);
                    }

                    std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> batchTargets;
                    for (size_t batchIdx = batchStart; batchIdx < batchStart + batchSize; batchIdx++)
                    {
                        batchTargets.push_back(targets[batchIdx]);
                    }

                    DT batchLoss = loss->operator()(batchPredictions, batchTargets, true);
                    epochLosses.push_back(batchLoss);

// Call backward on each output produced by forward() to accumulate gradients in the parameters.
#pragma omp parallel for default(shared)
                    for (std::shared_ptr<PPGrad::TensorBase<Dim, DT>> &prediction : batchPredictions)
                    {
                        PPGrad::TensorBase<Dim, DT>::backward(prediction);
                    }

                    // Allreduce the gradients across all processes (MPI_Allreduce) every Nth batch.
                    if (++gradSyncCounter == gradSyncFreq)
                    {
                        if (verbose)
                        {
                            std::cout << "[Rank: " << worldRank << "] "
                                      << " Syncing " << params.size() << " parameters." << std::endl;
                        }

                        for (std::shared_ptr<PPGrad::TensorBase<Dim, DT>> &param : params)
                        {
                            MPI_Allreduce(MPI_IN_PLACE, param->getGrad()->data(), param->getGrad()->size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                            *param->getGrad() = *param->getGrad() / (DT)worldSize;

                            if (this->gradientAccumulation)
                                *param->getGrad() = *param->getGrad() / (DT)gradSyncFreq;
                        }
                        gradSyncCounter = 0;

                        if (this->gradientAccumulation)
                        {
                            // Update the parameters
                            optimizer->update(params);
                        }
                    }

                    if (!this->gradientAccumulation)
                    {
                        // Update the parameters
                        optimizer->update(params);
                    }
                }

                if (verbose)
                {
                    std::cout << "[Rank: " << worldRank << "] "
                              << "Epoch: " << epoch << ", Loss: " << std::accumulate(epochLosses.begin(), epochLosses.end(), 0.0) / epochLosses.size() << std::endl;
                }
            }

            // Last sync
            for (std::shared_ptr<PPGrad::TensorBase<Dim, DT>> &param : params)
            {
                MPI_Allreduce(MPI_IN_PLACE, param->getGrad()->data(), param->getGrad()->size(), MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
                *param->getGrad() = *param->getGrad() / (DT)worldSize;
            }
            optimizer->update(params);

#ifdef PPGRAD_DEBUG
            // print 10 params from each parameter tensor
            for (std::shared_ptr<PPGrad::TensorBase<Dim, DT>> &param : params)
            {
                for (size_t i = 0; i < 10; i++)
                {
                    std::cout << "[Rank: " << worldRank << "] param " << i << ": " << (param->getData()->data())[i] << std::endl;
                }
            }
#endif

            // Finalize the training across nodes
            MPI_Barrier(MPI_COMM_WORLD);
        }
    };

} // namespace PPNN