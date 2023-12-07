
/** @file */

#pragma once

#include "Tensor/TensorBase.hpp"
#include "NN/Model.hpp"
#include "NN/Optimizer.hpp"
#include "NN/Loss.hpp"
#include <vector>
#include <memory>
#include <iostream>

namespace PPNN
{

    template <int Dim, typename DT>
    class Trainer
    {
    private:
        std::shared_ptr<Model<Dim, DT>> model;
        std::shared_ptr<Optimizer<Dim, DT>> optimizer;
        std::shared_ptr<Loss<Dim, DT>> loss;
        std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> params;

    public:
        Trainer(std::shared_ptr<Model<Dim, DT>> model, std::shared_ptr<Optimizer<Dim, DT>> optimizer, std::shared_ptr<Loss<Dim, DT>> loss)
        {
            this->model = model;
            this->optimizer = optimizer;
            this->loss = loss;
            this->params = model->getParams(); // TODO: Return by reference
        }

        void train(std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> inputs,
                   std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> targets,
                   size_t epochs,
                   size_t batchSize,
                   bool verbose = false)
        {
            for (size_t epoch = 0; epoch < epochs; epoch++)
            {
                std::vector<DT> epochLosses;
                for (size_t batchStart = 0; batchStart < inputs.size(); batchStart += batchSize)
                {
                    std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> batchInputs;
                    std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> batchTargets;
                    for (size_t batchIdx = batchStart; batchIdx < batchStart + batchSize; batchIdx++)
                    {
                        batchInputs.push_back(inputs[batchIdx]);
                        batchTargets.push_back(targets[batchIdx]);
                    }

                    std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> batchPredictions = model->forward(batchInputs);
                    DT batchLoss = loss->operator()(batchPredictions, batchTargets, true);
                    epochLosses.push_back(batchLoss);

                    // Call backward on each output produced by forward() to accumulate gradients in the parameters.
                    for (std::shared_ptr<PPGrad::TensorBase<Dim, DT>>& prediction : batchPredictions)
                    {
                        PPGrad::TensorBase<Dim, DT>::backward(prediction);
                    }

                    optimizer->update(params);
                }

                if (verbose)
                {
                    std::cout << "Epoch: " << epoch << ", Loss: " << std::accumulate(epochLosses.begin(), epochLosses.end(), 0.0) / epochLosses.size() << std::endl;
                }
            }
        }
    };

} // namespace PPNN