
/** @file */

#pragma once

#include "Tensor/TensorBase.hpp"
#include "Tensor/Tensor.hpp"
#include "NN/Model.hpp"
#include "NN/WeightInitializers.hpp"
#include "NN/Activations.hpp"
#include <vector>
#include <memory>

namespace PPNN
{

    /// @brief Dense layer (fully connected layer).
    /// @details Subclass of Model class, intended to be used as a fully connected layer in a neural network, i.e., have a set of parameters W and b, and perform the following operation: `y = Wx + b`.
    template <int Dim, typename DT>
    class Dense : public Model<Dim, DT>
    {
    private:
        /// @brief List of all [trainable] parameters of the model (in order: W, b)
        std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> params;

    protected:
        std::shared_ptr<PPGrad::TensorBase<Dim, DT>> W;
        std::shared_ptr<PPGrad::TensorBase<Dim, DT>> b;
        std::shared_ptr<PPNN::Activation<Dim, DT>> activation;


    public:
        /// @brief Constructor for Dense layer.
        /// @param inDim Dimension of input tensor.
        /// @param outDim Dimension of output tensor.
        Dense(
            int inDim,
            int outDim,
            WeightInititializers initializer = WeightInititializers::XAVIER,
            PPNN::Activations activation = PPNN::Activations::Linear)
        {
            W = PPGrad::Tensor<Dim, DT>::zeros({outDim, inDim}, true);
            b = PPGrad::Tensor<Dim, DT>::zeros({outDim, 1}, true);

            // Initialize the parameters
            WeightInitializer<Dim, DT>::init(W, initializer);
            WeightInitializer<Dim, DT>::init(b, WeightInititializers::ZEROS);

            params.push_back(W);
            params.push_back(b);

            if (activation == PPNN::Activations::ReLU)
            {
                this->activation = std::make_shared<PPNN::ReLU<Dim, DT>>();
            }
            else
            {
                this->activation = std::make_shared<PPNN::Linear<Dim, DT>>();
            }
        }

        /// @brief Forward function of the model, intended to produce prediction for batched input.
        /// @param inputs Batch (vector) of input tensors intended for model prediction.
        /// @return Batch (vector) of output tensors produced by the model.
        std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> forward(std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> inputs) override
        {
            std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> outputs;
            outputs.reserve(inputs.size());

            for (std::shared_ptr<PPGrad::TensorBase<Dim, DT>> input : inputs)
            {
                std::shared_ptr<PPGrad::TensorBase<Dim, DT>> output = this->activation->operator()((W * input) + b); // PPGrad magic!

                outputs.push_back(output);
            }
            return outputs;
        }

        /// @brief Forward function of the model, intended to produce prediction for single input.
        /// @param inputs Batch (vector) of input tensors intended for model prediction.
        /// @return Batch (vector) of output tensors produced by the model.
        std::shared_ptr<PPGrad::TensorBase<Dim, DT>> forward(std::shared_ptr<PPGrad::TensorBase<Dim, DT>> input) override
        {
            return this->activation->operator()((W * input) + b); // PPGrad magic!
        }

        /// @brief Return list of [trainable] parameters of the model (they shall contain the gradients after running backward() on each output produced by `forward()`).
        /// @return List of [trainable] parameters of the model.
        std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> &getParams() override
        {
            return this->params;
        }

        /// @brief Set list of [trainable] parameters of the model (intended to contain updated parameters after running optimizer).
        /// @param params List of (updated) [trainable] parameters of the model (in order: W, b).
        void setParams(std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> &params) override
        {
            W = params[0];
            b = params[1];
        }
    };

}