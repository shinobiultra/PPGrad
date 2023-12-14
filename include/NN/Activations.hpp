
/** @file */

#pragma once

#include "Tensor/TensorBase.hpp"
#include "Tensor/ReLUTensor.hpp"
#include <vector>
#include <memory>

namespace PPNN
{

    enum class Activations
    {
        Linear,
        ReLU,        
    };

    /// @brief Base class for all activation functions in PPNN library.
    /// @tparam DT DType of the expected Tensors.
    /// @tparam Dim Dimension of the expected Tensors.
    template <int Dim, typename DT>
    class Activation
    {
    public:
        /// @brief Calculate the activation of the input Tensor.
        /// @param input Input Tensor.
        /// @return Activation of the input Tensor.
        virtual std::shared_ptr<PPGrad::TensorBase<Dim, DT>> operator()(std::shared_ptr<PPGrad::TensorBase<Dim, DT>> input) = 0;

        
    };

    /// @brief Linear activation function.
    /// @tparam DT DType of the expected Tensors.
    /// @tparam Dim Dimension of the expected Tensors.
    template <int Dim, typename DT>
    class Linear : public Activation<Dim, DT>
    {
    public:
        std::shared_ptr<PPGrad::TensorBase<Dim, DT>> operator()(std::shared_ptr<PPGrad::TensorBase<Dim, DT>> input) override
        {
            return input;
        }
    };

    /// @brief ReLU activation function.
    /// @tparam DT DType of the expected Tensors.
    /// @tparam Dim Dimension of the expected Tensors.
    template <int Dim, typename DT>
    class ReLU : public Activation<Dim, DT>
    {
    public:
        std::shared_ptr<PPGrad::TensorBase<Dim, DT>> operator()(std::shared_ptr<PPGrad::TensorBase<Dim, DT>> input) override
        {
            return std::make_shared<PPGrad::ReLUTensor<Dim, DT>>(input, input->getRequiresGrad());
        }
    };

    

}