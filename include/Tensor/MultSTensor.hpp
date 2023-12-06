#pragma once

#include "Tensor/Tensor.hpp"

namespace PPGrad
{
    template <int Dim, typename DT>
    class MultSTensor : public Tensor<Dim, DT>
    {

    protected:
        std::shared_ptr<TensorBase<Dim, DT>> inputA; ///< The inputs to the multiplication operation. Result stored in `data`.
        DT inputB;                                   ///< The inputs to the multiplication operation. Result stored in `data`.

    public:
        /// @brief Construct a new MultTensor object from two tensors intended to be multiplied.
        /// @param inputA The first input to the multiplication operation.
        /// @param inputB The second input to the multiplication operation.
        MultSTensor(std::shared_ptr<TensorBase<Dim, DT>> inputA, DT inputB)
        {
            this->inputA = inputA;
            this->inputB = inputB;
            this->data = std::make_shared<Eigen::Tensor<DT, Dim>>(*inputA->getData() * inputB);
            this->gradient = std::make_shared<Eigen::Tensor<DT, Dim>>(Eigen::Tensor<DT, Dim>(inputA->getData()->dimensions()));
            this->gradient->setZero();
        }

        /// @brief Construct a new MultTensor object from two tensors intended to be multiplied & allow enable/disable gradient accumulation.
        /// @param inputA The first input to the multiplication operation.
        /// @param inputB The second input to the multiplication operation.
        /// @param requiresGrad Whether or not to accumulate gradients for this tensor.
        MultSTensor(std::shared_ptr<TensorBase<Dim, DT>> inputA, DT inputB, bool requiresGrad)
        {
            this->inputA = inputA;
            this->inputB = inputB;
            this->data = std::make_shared<Eigen::Tensor<DT, Dim>>(*inputA->getData() * inputB);
            this->gradient = std::make_shared<Eigen::Tensor<DT, Dim>>(Eigen::Tensor<DT, Dim>(inputA->getData()->dimensions()));
            this->gradient->setZero();
            this->requiresGrad = requiresGrad;
        }

        /// @brief Calculate the gradient of this tensor with respect to it's inputs.
        void _backward() override;
    };
}