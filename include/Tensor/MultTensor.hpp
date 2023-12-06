
/** @file */


#pragma once

#include "Tensor/Tensor.hpp"

namespace PPGrad
{
    template <int Dim, typename DT>
    class MultTensor : public Tensor<Dim, DT>
    {

    protected:
        std::shared_ptr<TensorBase<Dim, DT>> inputA, inputB; ///< The inputs to the addition operation. Result stored in `data`.

    public:
        /// @brief Construct a new AddTensor object from two tensors intended to be added.
        /// @param inputA The first input to the addition operation.
        /// @param inputB The second input to the addition operation.
        MultTensor(std::shared_ptr<TensorBase<Dim, DT>> inputA, std::shared_ptr<TensorBase<Dim, DT>> inputB)
        {
            this->inputA = inputA;
            this->inputB = inputB;
            Eigen::array<Eigen::IndexPair<int>, 1> contractionPair = {Eigen::IndexPair<int>(Dim - 1, 0)};
            Eigen::Tensor<DT, 2 * (Dim - 1)> contractionResult = inputA->getData()->contract(*inputB->getData(), contractionPair);
            this->data = std::make_shared<Eigen::Tensor<DT, Dim>>(contractionResult);
        }

        /// @brief Construct a new AddTensor object from two tensors intended to be added & allow enable/disable gradient accumulation.
        /// @param inputA The first input to the addition operation.
        /// @param inputB The second input to the addition operation.
        /// @param requiresGrad Whether or not to accumulate gradients for this tensor.
        MultTensor(std::shared_ptr<TensorBase<Dim, DT>> inputA, std::shared_ptr<TensorBase<Dim, DT>> inputB, bool requiresGrad)
        {
            this->inputA = inputA;
            this->inputB = inputB;
            Eigen::array<Eigen::IndexPair<int>, 1> contractionPair = {Eigen::IndexPair<int>(Dim - 1, 0)};
            Eigen::Tensor<DT, 2 * (Dim - 1)> contractionResult = inputA->getData()->contract(*inputB->getData(), contractionPair);
            this->data = std::make_shared<Eigen::Tensor<DT, Dim>>(contractionResult);
            this->gradient = std::make_shared<Eigen::Tensor<DT, Dim>>(Eigen::Tensor<DT, Dim>(inputA->getData()->dimensions()[0], inputB->getData()->dimensions()[1]));
            this->gradient->setZero();
            this->requiresGrad = requiresGrad;
        }

        std::vector <std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> getParents() override;

        /// @brief Calculate the gradient of this tensor with respect to it's inputs.
        void _backward() override;
    };
}