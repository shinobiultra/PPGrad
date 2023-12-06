
/** @file */


#pragma once

#include "Tensor/Tensor.hpp"

namespace PPGrad
{
    template <int Dim, typename DT>
    class DivSTensor : public Tensor<Dim, DT>
    {

    protected:
        std::shared_ptr<TensorBase<Dim, DT>> inputA; ///< The inputs to the division operation. Result stored in `data`.
        DT inputB;                                   ///< The inputs to the division operation. Result stored in `data`.

    public:
        /// @brief Construct a new DivTensor object from two tensors intended to be divided.
        /// @param inputA The first input to the division operation.
        /// @param inputB The second input to the division operation.
        DivSTensor(std::shared_ptr<TensorBase<Dim, DT>> inputA, DT inputB)
        {
            this->inputA = inputA;
            this->inputB = inputB;
            this->data = std::make_shared<Eigen::Tensor<DT, Dim>>(*inputA->getData() / inputB);
        }

        /// @brief Construct a new DivTensor object from two tensors intended to be divided & allow enable/disable gradient accumulation.
        /// @param inputA The first input to the division operation.
        /// @param inputB The second input to the division operation.
        /// @param requiresGrad Whether or not to accumulate gradients for this tensor.
        DivSTensor(std::shared_ptr<TensorBase<Dim, DT>> inputA, DT inputB, bool requiresGrad)
        {
            this->inputA = inputA;
            this->inputB = inputB;
            this->data = std::make_shared<Eigen::Tensor<DT, Dim>>(*inputA->getData() / inputB);
            this->gradient = std::make_shared<Eigen::Tensor<DT, Dim>>(Eigen::Tensor<DT, Dim>(inputA->getData()->dimensions()));
            this->gradient->setZero();
            this->requiresGrad = requiresGrad;
        }

        std::vector <std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> getParents() override;

        /// @brief Calculate the gradient of this tensor with respect to it's inputs.
        void _backward() override;
    };
}