
/** @file */

#pragma once
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include "Tensor/Tensor.hpp"

namespace PPGrad
{

    /// @brief Tensor that get's returned when adding two tensors.
    /// @tparam T The type of the underlying data (Eigen::Matrix or Eigen::Tensor).
    /// @tparam DT The type of the scalar data (i.e., individual elements of `data`) - float or double expected.
    template <int Dim, typename DT>
    class AddTensor : public Tensor<Dim, DT>
    {

    protected:
        std::shared_ptr<TensorBase<Dim, DT>> inputA, inputB; ///< The inputs to the addition operation. Result stored in `data`.

    public:
        /// @brief Construct a new AddTensor object from two tensors intended to be added.
        /// @param inputA The first input to the addition operation.
        /// @param inputB The second input to the addition operation.
        AddTensor(std::shared_ptr<TensorBase<Dim, DT>> inputA, std::shared_ptr<TensorBase<Dim, DT>> inputB)
        {
            this->inputA = inputA;
            this->inputB = inputB;
            this->data = std::make_shared<Eigen::Tensor<DT, Dim>>(*inputA->getData() + *inputB->getData());
        }

        /// @brief Construct a new AddTensor object from two tensors intended to be added & allow enable/disable gradient accumulation.
        /// @param inputA The first input to the addition operation.
        /// @param inputB The second input to the addition operation.
        /// @param requiresGrad Whether or not to accumulate gradients for this tensor.
        AddTensor(std::shared_ptr<TensorBase<Dim, DT>> inputA, std::shared_ptr<TensorBase<Dim, DT>> inputB, bool requiresGrad)
        {
            this->inputA = inputA;
            this->inputB = inputB;
            this->data = std::make_shared<Eigen::Tensor<DT, Dim>>(*inputA->getData() + *inputB->getData());
            this->gradient = std::make_shared<Eigen::Tensor<DT, Dim>>(inputA->getData()->dimensions());
            this->gradient->setZero();
            this->requiresGrad = requiresGrad;
        }

        /// @brief Calculate the gradient of this tensor with respect to it's inputs.
        void _backward() override;

        std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> getParents() override;
    };
}
