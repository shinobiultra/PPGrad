#pragma once

#include "Tensor/Tensor.hpp"


namespace PPGrad {
    template<int Dim, typename DT>
    class AddSTensor : public Tensor<Dim, DT> {


    protected:
        std::shared_ptr<TensorBase<Dim, DT>> inputA; ///< The inputs to the addition operation. Result stored in `data`.
        DT inputB; ///< The inputs to the addition operation. Result stored in `data`.

    public:

        /// @brief Construct a new AddTensor object from two tensors intended to be added.
        /// @param inputA The first input to the addition operation.
        /// @param inputB The second input to the addition operation.
        AddSTensor(std::shared_ptr<TensorBase<Dim, DT>> inputA, DT inputB)
        {
            this->inputA = inputA;
            this->inputB = inputB;
            this->data = std::make_shared<Eigen::Tensor<DT, Dim>>(*inputA->getData() + inputB);
        }

        /// @brief Construct a new AddTensor object from two tensors intended to be added & allow enable/disable gradient accumulation.
        /// @param inputA The first input to the addition operation.
        /// @param inputB The second input to the addition operation.
        /// @param requiresGrad Whether or not to accumulate gradients for this tensor.
        AddSTensor(std::shared_ptr<TensorBase<Dim, DT>> inputA, DT inputB, bool requiresGrad)
        {
            this->inputA = inputA;
            this->inputB = inputB;
            this->data = std::make_shared<Eigen::Tensor<DT, Dim>>(*inputA->getData() + inputB);
            this->requiresGrad = requiresGrad;
        }

        /// @brief Calculate the gradient of this tensor with respect to it's inputs.
        void backward() override;
    };
}