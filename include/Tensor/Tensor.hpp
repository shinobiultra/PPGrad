#pragma once
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

#include "Tensor/TensorBase.hpp"

namespace PPGrad
{

    /// @brief Abstract Tensor class for all Eigen accelerated math on tensors.
    /// @details This class is a wrapper around Eigen::Matrix and Eigen::Tensor and is supposed to be immutable. Each operation instantiates a new Tensor object.s
    /// @tparam T The type of the underlying data (Eigen::Matrix or Eigen::Tensor).
    /// @tparam DT The type of the scalar data (i.e., individual elements of `data`) - float or double expected.
    template <int Dim, typename DT>
    class Tensor : public TensorBase<Dim, DT>
    {

    public:
        /// @brief Construct a new Tensor object, initialized with zeros.
        Tensor()
        {
            this->data = std::make_shared<Eigen::Tensor<DT, Dim>>(Eigen::Tensor<DT, Dim>());
            this->gradient = std::make_shared<Eigen::Tensor<DT, Dim>>(Eigen::Tensor<DT, Dim>());
            this->gradient->setZero();
        }

        /// @brief Construct a new Tensor object
        /// @param data The data to wrap.
        Tensor(std::shared_ptr<Eigen::Tensor<DT, Dim>> data)
        {
            this->data = data;
            this->gradient = std::make_shared<Eigen::Tensor<DT, Dim>>(Eigen::Tensor<DT, Dim>(data->dimensions()));
            this->gradient->setZero();
        }

        /// @brief Construct zero-initialized Tensor and allow enable/disable gradient accumulation.
        /// @param requiresGrad Whether or not to accumulate gradients for this tensor.
        Tensor(bool requiresGrad)
        {
            this->data = std::make_shared<Eigen::Tensor<DT, Dim>>(Eigen::Tensor<DT, Dim>());
            this->gradient = std::make_shared<Eigen::Tensor<DT, Dim>>(Eigen::Tensor<DT, Dim>());
            this->gradient->setZero();
            this->requiresGrad = requiresGrad;
        }

        /// @brief Construct a new Tensor object
        /// @param data The data to wrap.
        /// @param requiresGrad Whether or not to accumulate gradients for this tensor.
        Tensor(std::shared_ptr<Eigen::Tensor<DT, Dim>> data, bool requiresGrad)
        {
            this->data = data;
            this->gradient = std::make_shared<Eigen::Tensor<DT, Dim>>(Eigen::Tensor<DT, Dim>(data->dimensions()));
            this->gradient->setZero();
            this->requiresGrad = requiresGrad;
        }

        /// @brief Backward will simply do nothing for raw Tensors as they are leaf nodes in backpropagation.
        void _backward() override;

        /// @brief Get the underlying data of the tensor (of type T).
        /// @details Will probably not be implemented outside of debugging.
        /// @return The actual data.
        std::shared_ptr<Eigen::Tensor<DT, Dim>> getData() override;

        /// @brief Zero out the gradient of this tensor.
        void zeroGrad() override;

        /// @brief Accumulate the gradient of this tensor (intended to be called during backpropagation from nodes "upstream")
        void addGrad(std::shared_ptr<Eigen::Tensor<DT, Dim>> grad) override;
    };
}
