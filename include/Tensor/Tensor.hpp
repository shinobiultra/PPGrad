
/** @file */

#pragma once
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

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
        }

        /// @brief Construct a new Tensor object
        /// @param data The data to wrap.
        Tensor(std::shared_ptr<Eigen::Tensor<DT, Dim>> data)
        {
            this->data = data;
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

        /// @brief Raw Tensors are leaf nodes of the computational graph, thus have no parents.
        virtual std::vector <std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> getParents() override
        {
            return std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>>();
        }


        // ------- Static methods ------- //

        // ------------ Tensor generation ------------ //
    
        /// @brief Generate instance of `Tensor` zero initialized. Note that the Tensors do not require gradient by default!
        /// @tparam DT Datatype of the underlying Eigen Tensor
        /// @tparam Dim Number of dimensions of the Tensor
        /// @param shape Shape of the Tensor
        /// @return Shared pointer to the zero-initialized Tensor
        static std::shared_ptr<TensorBase<Dim, DT>> zeros(std::array<int, Dim> shape);

        /// @brief Generate instance of `Tensor` zero initialized
        /// @tparam DT Datatype of the underlying Eigen Tensor
        /// @tparam Dim Number of dimensions of the Tensor
        /// @param shape Shape of the Tensor
        /// @param requiresGrad Whether or not to accumulate gradients for this tensor.
        /// @return Shared pointer to the zero-initialized Tensor
        static std::shared_ptr<TensorBase<Dim, DT>> zeros(std::array<int, Dim> shape, bool requiresGrad);

        /// @brief Generate instance of `Tensor` with random values sampled from normalized Normal distribution. Note that the Tensors do not require gradient by default!
        /// @tparam DT Datatype of the underlying Eigen Tensor
        /// @tparam Dim Number of dimensions of the Tensor
        /// @param shape Shape of the Tensor
        /// @return Shared pointer to the random Tensor
        static std::shared_ptr<TensorBase<Dim, DT>> randn(std::array<int, Dim> shape);

        /// @brief Generate instance of `Tensor` with random values sampled from normalized Normal distribution
        /// @tparam DT Datatype of the underlying Eigen Tensor
        /// @tparam Dim Number of dimensions of the Tensor
        /// @param shape Shape of the Tensor
        /// @param requiresGrad Whether or not to accumulate gradients for this tensor.
        /// @return Shared pointer to the random Tensor
        static std::shared_ptr<TensorBase<Dim, DT>> randn(std::array<int, Dim> shape, bool requiresGrad);

        /// @brief Generate instance of `Tensor` with random values from parametrized Normal Distribution. Note that the Tensors do not require gradient by default!
        /// @tparam DT Datatype of the underlying Eigen Tensor
        /// @tparam Dim Number of dimensions of the Tensor
        /// @param shape Shape of the Tensor
        /// @param mean Mean of the Normal distribution
        /// @param stddev Standard deviation of the Normal distribution
        /// @return Shared pointer to the random Tensor
        static std::shared_ptr<TensorBase<Dim, DT>> randn(std::array<int, Dim> shape, DT mean, DT stddev);

        /// @brief Generate instance of `Tensor` with random values from parametrized Normal Distribution
        /// @tparam DT Datatype of the underlying Eigen Tensor
        /// @tparam Dim Number of dimensions of the Tensor
        /// @param shape Shape of the Tensor
        /// @param mean Mean of the Normal distribution
        /// @param stddev Standard deviation of the Normal distribution
        /// @param requiresGrad Whether or not to accumulate gradients for this tensor.
        /// @return Shared pointer to the random Tensor
        static std::shared_ptr<TensorBase<Dim, DT>> randn(std::array<int, Dim> shape, DT mean, DT stddev, bool requiresGrad);

        /// @brief Generate instance of `Tensor` with 1s on the generalized diagonal (i.e., where all indices equal) and 0s elsewhere. Note that the Tensors do not require gradient by default!
        /// @tparam DT Datatype of the underlying Eigen Tensor
        /// @tparam Dim Number of dimensions of the Tensor
        /// @param shape Shape of the Tensor
        /// @return Shared pointer to the random Tensor
        static std::shared_ptr<TensorBase<Dim, DT>> eye(std::array<int, Dim> shape);

        /// @brief Generate instance of `Tensor` with 1s on the generalized diagonal (i.e., where all indices equal) and 0s elsewhere
        /// @tparam DT Datatype of the underlying Eigen Tensor
        /// @tparam Dim Number of dimensions of the Tensor
        /// @param shape Shape of the Tensor
        /// @param requiresGrad Whether or not to accumulate gradients for this tensor.
        /// @return Shared pointer to the random Tensor
        static std::shared_ptr<TensorBase<Dim, DT>> eye(std::array<int, Dim> shape, bool requiresGrad);


    };
}
