
/** @file */

#include "Tensor/Tensor.hpp"
#include "Tensor/AddTensor.hpp"
#include "Tensor/SubTensor.hpp"
#include "Tensor/MultTensor.hpp"
#include "Tensor/AddSTensor.hpp"
#include "Tensor/SubSTensor.hpp"
#include "Tensor/MultSTensor.hpp"
#include "Tensor/DivSTensor.hpp"

#include <memory>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

namespace PPGrad
{


    /// @brief Generate instance of `Tensor` zero initialized
    /// @tparam DT Datatype of the underlying Eigen Tensor
    /// @tparam Dim Number of dimensions of the Tensor
    /// @param shape Shape of the Tensor
    /// @return Shared pointer to the zero-initialized Tensor
    template <int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> Tensor<Dim, DT>::zeros(std::array<int, Dim> shape) {
        // convert shape to Dimensions object
        Eigen::array<Eigen::Index, Dim> dimensions;
        for (int i = 0; i < Dim; i++) {
            dimensions[i] = shape[i];
        }

        Eigen::Tensor<DT, Dim> data = Eigen::Tensor<DT, Dim>(dimensions);
        data.setZero();
        return std::make_shared<Tensor<Dim, DT>>(std::make_shared<Eigen::Tensor<DT, Dim>>(data));
    }

    /// @brief Generate instance of `Tensor` zero initialized
    /// @tparam DT Datatype of the underlying Eigen Tensor
    /// @tparam Dim Number of dimensions of the Tensor
    /// @param shape Shape of the Tensor
    /// @param requiresGrad Whether or not to accumulate gradients for this tensor.
    /// @return Shared pointer to the zero-initialized Tensor
    template <int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> Tensor<Dim, DT>::zeros(std::array<int, Dim> shape, bool requiresGrad) {
        // convert shape to Dimensions object
        Eigen::array<Eigen::Index, Dim> dimensions;
        for (int i = 0; i < Dim; i++) {
            dimensions[i] = shape[i];
        }

        Eigen::Tensor<DT, Dim> data = Eigen::Tensor<DT, Dim>(dimensions);
        data.setZero();
        return std::make_shared<Tensor<Dim, DT>>(std::make_shared<Eigen::Tensor<DT, Dim>>(data), requiresGrad);
    }

    /// @brief Generate instance of `Tensor` with random values sampled from normalized Normal distribution
    /// @tparam DT Datatype of the underlying Eigen Tensor
    /// @tparam Dim Number of dimensions of the Tensor
    /// @param shape Shape of the Tensor
    /// @return Shared pointer to the random Tensor
    template <int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> Tensor<Dim, DT>::randn(std::array<int, Dim> shape) {
        // convert shape to Dimensions object
        Eigen::array<Eigen::Index, Dim> dimensions;
        for (int i = 0; i < Dim; i++) {
            dimensions[i] = shape[i];
        }

        Eigen::Tensor<DT, Dim> data = Eigen::Tensor<DT, Dim>(dimensions);
        data.setRandom();
        return std::make_shared<Tensor<Dim, DT>>(std::make_shared<Eigen::Tensor<DT, Dim>>(data));
    }

    /// @brief Generate instance of `Tensor` with random values sampled from normalized Normal distribution
    /// @tparam DT Datatype of the underlying Eigen Tensor
    /// @tparam Dim Number of dimensions of the Tensor
    /// @param shape Shape of the Tensor
    /// @param requiresGrad Whether or not to accumulate gradients for this tensor.
    /// @return Shared pointer to the random Tensor
    template <int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> Tensor<Dim, DT>::randn(std::array<int, Dim> shape, bool requiresGrad) {
        // convert shape to Dimensions object
        Eigen::array<Eigen::Index, Dim> dimensions;
        for (int i = 0; i < Dim; i++) {
            dimensions[i] = shape[i];
        }

        Eigen::Tensor<DT, Dim> data = Eigen::Tensor<DT, Dim>(dimensions);
        data.setRandom();
        return std::make_shared<Tensor<Dim, DT>>(std::make_shared<Eigen::Tensor<DT, Dim>>(data), requiresGrad);
    }

    /// @brief Generate instance of `Tensor` with random values from parametrized Normal Distribution
    /// @tparam DT Datatype of the underlying Eigen Tensor
    /// @tparam Dim Number of dimensions of the Tensor
    /// @param shape Shape of the Tensor
    /// @param mean Mean of the Normal distribution
    /// @param stddev Standard deviation of the Normal distribution
    /// @return Shared pointer to the random Tensor
    template <int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> Tensor<Dim, DT>::randn(std::array<int, Dim> shape, DT mean, DT stddev) {
        // convert shape to Dimensions object
        Eigen::array<Eigen::Index, Dim> dimensions;
        for (int i = 0; i < Dim; i++) {
            dimensions[i] = shape[i];
        }

        Eigen::Tensor<DT, Dim> data = Eigen::Tensor<DT, Dim>(dimensions);
        data.setRandom();
        data = data * stddev + mean;
        return std::make_shared<Tensor<Dim, DT>>(std::make_shared<Eigen::Tensor<DT, Dim>>(data));
    }

    /// @brief Generate instance of `Tensor` with random values from parametrized Normal Distribution
    /// @tparam DT Datatype of the underlying Eigen Tensor
    /// @tparam Dim Number of dimensions of the Tensor
    /// @param shape Shape of the Tensor
    /// @param mean Mean of the Normal distribution
    /// @param stddev Standard deviation of the Normal distribution
    /// @param requiresGrad Whether or not to accumulate gradients for this tensor.
    /// @return Shared pointer to the random Tensor
    template <int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> Tensor<Dim, DT>::randn(std::array<int, Dim> shape, DT mean, DT stddev, bool requiresGrad) {
        // convert shape to Dimensions object
        Eigen::array<Eigen::Index, Dim> dimensions;
        for (int i = 0; i < Dim; i++) {
            dimensions[i] = shape[i];
        }

        Eigen::Tensor<DT, Dim> data = Eigen::Tensor<DT, Dim>(dimensions);
        data.setRandom();
        data = data * stddev + mean;
        return std::make_shared<Tensor<Dim, DT>>(std::make_shared<Eigen::Tensor<DT, Dim>>(data), requiresGrad);
    }

    /// @brief Generate instance of `Tensor` with 1s on the generalized diagonal (i.e., where all indices equal) and 0s elsewhere
    /// @tparam DT Datatype of the underlying Eigen Tensor
    /// @tparam Dim Number of dimensions of the Tensor
    /// @param shape Shape of the Tensor
    /// @return Shared pointer to the random Tensor
    template <int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> Tensor<Dim, DT>::eye(std::array<int, Dim> shape) {
        // convert shape to Dimensions object
        Eigen::array<Eigen::Index, Dim> dimensions;
        for (int i = 0; i < Dim; i++) {
            dimensions[i] = shape[i];
        }

        Eigen::Tensor<DT, Dim> data = Eigen::Tensor<DT, Dim>(dimensions);
        data.setZero();
        Eigen::array<int, Dim> indices;
        
        for (int i = 0; i < Dim; i++) {
            for (int j = 0; j < Dim; j++) {
                indices[j] = i;
            }
            data(indices) = 1;
        }

        return std::make_shared<Tensor<Dim, DT>>(std::make_shared<Eigen::Tensor<DT, Dim>>(data));
    }

    /// @brief Generate instance of `Tensor` with 1s on the generalized diagonal (i.e., where all indices equal) and 0s elsewhere
    /// @tparam DT Datatype of the underlying Eigen Tensor
    /// @tparam Dim Number of dimensions of the Tensor
    /// @param shape Shape of the Tensor
    /// @param requiresGrad Whether or not to accumulate gradients for this tensor.
    /// @return Shared pointer to the random Tensor
    template <int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> Tensor<Dim, DT>::eye(std::array<int, Dim> shape, bool requiresGrad) {
        // convert shape to Dimensions object
        Eigen::array<Eigen::Index, Dim> dimensions;
        for (int i = 0; i < Dim; i++) {
            dimensions[i] = shape[i];
        }

        Eigen::Tensor<DT, Dim> data = Eigen::Tensor<DT, Dim>(dimensions);
        data.setZero();
        Eigen::array<int, Dim> indices;
        
        for (int i = 0; i < Dim; i++) {
            for (int j = 0; j < Dim; j++) {
                indices[j] = i;
            }
            data(indices) = 1;
        }

        return std::make_shared<Tensor<Dim, DT>>(std::make_shared<Eigen::Tensor<DT, Dim>>(data), requiresGrad);
    }


    /// @brief Backward will simply do nothing for raw Tensors as they are leaf nodes in backpropagation.
    template <int Dim, typename DT>
    void Tensor<Dim, DT>::_backward()
    {
        return;
    }

    /// @brief Get the underlying data of the tensor (of type T).
    /// @details Will probably not be implemented outside of debugging.
    /// @return The actual data.
    template <int Dim, typename DT>
    std::shared_ptr<Eigen::Tensor<DT, Dim>> Tensor<Dim, DT>::getData()
    {
        return this->data;
    }

    /// @brief Zero out the gradient of this tensor.
    template <int Dim, typename DT>
    void Tensor<Dim, DT>::zeroGrad()
    {
        this->gradient->setZero();
    }

    /// @brief Accumulate the gradient of this tensor (intended to be called during backpropagation from nodes "upstream")
    template <int Dim, typename DT>
    void Tensor<Dim, DT>::addGrad(std::shared_ptr<Eigen::Tensor<DT, Dim>> grad)
    {
        *this->gradient += *grad;
    }

    // Explicit template instantiations
    template class Tensor<2, double>; // Used in tests
}