
/** @file */


#pragma once
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <vector>

namespace PPGrad
{

    /// @brief Abstract Tensor class for all Eigen accelerated math on tensors.
    /// @details This class is a wrapper around Eigen::Matrix and Eigen::Tensor and is supposed to be immutable. Each operation instantiates a new Tensor object.s
    /// @tparam T The type of the underlying data (Eigen::Matrix or Eigen::Tensor).
    /// @tparam DT The type of the scalar data (i.e., individual elements of `data`) - float or double expected.
    template <int Dim, typename DT>
    class TensorBase
    {

    protected:
        std::shared_ptr<Eigen::Tensor<DT, Dim>> data;     ///< Eigen::Matrix or Eigen::Tensor
        std::shared_ptr<Eigen::Tensor<DT, Dim>> gradient; ///< Same shape as data because we'll only ever be doing gradient descent on scalar loss functions.

        // TensorBase() = delete; ///< No default constructor.

        bool requiresGrad = false; ///< Whether or not this tensor requires gradient accumulation.

    public:
        /// @brief Get the underlying data of the tensor (of type T).
        /// @details Will probably not be implemented outside of debugging.
        /// @return The actual data.
        virtual std::shared_ptr<Eigen::Tensor<DT, Dim>> getData() = 0;

        /// @brief Return true if this tensor requires gradient accumulation.
        /// @return `true` if this tensor requires gradient accumulation.
        virtual bool getRequiresGrad()
        {
            return this->requiresGrad;
        }

        /// @brief Calculate the gradient of this tensor with respect to it's inputs.
        virtual void _backward() = 0;

        /// @brief Zero out the gradient of this tensor.
        virtual void zeroGrad() = 0;

        /// @brief Accumulate the gradient of this tensor (intended to be called during backpropagation from nodes "upstream")
        /// @param grad The gradient to accumulate.
        virtual void addGrad(std::shared_ptr<Eigen::Tensor<DT, Dim>> grad) = 0;

        std::shared_ptr<Eigen::Tensor<DT, Dim>> getGrad()
        {
            return this->gradient;
        }

        /// @brief Get the parents of this tensor in the computation graph.
        virtual std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> getParents() = 0;


        // ------- Static methods ------- //

        /// @brief Topologically order all parents of this tensor and call _backward() on them.
        static void backward(std::shared_ptr<PPGrad::TensorBase<Dim, DT>> root);
    };

    // ------------ Tensor operations ------------ //

    // Tensor-Tensor operations

    /// @brief Operator overload for adding two tensors with correct gradient accumulation.
    /// @param other The other tensor to add.
    /// @return AddTensor<Dim, DT> The result of the addition with proper _backward() implementation.
    template <int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> operator+(std::shared_ptr<TensorBase<Dim, DT>> a, std::shared_ptr<TensorBase<Dim, DT>> b);

    /// @brief Operator overload for subtracting two tensors with correct gradient accumulation.
    /// @param other The other tensor to subtract.
    /// @return SubTensor<Dim, DT> The result of the subtraction with proper _backward() implementation.
    template <int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> operator-(std::shared_ptr<TensorBase<Dim, DT>> a, std::shared_ptr<TensorBase<Dim, DT>> b);

    /// @brief Operator overload for multiplying two tensors with correct gradient accumulation.
    /// @param other The other tensor to multiply.
    /// @return MultTensor<Dim, DT> The result of the multiplication with proper _backward() implementation.
    template <int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> operator*(std::shared_ptr<TensorBase<Dim, DT>> a, std::shared_ptr<TensorBase<Dim, DT>> b);

    // Scalar-Tensor operations

    /// @brief Operator overload for adding a scalar to a tensor with correct gradient accumulation.
    /// @param other The scalar to add.
    /// @return AddTensor<Dim, DT> The result of the addition with proper _backward() implementation.
    template <int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> operator+(std::shared_ptr<TensorBase<Dim, DT>> a, DT other);

    /// @brief Operator overload for subtracting a scalar from a tensor with correct gradient accumulation.
    /// @param other The scalar to subtract.
    /// @return SubTensor<Dim, DT> The result of the subtraction with proper _backward() implementation.
    template <int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> operator-(std::shared_ptr<TensorBase<Dim, DT>> a, DT other);

    /// @brief Operator overload for multiplying a scalar with a tensor with correct gradient accumulation.
    /// @param other The scalar to multiply.
    /// @return MultTensor<Dim, DT> The result of the multiplication with proper _backward() implementation.
    template <int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> operator*(std::shared_ptr<TensorBase<Dim, DT>> a, DT other);

    /// @brief Operator overload for dividing a tensor by a scalar with correct gradient accumulation.
    /// @param other The scalar to divide by.
    /// @return DivTensor<Dim, DT> The result of the division with proper _backward() implementation.
    template <int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> operator/(std::shared_ptr<TensorBase<Dim, DT>> a, DT other);


    /// @brief Operator overload for adding a scalar to a tensor with correct gradient accumulation.
    /// @param other The scalar to add.
    /// @return AddTensor<Dim, DT> The result of the addition with proper _backward() implementation.
    template <int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> operator+(DT other, std::shared_ptr<TensorBase<Dim, DT>> a);

    /// @brief Operator overload for subtracting a scalar from a tensor with correct gradient accumulation.
    /// @param other The scalar to subtract.
    /// @return SubTensor<Dim, DT> The result of the subtraction with proper _backward() implementation.
    template <int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> operator-(DT other, std::shared_ptr<TensorBase<Dim, DT>> a);

    /// @brief Operator overload for multiplying a scalar with a tensor with correct gradient accumulation.
    /// @param other The scalar to multiply.
    /// @return MultTensor<Dim, DT> The result of the multiplication with proper _backward() implementation.
    template <int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> operator*(DT other, std::shared_ptr<TensorBase<Dim, DT>> a);

    
}
