#include "Tensor/Tensor.hpp"
#include "Tensor/AddTensor.hpp"
#include "Tensor/SubTensor.hpp"
#include "Tensor/MultTensor.hpp"
#include "Tensor/AddSTensor.hpp"
#include "Tensor/SubSTensor.hpp"
#include "Tensor/MultSTensor.hpp"
#include "Tensor/DivSTensor.hpp"

#include <memory>

namespace PPGrad {

    
        /// @brief Backward will simply do nothing for raw Tensors as they are leaf nodes in backpropagation.
        template <int Dim, typename DT>
        void Tensor<Dim, DT>::backward() 
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

        // Tensor-Tensor operations

        /// @brief Operator overload for adding two tensors with correct gradient accumulation.
        /// @param other The other tensor to add.
        /// @return AddTensor<Dim, DT> The result of the addition with proper backward() implementation.
        template <int Dim, typename DT>
        std::shared_ptr<TensorBase<Dim, DT>> Tensor<Dim, DT>::operator+(std::shared_ptr<TensorBase<Dim, DT>> other) 
        {
            std::shared_ptr<TensorBase<Dim, DT>> result = std::make_shared<AddTensor<Dim, DT>>(std::make_shared<Tensor<Dim, DT>>(this->data), 
                                                                    std::make_shared<Tensor<Dim, DT>>(other->getData()));
            return result;
        }

        /// @brief Operator overload for subtracting two tensors with correct gradient accumulation.
        /// @param other The other tensor to subtract.
        /// @return SubTensor<Dim, DT> The result of the subtraction with proper backward() implementation.
        template <int Dim, typename DT>
        std::shared_ptr<TensorBase<Dim, DT>> Tensor<Dim, DT>::operator-(std::shared_ptr<TensorBase<Dim, DT>> other) 
        {
            std::shared_ptr<TensorBase<Dim, DT>> result = std::make_shared<SubTensor<Dim, DT>>(std::make_shared<Tensor<Dim, DT>>(this->data), 
                                                                    std::make_shared<Tensor<Dim, DT>>(other->getData()));
            return result;
        }

        /// @brief Operator overload for multiplying two tensors with correct gradient accumulation.
        /// @param other The other tensor to multiply.
        /// @return MultTensor<Dim, DT> The result of the multiplication with proper backward() implementation.
        template <int Dim, typename DT>
        std::shared_ptr<TensorBase<Dim, DT>> Tensor<Dim, DT>::operator*(std::shared_ptr<TensorBase<Dim, DT>> other) 
        {
            std::shared_ptr<TensorBase<Dim, DT>> result = std::make_shared<MultTensor<Dim, DT>>(std::make_shared<Tensor<Dim, DT>>(this->data), 
                                                                    std::make_shared<Tensor<Dim, DT>>(other->getData()));
            return result;
        }

        // Scalar-Tensor operations

        /// @brief Operator overload for adding a scalar to a tensor with correct gradient accumulation.
        /// @param other The scalar to add.
        /// @return AddTensor<Dim, DT> The result of the addition with proper backward() implementation.
        template <int Dim, typename DT>
        std::shared_ptr<TensorBase<Dim, DT>> Tensor<Dim, DT>::operator+(DT &other) 
        {
            std::shared_ptr<TensorBase<Dim, DT>> result = std::make_shared<AddSTensor<Dim, DT>>(std::make_shared<Tensor<Dim, DT>>(this->data), other);
            return result;
        }

        /// @brief Operator overload for subtracting a scalar from a tensor with correct gradient accumulation.
        /// @param other The scalar to subtract.
        /// @return SubTensor<Dim, DT> The result of the subtraction with proper backward() implementation.
        template <int Dim, typename DT>
        std::shared_ptr<TensorBase<Dim, DT>> Tensor<Dim, DT>::operator-(DT &other) 
        {
            std::shared_ptr<TensorBase<Dim, DT>> result = std::make_shared<SubSTensor<Dim, DT>>(std::make_shared<Tensor<Dim, DT>>(this->data), other);
            return result;
        }

        /// @brief Operator overload for multiplying a scalar with a tensor with correct gradient accumulation.
        /// @param other The scalar to multiply.
        /// @return MultTensor<Dim, DT> The result of the multiplication with proper backward() implementation.
        template <int Dim, typename DT>
        std::shared_ptr<TensorBase<Dim, DT>> Tensor<Dim, DT>::operator*(DT &other) 
        {
            std::shared_ptr<TensorBase<Dim, DT>> result = std::make_shared<MultSTensor<Dim, DT>>(std::make_shared<Tensor<Dim, DT>>(this->data), other);
            return result;
        }

        /// @brief Operator overload for dividing a tensor by a scalar with correct gradient accumulation.
        /// @param other The scalar to divide by.
        /// @return DivTensor<Dim, DT> The result of the division with proper backward() implementation.
        template <int Dim, typename DT>
        std::shared_ptr<TensorBase<Dim, DT>> Tensor<Dim, DT>::operator/(DT &other) 
        {
            std::shared_ptr<TensorBase<Dim, DT>> result = std::make_shared<DivSTensor<Dim, DT>>(std::make_shared<Tensor<Dim, DT>>(this->data), other);
            return result;
        }

    // Explicit template instantiations
    template class Tensor<2, double>; // Used in tests  
}