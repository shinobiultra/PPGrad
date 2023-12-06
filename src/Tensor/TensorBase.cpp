
/** @file */

#include "Tensor/TensorBase.hpp"
#include "Tensor/Tensor.hpp"
#include "Tensor/AddTensor.hpp"
#include "Tensor/SubTensor.hpp"
#include "Tensor/MultTensor.hpp"
#include "Tensor/AddSTensor.hpp"
#include "Tensor/SubSTensor.hpp"
#include "Tensor/MultSTensor.hpp"
#include "Tensor/DivSTensor.hpp"
#include "TopologicalSort.hpp"
#include <memory>
#include <stack>

namespace PPGrad
{

    /// @brief Calculate the gradient of this tensor with respect to it's inputs.
    template <int Dim, typename DT>
    void TensorBase<Dim, DT>::backward(std::shared_ptr<PPGrad::TensorBase<Dim, DT>> root)
    {
        std::stack<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> sortedNodes = topologicalSort<Dim, DT>(root);
        while (!sortedNodes.empty())
        {
            sortedNodes.top()->_backward();
            sortedNodes.pop();
        }
    }

    template <int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> operator+(std::shared_ptr<TensorBase<Dim, DT>> a, std::shared_ptr<TensorBase<Dim, DT>> b)
    {
        bool requiresGrad = a->getRequiresGrad() || b->getRequiresGrad();
        return std::make_shared<AddTensor<Dim, DT>>(a, b, requiresGrad);
    }

    template <int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> operator-(std::shared_ptr<TensorBase<Dim, DT>> a, std::shared_ptr<TensorBase<Dim, DT>> b)
    {
        bool requiresGrad = a->getRequiresGrad() || b->getRequiresGrad();
        return std::make_shared<SubTensor<Dim, DT>>(a, b, requiresGrad);
    }

    template <int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> operator*(std::shared_ptr<TensorBase<Dim, DT>> a, std::shared_ptr<TensorBase<Dim, DT>> b)
    {
        bool requiresGrad = a->getRequiresGrad() || b->getRequiresGrad();
        return std::make_shared<MultTensor<Dim, DT>>(a, b, requiresGrad);
    }

    // Scalar-Tensor operations

    template <int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> operator+(std::shared_ptr<TensorBase<Dim, DT>> a, DT other)
    {
        bool requiresGrad = a->getRequiresGrad();
        return std::make_shared<AddSTensor<Dim, DT>>(a, other, requiresGrad);
    }

    template <int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> operator-(std::shared_ptr<TensorBase<Dim, DT>> a, DT other)
    {
        bool requiresGrad = a->getRequiresGrad();
        return std::make_shared<SubSTensor<Dim, DT>>(a, other, requiresGrad);
    }

    template <int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> operator*(std::shared_ptr<TensorBase<Dim, DT>> a, DT other)
    {
        bool requiresGrad = a->getRequiresGrad();
        return std::make_shared<MultSTensor<Dim, DT>>(a, other, requiresGrad);
    }

    template <int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> operator/(std::shared_ptr<TensorBase<Dim, DT>> a, DT other)
    {
        bool requiresGrad = a->getRequiresGrad();
        return std::make_shared<DivSTensor<Dim, DT>>(a, other, requiresGrad);
    }

    /// @brief Operator overload for adding a scalar to a tensor with correct gradient accumulation.
    /// @param other The scalar to add.
    /// @return AddTensor<Dim, DT> The result of the addition with proper _backward() implementation.
    template <int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> operator+(DT other, std::shared_ptr<TensorBase<Dim, DT>> a)
    {
        bool requiresGrad = a->getRequiresGrad();
        return std::make_shared<AddSTensor<Dim, DT>>(a, other, requiresGrad);
    }

    /// @brief Operator overload for subtracting a scalar from a tensor with correct gradient accumulation.
    /// @param other The scalar to subtract.
    /// @return SubTensor<Dim, DT> The result of the subtraction with proper _backward() implementation.
    template <int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> operator-(DT other, std::shared_ptr<TensorBase<Dim, DT>> a)
    {
        bool requiresGrad = a->getRequiresGrad();
        return std::make_shared<SubSTensor<Dim, DT>>(a, other, requiresGrad);
    }

    /// @brief Operator overload for multiplying a scalar with a tensor with correct gradient accumulation.
    /// @param other The scalar to multiply.
    /// @return MultTensor<Dim, DT> The result of the multiplication with proper _backward() implementation.
    template <int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> operator*(DT other, std::shared_ptr<TensorBase<Dim, DT>> a)
    {
        bool requiresGrad = a->getRequiresGrad();
        return std::make_shared<MultSTensor<Dim, DT>>(a, other, requiresGrad);
    }



    // Explicit template operator instantiations
    template std::shared_ptr<TensorBase<2, double>> operator+(std::shared_ptr<TensorBase<2, double>> a, std::shared_ptr<TensorBase<2, double>> b);
    template std::shared_ptr<TensorBase<2, double>> operator-(std::shared_ptr<TensorBase<2, double>> a, std::shared_ptr<TensorBase<2, double>> b);
    template std::shared_ptr<TensorBase<2, double>> operator*(std::shared_ptr<TensorBase<2, double>> a, std::shared_ptr<TensorBase<2, double>> b);
    template std::shared_ptr<TensorBase<2, double>> operator+(std::shared_ptr<TensorBase<2, double>> a, double b);
    template std::shared_ptr<TensorBase<2, double>> operator-(std::shared_ptr<TensorBase<2, double>> a, double b);
    template std::shared_ptr<TensorBase<2, double>> operator*(std::shared_ptr<TensorBase<2, double>> a, double b);
    template std::shared_ptr<TensorBase<2, double>> operator/(std::shared_ptr<TensorBase<2, double>> a, double b);

    // explicit bakward() template instantiations
    template void TensorBase<2, double>::backward(std::shared_ptr<PPGrad::TensorBase<2, double>> root);
} // namespace PPGrad