#include "Tensor/TensorBase.hpp"
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

    template<int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> operator+(std::shared_ptr<TensorBase<Dim, DT>> a, std::shared_ptr<TensorBase<Dim, DT>> b) {
        bool requiresGrad = a->getRequiresGrad() || b->getRequiresGrad();
        return std::make_shared<AddTensor<Dim, DT>>(a, b, requiresGrad);
    }

    template<int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>>operator-(std::shared_ptr<TensorBase<Dim, DT>> a, std::shared_ptr<TensorBase<Dim, DT>> b) {
        bool requiresGrad = a->getRequiresGrad() || b->getRequiresGrad();
        return std::make_shared<SubTensor<Dim, DT>>(a, b, requiresGrad);
    }

    template<int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> operator*(std::shared_ptr<TensorBase<Dim, DT>> a, std::shared_ptr<TensorBase<Dim, DT>> b) {
        bool requiresGrad = a->getRequiresGrad() || b->getRequiresGrad();
        return std::make_shared<MultTensor<Dim, DT>>(a, b, requiresGrad);
    }

    // Scalar-Tensor operations

    template<int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> operator+(std::shared_ptr<TensorBase<Dim, DT>> a, DT other) {
        bool requiresGrad = a->getRequiresGrad();
        return std::make_shared<AddSTensor<Dim, DT>>(a, other, requiresGrad);
    }

    template<int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> operator-(std::shared_ptr<TensorBase<Dim, DT>> a, DT other) {
        bool requiresGrad = a->getRequiresGrad();
        return std::make_shared<SubSTensor<Dim, DT>>(a, other, requiresGrad);
    }

    template<int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> operator*(std::shared_ptr<TensorBase<Dim, DT>> a, DT other) {
        bool requiresGrad = a->getRequiresGrad();
        return std::make_shared<MultSTensor<Dim, DT>>(a, other, requiresGrad);
    }


    template<int Dim, typename DT>
    std::shared_ptr<TensorBase<Dim, DT>> operator/(std::shared_ptr<TensorBase<Dim, DT>> a, DT other) {
        bool requiresGrad = a->getRequiresGrad();
        return std::make_shared<DivSTensor<Dim, DT>>(a, other, requiresGrad);
    }

    // Explicit template operator instantiations
    template std::shared_ptr<TensorBase<2, double>> operator+(std::shared_ptr<TensorBase<2, double>> a, std::shared_ptr<TensorBase<2, double>> b);
    template std::shared_ptr<TensorBase<2, double>> operator-(std::shared_ptr<TensorBase<2, double>> a, std::shared_ptr<TensorBase<2, double>> b);
    template std::shared_ptr<TensorBase<2, double>> operator*(std::shared_ptr<TensorBase<2, double>> a, std::shared_ptr<TensorBase<2, double>> b);
    template std::shared_ptr<TensorBase<2, double>> operator+(std::shared_ptr<TensorBase<2, double>> a, double b);
    template std::shared_ptr<TensorBase<2, double>> operator-(std::shared_ptr<TensorBase<2, double>> a, double b);
    template std::shared_ptr<TensorBase<2, double>> operator*(std::shared_ptr<TensorBase<2, double>> a, double b);
    template std::shared_ptr<TensorBase<2, double>> operator/(std::shared_ptr<TensorBase<2, double>> a, double b);
} // namespace PPGrad