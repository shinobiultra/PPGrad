
/** @file */

#pragma once
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <vector>

#include "Tensor/Tensor.hpp"

namespace PPGrad
{

    template <int Dim, typename DT>
    class ReLUTensor : public Tensor<Dim, DT>
    {
    protected:
        std::shared_ptr<TensorBase<Dim, DT>> input; ///< The input to the ReLu operation. Result stored in `data`.

    public:
        ReLUTensor(std::shared_ptr<TensorBase<Dim, DT>> input)
        {
            this->input = input;
            this->data = std::make_shared<Eigen::Tensor<DT, Dim>>(input->getData()->cwiseMax((DT)0));
        }

        ReLUTensor(std::shared_ptr<TensorBase<Dim, DT>> input, bool requiresGrad)
        {
            this->input = input;
            this->data = std::make_shared<Eigen::Tensor<DT, Dim>>(input->getData()->cwiseMax((DT)0));
            this->gradient = std::make_shared<Eigen::Tensor<DT, Dim>>(input->getData()->dimensions());
            this->gradient->setZero();
            this->requiresGrad = requiresGrad;
        }

        void _backward() override;

        std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> getParents() override;
    };
}
