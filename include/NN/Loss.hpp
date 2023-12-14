
/** @file */

#pragma once

#include "Tensor/TensorBase.hpp"
#include <vector>
#include <memory>

namespace PPNN
{

    enum class Losses
    {
        MSE,
        NLL
    };

    /// @brief Base class for all loss functions in PPNN library.
    /// @tparam DT DType of the expected Tensors.
    /// @tparam Dim Dimension of the expected Tensors.
    template <int Dim, typename DT>
    class Loss
    {
    public:
        /// @brief Calculate the loss between the prediction and the label (single example) and (in case training == true) initialize the gradients in the prediction Tensor to prepare for calliong backward().
        /// @param prediction Prediction Tensor produced by the model.
        /// @param label Ground truth label Tensor.
        /// @param training Determines whether the gradients should be initialized or not.
        /// @return Loss value between the prediction and the label.
        virtual DT operator()(std::shared_ptr<PPGrad::TensorBase<Dim, DT>> prediction, std::shared_ptr<PPGrad::TensorBase<Dim, DT>> label, bool training = false) = 0;

        /// @brief Calculate the loss between the predictions and the labels (batch of examples) and (in case training == true) initialize the gradients in the prediction Tensor to prepare for calliong backward().
        /// @param predictions Prediction Tensors produced by the model.
        /// @param labels Ground truth labels Tensor.
        /// @param training Determines whether the gradients should be initialized or not.
        /// @return Loss value between the prediction and the label.
        virtual DT operator()(std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> predictions, std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> labels, bool training = false) = 0;
    };

    /// @brief Mean Squared Error loss function.
    /// @tparam DT DType of the expected Tensors.
    /// @tparam Dim Dimension of the expected Tensors.
    template <int Dim, typename DT>
    class MSE : public Loss<Dim, DT>
    {
    public:
        DT operator()(std::shared_ptr<PPGrad::TensorBase<Dim, DT>> prediction, std::shared_ptr<PPGrad::TensorBase<Dim, DT>> label, bool training = false) override
        {
            Eigen::Tensor<double, 0> lossT = (*prediction->getData() - *label->getData()).square().sum() / (double)prediction->getData()->size();
            DT loss = lossT();
            if (training)
            {
                *prediction->getGrad() += (*prediction->getData() - *label->getData()) * (2.0 / prediction->getData()->size());
            }
            return loss;
        }

        DT operator()(std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> predictions, std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> labels, bool training = false) override
        {
            DT loss = 0.0;
            for (size_t i = 0; i < predictions.size(); i++)
            {
                loss += this->operator()(predictions[i], labels[i], training);
            }
            return loss / predictions.size();
        }
    };

    /// @brief Negative Log Likelihood loss function.
    /// @tparam DT DType of the expected Tensors.
    /// @tparam Dim Dimension of the expected Tensors.
    template <int Dim, typename DT>
    class NLL : public Loss<Dim, DT>
    {
    public:
        DT operator()(std::shared_ptr<PPGrad::TensorBase<Dim, DT>> prediction, std::shared_ptr<PPGrad::TensorBase<Dim, DT>> label, bool training = false) override
        {
            Eigen::Tensor<double, 0> lossT = -(*label->getData() * prediction->getData()->log()).sum();
            DT loss = lossT();
            if (training)
            {
                *prediction->getGrad() += -(*label->getData() / *prediction->getData());
            }
            return loss;
        }

        DT operator()(std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> predictions, std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> labels, bool training = false) override
        {
            DT loss = 0.0;
            for (size_t i = 0; i < predictions.size(); i++)
            {
                loss += this->operator()(predictions[i], labels[i], training);
            }
            return loss / predictions.size();
        }
    };

}