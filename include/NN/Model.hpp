
/** @file */

#pragma once

#include "Tensor/TensorBase.hpp"
#include <vector>
#include <memory>

namespace PPNN
{

    /// @brief Base class for all models in PPNN library.
    /// @details This class is used to define the interface for all models in PPNN library and the users should override it to implement their own models.
    template <int Dim, typename DT>
    class Model
    {

    public:
        /// @brief Forward function of the model, intended to produce prediction for batched input.
        /// @param inputs Batch (vector) of input tensors intended for model prediction.
        /// @return Batch (vector) of output tensors produced by the model.
        virtual std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> forward(std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> inputs) = 0;

        /// @brief Return list of [trainable] parameters of the model (they shall contain the gradients after running backward() on each output produced by `forward()`).
        /// @return List of [trainable] parameters of the model.
        virtual std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> getParams() = 0;

        /// @brief Set list of [trainable] parameters of the model (intended to contain updated parameters after running optimizer).
        /// @param params List of (updated) [trainable] parameters of the model.
        virtual void setParams(std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> params) = 0;
    };

}