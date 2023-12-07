
/** @file */

#pragma once

#include "Tensor/TensorBase.hpp"
#include <vector>
#include <memory>

namespace PPNN
{

    enum class Optimizers
    {
        SGD,
        ADAM
    };

    template <int Dim, typename DT>
    class Optimizer
    {

    public:
        /// @brief The main function of the optimizer, intended to update the parameters of the model based on the gradients accumulated in the parameters & optimizer state.
        /// @param params List of [trainable] parameters of the model.
        virtual void update(std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> params) = 0;

        /// @brief Reset the state of the optimizer (e.g., momentum, velocity, etc.)
        virtual void resetState() = 0;
    };

    template <int Dim, typename DT>
    class SGD : public Optimizer<Dim, DT>
    {
    private:
        double learningRate;

    public:
        SGD(double learningRate)
        {
            this->learningRate = learningRate;
        }

        void update(std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> params) override
        {
            for (auto& param : params)
            {
                *param->getData() = *param->getData() - (learningRate * *param->getGrad());
                param->zeroGrad();
            }
        }

        void resetState() override
        {
            // Nothing to reset for SGD.
        }
    };

    template <int Dim, typename DT>
    class Adam : public Optimizer<Dim, DT>
    {
    private:
        double learningRate;
        double beta1;
        double beta2;
        double epsilon;

        std::vector<std::shared_ptr<Eigen::Tensor<DT, Dim>>> m;
        std::vector<std::shared_ptr<Eigen::Tensor<DT, Dim>>> v;
        int32_t t = 0;
        int32_t numParams;

    public:
        Adam(double learningRate, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        {
            this->learningRate = learningRate;
            this->beta1 = beta1;
            this->beta2 = beta2;
            this->epsilon = epsilon;
        }

        void update(std::vector<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> params) override
        {
            if (m.size() == 0)
            {
                numParams = params.size();
                for (std::shared_ptr<Eigen::Tensor<DT, Dim>>& param : params)
                {
                    std::shared_ptr<Eigen::Tensor<DT, Dim>> m0 = std::make_shared<Eigen::Tensor<DT, Dim>>(param->getData()->dimensions());
                    m0->setZero();
                    std::shared_ptr<Eigen::Tensor<DT, Dim>> v0 = std::make_shared<Eigen::Tensor<DT, Dim>>(param->getData()->dimensions());
                    v0->setZero();

                    m.push_back(m0);
                    v.push_back(v0);
                }
            }
            else if (m.size() != params.size())
            {
                throw std::runtime_error("Number of parameters changed between updates!");
            }

            t++;

            for (int32_t i = 0; i < numParams; i++)
            {
                *m[i] = (beta1 * *m[i]) + ((1 - beta1) * *params[i]->getGrad());
                *v[i] = (beta2 * *v[i]) + ((1 - beta2) * (params[i]->getGrad()->pow(2)));

                Eigen::Tensor<DT, Dim> mHat = *m[i] / (1 - std::pow(beta1, t));
                Eigen::Tensor<DT, Dim> vHat = *v[i] / (1 - std::pow(beta2, t));

                *params[i]->getData() = *params[i]->getData() - (learningRate * (mHat / (vHat.sqrt() + epsilon)));
                params[i]->zeroGrad();
            }
        }

        void resetState() override
        {
            m.clear();
            v.clear();
            t = 0;
        }
    };

}