
/** @file */

#pragma once

#include "Tensor/TensorBase.hpp"
#include <vector>
#include <memory>
#include <random>
#include <cmath>

namespace PPNN
{

    /// @brief Currently supported weight initializers that can be used to initialize Tensors  in Neural Network layer.
    enum class WeightInititializers
    {
        ZEROS,
        ONES,
        RANDOM,
        XAVIER,
        HE
    };

    template <typename DT>
    class XavierUniform
    {
    public:
        explicit XavierUniform(int input_size, int output_size)
            : distribution_(-std::sqrt(6.0 / (input_size + output_size)),
                            std::sqrt(6.0 / (input_size + output_size))) {}

        DT operator()()
        {
            return distribution_(generator_);
        }

    private:
        std::mt19937 generator_;
        std::uniform_real_distribution<DT> distribution_;
    };

    template <typename DT>
    class He
    {
    public:
        explicit He(int input_size)
            : distribution_(0, std::sqrt(2.0 / input_size)) {}

        DT operator()()
        {
            return distribution_(generator_);
        }

    private:
        std::mt19937 generator_;
        std::normal_distribution<DT> distribution_;
    };

    template <int Dim, typename DT>
    class WeightInitializer
    {
    public:
        static void init(std::shared_ptr<PPGrad::TensorBase<Dim, DT>> tensor, WeightInititializers initializer)
        {
            switch (initializer)
            {
            case WeightInititializers::ZEROS:
                initZeros(tensor);
                break;
            case WeightInititializers::ONES:
                initOnes(tensor);
                break;
            case WeightInititializers::RANDOM:
                initRandom(tensor);
                break;
            case WeightInititializers::XAVIER:
                initXavier(tensor);
                break;
            case WeightInititializers::HE:
                initHe(tensor);
                break;
            default:
                break;
            }
        }

    private:
        static void initZeros(std::shared_ptr<PPGrad::TensorBase<Dim, DT>> tensor)
        {
            tensor->getData()->setZero();
        }

        static void initOnes(std::shared_ptr<PPGrad::TensorBase<Dim, DT>> tensor)
        {
            tensor->getData()->setConstant(1);
        }

        static void initRandom(std::shared_ptr<PPGrad::TensorBase<Dim, DT>> tensor)
        {
            tensor->getData()->setRandom();
        }

        static void initXavier(std::shared_ptr<PPGrad::TensorBase<Dim, DT>> tensor)
        {
            XavierUniform<DT> xavierUniform(tensor->getData()->dimensions()[1], tensor->getData()->dimensions()[0]);
            // tensor->getData()->unaryExpr([&](DT /*x*/)
            //                              { return xavierUniform(); }); // `unaryExpr` seems to not be implemented or something
            
            // Manual (recursive due to Dim) initialization
            std::array<int, Dim> indices;
            std::function<void(int)> init = [&](int dim)
            {
                if (dim == Dim)
                {
                    (*tensor->getData())(indices) = xavierUniform();
                }
                else
                {
                    for (int i = 0; i < tensor->getData()->dimension(dim); i++)
                    {
                        indices[dim] = i;
                        init(dim + 1);
                    }
                }
            };
            init(0);
        }

        static void initHe(std::shared_ptr<PPGrad::TensorBase<Dim, DT>> tensor)
        {
            He<DT> he(tensor->getData()->dimensions()[1]);
            // tensor->getData()->unaryExpr([&](DT /*x*/)
            //                              { return he(); }); // `unaryExpr` seems to not be implemented or something
            
            // Manual (recursive due to Dim) initialization
            std::array<int, Dim> indices;
            std::function<void(int)> init = [&](int dim)
            {
                if (dim == Dim)
                {
                    (*tensor->getData())(indices) = he();
                }
                else
                {
                    for (int i = 0; i < tensor->getData()->dimension(dim); i++)
                    {
                        indices[dim] = i;
                        init(dim + 1);
                    }
                }
            };
            init(0);
        }
    };
}