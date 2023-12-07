/** @file 
 * @brief Simplest (arguably) example of a Neural Network model built with PPGrad & PPNN framework. The goal is to teach the model to invert a sign of a single number.
 * @details The neural network used is simple Dense layer with 1 input and 1 output, and MSE loss function. The model is trained on 1000 examples of random numbers in range [-R, R] and their inverted values.
*/


#include "NN/Model.hpp"
#include "NN/Dense.hpp"
#include "NN/Loss.hpp"
#include "NN/Optimizer.hpp"
#include "NN/WeightInitializers.hpp"
#include "NN/Trainer.hpp"
#include "NN/DPTrainer.hpp"
#include "Tensor/TensorBase.hpp"
#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <cmath>


constexpr double LEARNING_RATE = 0.0001;
constexpr double R = 10.0;


class InvertorNN : public PPNN::Model<2, double>
{
private:
    std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> params;
    std::shared_ptr<PPNN::Dense<2, double>> dense;
    
public:

    InvertorNN()
    {
        dense = std::make_shared<PPNN::Dense<2, double>>(1, 1);
        params = dense->getParams();
    }

    std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> forward(std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> inputs) override
    {
        return dense->forward(inputs);
    }

    std::shared_ptr<PPGrad::TensorBase<2, double>> forward(std::shared_ptr<PPGrad::TensorBase<2, double>> input) override
    {
        return dense->forward(input);
    }

    std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>>& getParams()
    {
        return params;
    }

    void setParams(std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>>& params)
    {
        dense->setParams(params);
    }
};


int main()
{
    // Create model
    std::shared_ptr<InvertorNN> model = std::make_shared<InvertorNN>();

    // // Create optimizer
    std::shared_ptr<PPNN::Optimizer<2, double>> optimizer = std::make_shared<PPNN::SGD<2, double>>(LEARNING_RATE);

    // // Create loss function
    std::shared_ptr<PPNN::Loss<2, double>> loss = std::make_shared<PPNN::MSE<2, double>>();

    // // Create trainer
    std::shared_ptr<PPNN::DPTrainer<2, double>> trainer = std::make_shared<PPNN::DPTrainer<2, double>>(model, optimizer, loss);

    // // Create random number generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-R, R);

    // // Create training data
    std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> inputs;
    std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> targets;
    for (int i = 0; i < 1000; i++)
    {
        double x = dis(gen);
        Eigen::Tensor<double, 2> xT(1, 1);
        xT(0, 0) = x;
        Eigen::Tensor<double, 2> yT(1, 1);
        yT(0, 0) = -x;
        inputs.push_back(std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(xT)));
        targets.push_back(std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(yT)));
    }

    // // Train the model
    trainer->train(inputs, targets, 1000, 100, true);

    // // Test the model
    std::cout << "Testing the model..." << std::endl;
    for (int i = 0; i < 10; i++)
    {
        double x = dis(gen);
        Eigen::Tensor<double, 2> xT(1, 1);
        xT(0, 0) = x;
        std::shared_ptr<PPGrad::TensorBase<2, double>> input = std::make_shared<PPGrad::Tensor<2, double>>(std::make_shared<Eigen::Tensor<double, 2>>(xT));
        std::shared_ptr<PPGrad::TensorBase<2, double>> prediction = model->forward(input);
        std::cout << "Input: " << x << ", Prediction: " << (*prediction->getData())(0, 0) << ", Target: " << -x << std::endl;
    }

    // Print out the single learned weight (which should be -1.0)
    std::cout << "Learned weight: " << (*model->getParams()[0]->getData())(0, 0) << std::endl;

    return 0;
}