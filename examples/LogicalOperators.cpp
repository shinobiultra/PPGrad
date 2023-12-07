/** @file 
 * @brief Example of using PPNN library to train a simple neural network to learn given logical operator. 
 * @details The neural network used is simple Dense layer with 2 inputs, X hidden layer with Y neurons, and 1 output. The model is trained on N examples of (A, B) and their logical operator (AND, OR, XOR, etc.).
 * 
*/


#include "NN/Model.hpp"
#include "NN/Dense.hpp"
#include "NN/Loss.hpp"
#include "NN/Optimizer.hpp"
#include "NN/WeightInitializers.hpp"
#include "NN/Trainer.hpp"
#include "Tensor/TensorBase.hpp"
#include <iostream>
#include <vector>
#include <memory>
#include <random>
#include <cmath>

enum class LogicalOperator
{
    AND,
    OR,
    XOR,
    NAND,
    NOR,
    XNOR
};

constexpr double LEARNING_RATE = 0.001;
constexpr int HIDDEN_LAYERS = 2;
constexpr int HIDDEN_SIZE = 16;
constexpr int EPOCHS = 100;
constexpr int BATCH_SIZE = 2;
constexpr int N = 100;
constexpr LogicalOperator LOGICAL_OPERATOR = LogicalOperator::OR;


class LogicalNN : public PPNN::Model<2, double>
{
private:
    std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> params;
    std::vector<std::shared_ptr<PPNN::Dense<2, double>>> layers;
    
public:

    LogicalNN(int hiddenLayers, int hiddenSize)
    {
        if (hiddenLayers == 0){
            layers.push_back(std::make_shared<PPNN::Dense<2, double>>(2, 1));
            params = layers[0]->getParams();
        } else {
            layers.push_back(std::make_shared<PPNN::Dense<2, double>>(2, hiddenSize));
            params = layers[0]->getParams();
            for (int i = 0; i < hiddenLayers - 1; i++){
                layers.push_back(std::make_shared<PPNN::Dense<2, double>>(hiddenSize, hiddenSize));
                params.insert(params.end(), layers[i + 1]->getParams().begin(), layers[i + 1]->getParams().end());
            }
            layers.push_back(std::make_shared<PPNN::Dense<2, double>>(hiddenSize, 1));
            params.insert(params.end(), layers[hiddenLayers]->getParams().begin(), layers[hiddenLayers]->getParams().end());            
        }
    }

    std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> forward(std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> inputs) override
    {
        std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> outputs = inputs;
        for (std::shared_ptr<PPNN::Dense<2, double>> layer : layers)
        {
            outputs = layer->forward(outputs);
        }
        return outputs;
    }

    std::shared_ptr<PPGrad::TensorBase<2, double>> forward(std::shared_ptr<PPGrad::TensorBase<2, double>> input) override
    {
        std::shared_ptr<PPGrad::TensorBase<2, double>> output = input;
        for (std::shared_ptr<PPNN::Dense<2, double>> layer : layers)
        {
            output = layer->forward(output);
        }
        return output;
    }

    std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>>& getParams()
    {
        return params;
    }

    void setParams(std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>>& params)
    {
        int idx = 0;
        for (std::shared_ptr<PPNN::Dense<2, double>>& layer : layers)
        {
            std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> layerParams;
            for (size_t i = 0; i < layer->getParams().size(); i++)
            {
                layerParams.push_back(params[idx]);
                idx++;
            }
            layer->setParams(layerParams);
        }
    }
    

};


int main()
{
    // Create model
    std::shared_ptr<LogicalNN> model = std::make_shared<LogicalNN>(HIDDEN_LAYERS, HIDDEN_SIZE);

    // // Create optimizer
    std::shared_ptr<PPNN::Optimizer<2, double>> optimizer = std::make_shared<PPNN::SGD<2, double>>(LEARNING_RATE);

    // // Create loss function
    std::shared_ptr<PPNN::Loss<2, double>> loss = std::make_shared<PPNN::MSE<2, double>>();

    // // Create trainer
    std::shared_ptr<PPNN::Trainer<2, double>> trainer = std::make_shared<PPNN::Trainer<2, double>>(model, optimizer, loss);

    // // Create training data
    std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> inputs;
    std::vector<std::shared_ptr<PPGrad::TensorBase<2, double>>> targets;
    for (int i = 0; i < N; i++)
    {
        std::shared_ptr<PPGrad::TensorBase<2, double>> input = PPGrad::Tensor<2, double>::zeros({2, 1});
        std::shared_ptr<PPGrad::TensorBase<2, double>> target = PPGrad::Tensor<2, double>::zeros({1, 1});
        double a = (double) (rand() % 2);
        double b = (double) (rand() % 2);
        input->getData()->setValues({{a}, {b}});
        switch (LOGICAL_OPERATOR)
        {
            case LogicalOperator::AND:
                target->getData()->setValues({{a * b}});
                break;
            case LogicalOperator::OR:
                target->getData()->setValues({{a + b > 0.0 ? 1.0 : 0.0}});
                break;
            case LogicalOperator::XOR:
                target->getData()->setValues({{a + b == 1.0 ? 1.0 : 0.0}});
                break;
            case LogicalOperator::NAND:
                target->getData()->setValues({{a * b == 0.0 ? 1.0 : 0.0}});
                break;
            case LogicalOperator::NOR:
                target->getData()->setValues({{a + b == 0.0 ? 1.0 : 0.0}});
                break;
            case LogicalOperator::XNOR:
                target->getData()->setValues({{a + b != 1.0 ? 1.0 : 0.0}});
                break;
        }
        inputs.push_back(input);
        targets.push_back(target);
    }

    // Train model
    trainer->train(inputs, targets, EPOCHS, BATCH_SIZE, true);

    // Test model
    std::cout << "Testing the model..." << std::endl;
    for (int i = 0; i < 10; i++)
    {
        std::shared_ptr<PPGrad::TensorBase<2, double>> input = PPGrad::Tensor<2, double>::zeros({2, 1});
        double a = (double) (rand() % 2);
        double b = (double) (rand() % 2);
        input->getData()->setValues({{a}, {b}});
        std::shared_ptr<PPGrad::TensorBase<2, double>> prediction = model->forward(input);
        std::cout << "Input: " << a << ", " << b << ", Prediction: " << (*prediction->getData())(0, 0) << std::endl;
    }


    return 0;
}