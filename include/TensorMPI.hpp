
/** @file
 * @brief Helper functions for distributed training.
 */

#pragma once
#include "Tensor/TensorBase.hpp"
#include "NN/Model.hpp"
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <vector>
#include <mpi.h>

namespace PPGrad
{

    /// @brief Create an Eigen::TensorMap object from a raw pointer to data with shape of `dims` and index sequence `Is`.
    /// @tparam DT Data type of the tensor.
    /// @tparam Dim Number of tensor dimensions.
    /// @tparam ...Is Special Index Sequence to handle unknown number of dimensions in templates.
    /// @param data Raw pointer to the data.
    /// @param dims Array of dimensions (i.e., shape of the tensor).
    /// @param Is Index sequence.
    /// @return Eigen::TensorMap object of shape specified by `dims` and containing the data pointed to by `data`.
    template <int Dim, typename DT, std::size_t... Is>
    Eigen::TensorMap<Eigen::Tensor<DT, Dim>> createTensorMapHelper(DT *data, const std::array<int, Dim> &dims, std::index_sequence<Is...>)
    {
        const Eigen::array<Eigen::Index, Dim> dimensions = {dims[Is]...};
        return Eigen::TensorMap<Eigen::Tensor<DT, Dim>>(data, dimensions);
    }

    /// @brief Create an Eigen::TensorMap object from a raw pointer to data and a vector of dimensions. (Helper function due to "unknown" dimension).
    /// @tparam DT Data type of the tensor.
    /// @tparam Dim Number of tensor dimensions.
    /// @param data Raw pointer to the data.
    /// @param dims Vector of dimensions (i.e., shape of the tensor).
    /// @return Eigen::TensorMap object of shape specified by `dims` and containing the data pointed to by `data`.
    template <int Dim, typename DT>
    Eigen::TensorMap<Eigen::Tensor<DT, Dim>> createTensorMap(DT *data, const std::vector<int> &dims)
    {
        assert(dims.size() == Dim);
        std::array<int, Dim> dimsArray;
        std::copy(dims.begin(), dims.end(), dimsArray.begin());
        return createTensorMapHelper<Dim, DT>(data, dimsArray, std::make_index_sequence<Dim>{});
    }

    /// @brief Scatter a vector of Eigen::Tensor objects across MPI processes.
    /// @details Root (rank 0) process will flatten the data and scatter it across all processes, ignoring the remainder if the data is not evenly divisible.
    /// @tparam DT Data type of the tensors.
    /// @tparam Dim Dimension of the tensors.
    /// @param inputs Vector of Eigen::Tensor objects to scatter.
    /// @param worldSize Number of MPI processes in the comm world.
    /// @param worldRank Rank of the calling MPI process.
    /// @return Vector of shared pointers to TensorBase objects containing the scattered tensors (in the underlying Eigen::Tensor).
    template <int Dim, typename DT>
    std::vector<std::shared_ptr<TensorBase<2, double>>> tensorScatter(
        std::vector<Eigen::Tensor<DT, Dim>> &inputs,
        const int worldSize,
        const int worldRank)
    {
        std::vector<std::shared_ptr<TensorBase<Dim, DT>>> outputs;

        // Flatten the data on the root process
        std::vector<DT> flat_data;
        if (worldRank == 0)
        {
            for (const auto &tensor : inputs)
            {
                flat_data.insert(flat_data.end(), tensor.data(), tensor.data() + tensor.size());
            }
        }

        // Broadcast the number of elements to send to each process
        int sendCount;
        if (worldRank == 0)
        {
            // Calculate the number of elements to send to each process
            sendCount = flat_data.size() / worldSize;
        }
        MPI_Bcast(&sendCount, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Issue a graceful warning if the data is not evenly divisible amongst the processes
        if (flat_data.size() % worldSize != 0)
        {
            std::cerr << "Warning: Data is not evenly divisible amongst processes." << std::endl;

#ifdef PPGRAD_DEBUG
            std::cerr << "Data size: " << flat_data.size() << std::endl;
            std::cerr << "World size: " << worldSize << std::endl;
            std::cerr << "Send count: " << sendCount << std::endl;
            std::cerr << "Input Tensors: " << inputs.size() << std::endl;
#endif
        }

        // Create a receive buffer
        std::vector<double> recv_data(sendCount);

        // Scatter the data
        MPI_Scatter(flat_data.data(), sendCount, MPI_DOUBLE, recv_data.data(), sendCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        // Distribute the dimensions of the tensors (assuming they are all the same size)
        std::vector<int> tensorDims;
        int tensorDimLength;

        if (worldRank == 0)
        {
            auto &dimensions = inputs[0].dimensions();
            for (size_t i = 0; i < dimensions.size(); i++)
            {
                tensorDims.push_back(dimensions[i]);
            }
            tensorDimLength = dimensions.size();
        }

        // Distribute the lengths of the dimensions
        MPI_Bcast(&tensorDimLength, 1, MPI_INT, 0, MPI_COMM_WORLD);

        // Distribute the dimensions
        if (worldRank != 0)
        {
            tensorDims.resize(tensorDimLength);
        }
        MPI_Bcast(tensorDims.data(), tensorDimLength, MPI_INT, 0, MPI_COMM_WORLD);

        int tensorSize = tensorDims[0];
        for (size_t i = 1; i < tensorDims.size(); i++)
            tensorSize *= tensorDims[i];

        // Unflatten the data
        for (size_t i = 0; i < recv_data.size(); i += tensorSize)
        {
            Eigen::TensorMap<Eigen::Tensor<DT, Dim>> tensorMap = createTensorMap<Dim, DT>(recv_data.data() + i, tensorDims);
            Eigen::Tensor<DT, Dim> tensor = tensorMap;
            outputs.push_back(std::make_shared<Tensor<Dim, DT>>(std::make_shared<Eigen::Tensor<DT, Dim>>(tensor)));
        }

        return outputs;
    }

    /// @brief Broadcast all underlying parameters of `PPNN::Model` object to all MPI processes.
    /// @details All callers should have the same model ready, with different parameters only!
    /// @tparam DT Data type of the model parameters.
    /// @tparam Dim Dimension of the model parameters.
    /// @param model Pointer to the model to broadcast.
    /// @param worldSize Number of MPI processes in the comm world.
    /// @param worldRank Rank of the calling MPI process.
    /// @return Pointer to the model with parameters broadcasted to all MPI processes (i.e., same for all calling processes)
    template <int Dim, typename DT>
    std::shared_ptr<PPNN::Model<Dim, DT>> modelBroadcast(
        std::shared_ptr<PPNN::Model<Dim, DT>> model,
        const int worldSize,
        const int worldRank)
    {
        // Make sure all nodes have the same number of parameters
        int paramCount = model->getParams().size();
        MPI_Allreduce(MPI_IN_PLACE, &paramCount, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        paramCount /= worldSize;

        if (paramCount != model->getParams().size())
        {
            std::cerr << "Error: All nodes must have the same number of parameters in the model before broadcasting." << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        // Broadcast the parameters
        for (size_t i = 0; i < paramCount; i++)
        {
            // Get the data from the model
            std::vector<int> tensorDims;
            for (int dim : model->getParams()[i]->getData()->dimensions())
                tensorDims.push_back(dim);

            Eigen::TensorMap<Eigen::Tensor<DT, Dim>> tensorMap = createTensorMap<Dim, DT>(model->getParams()[i]->getData()->data(), tensorDims);
            Eigen::Tensor<DT, Dim> tensor = tensorMap;

            // Broadcast the data
            MPI_Bcast(tensor.data(), tensor.size(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

            // Set the data in the model
            *model->getParams()[i]->getData() = tensor;
        }

        return model;
    }

} // namespace PPGrad