
/** @file */

#include "Tensor/TensorBase.hpp"
#include "TopologicalSort.hpp"
#include <stack>
#include <unordered_set>
#include <memory>

namespace PPGrad
{

    template <int Dim, typename DT>
    std::stack<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> topologicalSort(std::shared_ptr<PPGrad::TensorBase<Dim, DT>> root)
    {
        std::stack<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> stack;
        std::unordered_set<std::shared_ptr<PPGrad::TensorBase<Dim, DT>>> visited;

        std::function<void(std::shared_ptr<PPGrad::TensorBase<Dim, DT>>)> visit = [&](std::shared_ptr<PPGrad::TensorBase<Dim, DT>> node)
        {
            visited.insert(node);

            for (std::shared_ptr<PPGrad::TensorBase<Dim, DT>> parent : node->getParents())
            {
                if (visited.count(parent))
                {
                    continue;
                }
                visit(parent);
            }

            stack.push(node);
        };

        visit(root);

        return stack;
    }

    // Explicit template instantiations
    template std::stack<std::shared_ptr<PPGrad::TensorBase<2, double>>> topologicalSort(std::shared_ptr<PPGrad::TensorBase<2, double>> root);
}