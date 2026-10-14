#include "assignment.hpp"

namespace assignment {

// TODO: implement training and inference. These constant outputs only exercise
// the interface and output shape; they are not working ML models or solutions.
void LogisticRegression::fit(const Matrix& x, const std::vector<int>&, std::uint64_t) {
    weights_.assign(x.empty() ? 0 : x.front().size(), 0.0);
    bias_ = 0.0;
}

std::vector<int> LogisticRegression::predict(const Matrix& x) const {
    return std::vector<int>(x.size(), 0);
}

void LinearSVM::fit(const Matrix& x, const std::vector<int>&, std::uint64_t) {
    weights_.assign(x.empty() ? 0 : x.front().size(), 0.0);
    bias_ = 0.0;
}

std::vector<int> LinearSVM::predict(const Matrix& x) const {
    return std::vector<int>(x.size(), 0);
}

void GBDTRegressor::fit(const Matrix&, const std::vector<double>&, std::uint64_t) {
    mean_ = 0.0;
    trees_.clear();
}

std::vector<double> GBDTRegressor::predict(const Matrix& x) const {
    return std::vector<double>(x.size(), 0.0);
}

void KMeans::fit(const Matrix& x, std::size_t k, std::uint64_t) {
    centers_.assign(k, std::vector<double>(x.empty() ? 0 : x.front().size(), 0.0));
}

std::vector<int> KMeans::predict(const Matrix& x) const {
    return std::vector<int>(x.size(), 0);
}

void DiscreteHMM::fit(
    const std::vector<std::vector<int>>&, const std::vector<std::vector<int>>&,
    std::size_t n_states, std::size_t n_symbols, std::uint64_t) {
    transitions_.assign(n_states, std::vector<double>(n_states, 0.0));
    emissions_.assign(n_states, std::vector<double>(n_symbols, 0.0));
    initial_.assign(n_states, 0.0);
}

std::vector<std::vector<int>> DiscreteHMM::predict(
    const std::vector<std::vector<int>>& observations) const {
    std::vector<std::vector<int>> paths;
    paths.reserve(observations.size());
    for (const auto& sequence : observations) {
        paths.emplace_back(sequence.size(), 0);
    }
    return paths;
}

}  // namespace assignment
