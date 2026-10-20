#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace assignment {

using Matrix = std::vector<std::vector<double>>;

// Course API scaffold only: no ML algorithm is implemented.
// Replace the zero-output placeholders in src/assignment.cpp with your models.
class LogisticRegression {
public:
    void fit(const Matrix& x, const std::vector<int>& y, std::uint64_t seed);
    std::vector<int> predict(const Matrix& x) const;
private:
    std::vector<double> weights_;
    double bias_ = 0.0;
};

class SVM {
public:
    void fit(const Matrix& x, const std::vector<int>& y, std::uint64_t seed);
    std::vector<int> predict(const Matrix& x) const;
};

// New-version classification API: labels 0..8, all 856 input features retained.
class GBDTClassifier {
public:
    void fit(const Matrix& x, const std::vector<int>& y, std::uint64_t seed);
    std::vector<int> predict(const Matrix& x) const;
};

class KMeans {
public:
    void fit(const Matrix& x, std::size_t k, std::uint64_t seed);
    std::vector<int> predict(const Matrix& x) const;
private:
    Matrix centers_;
};

class DiscreteHMM {
public:
    // State IDs keep the supervised training-label meaning across all sequences.
    void fit(const std::vector<std::vector<int>>& observations,
             const std::vector<std::vector<int>>& state_labels,
             std::size_t n_states, std::size_t n_symbols, std::uint64_t seed);
    std::vector<std::vector<int>> predict(
        const std::vector<std::vector<int>>& observations) const;
private:
    Matrix transitions_, emissions_;
    std::vector<double> initial_;
};

}  // namespace assignment
