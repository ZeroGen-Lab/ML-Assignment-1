// Public shared test driver for ML-Assignment-1.
//
// This driver is course-provided infrastructure: it reads a whitespace
// token protocol from stdin, calls the student assignment API, and prints
// results to stdout.  It is NOT part of the graded implementation area and
// is not accepted in source submissions (packaging only takes include/ and
// src/).  teacher/grader/driver.cpp forwards to this exact file so the
// public selftest and the teacher-side pipeline exercise one protocol.

#include "assignment.hpp"

#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>

using assignment::Matrix;

namespace {
template <class T> T read() {
    T value;
    if (!(std::cin >> value)) throw std::invalid_argument("truncated/invalid protocol input");
    return value;
}

Matrix matrix(std::size_t n, std::size_t d) {
    if (n == 0 || d == 0 || n > 10000 || d > 1000) throw std::invalid_argument("prototype dimensions");
    Matrix x(n, std::vector<double>(d));
    for (auto& row : x) for (auto& value : row) value = read<double>();
    return x;
}

template <class T> std::vector<T> vector(std::size_t n) {
    std::vector<T> values(n);
    for (auto& value : values) value = read<T>();
    return values;
}

template <class T> void output(const std::vector<T>& values) {
    std::cout << values.size();
    for (const auto& value : values) std::cout << ' ' << value;
    std::cout << '\n';
}
}  // namespace

int main() {
    try {
        std::cout << std::setprecision(17);
        const auto task = read<std::string>();
        if (task == "lr" || task == "svm" || task == "gbdt") {
            const auto seed = read<std::uint64_t>();
            const auto n = read<std::size_t>();
            const auto d = read<std::size_t>();
            const auto m = read<std::size_t>();
            const auto x = matrix(n, d);
            if (task == "gbdt") {
                const auto y = vector<int>(n);
                for (int label : y)
                    if (label < 0 || label > 8) throw std::invalid_argument("GBDT training label");
                const auto test = matrix(m, d);
                assignment::GBDTClassifier model;
                model.fit(x, y, seed);
                const auto prediction = model.predict(test);
                if (prediction.size() != m) throw std::invalid_argument("GBDT prediction count");
                for (int label : prediction)
                    if (label < 0 || label > 8) throw std::invalid_argument("GBDT prediction label");
                output(prediction);
            } else {
                const auto y = vector<int>(n);
                const auto test = matrix(m, d);
                if (task == "lr") {
                    assignment::LogisticRegression model;
                    model.fit(x, y, seed);
                    output(model.predict(test));
                } else {
                    assignment::SVM model;
                    model.fit(x, y, seed);
                    output(model.predict(test));
                }
            }
        } else if (task == "kmeans") {
            const auto seed = read<std::uint64_t>();
            const auto n = read<std::size_t>();
            const auto d = read<std::size_t>();
            const auto m = read<std::size_t>();
            const auto k = read<std::size_t>();
            if (k == 0 || k > n) throw std::invalid_argument("cluster count");
            const auto x = matrix(n, d);
            const auto test = matrix(m, d);
            assignment::KMeans model;
            model.fit(x, k, seed);
            output(model.predict(test));
        } else if (task == "hmm") {
            const auto seed = read<std::uint64_t>();
            const auto states = read<std::size_t>();
            const auto symbols = read<std::size_t>();
            const auto training_count = read<std::size_t>();
            const auto test_count = read<std::size_t>();
            if (states == 0 || states > 128 || symbols == 0 || symbols > 256 ||
                training_count == 0 || training_count > 10000 || test_count > 10000)
                throw std::invalid_argument("HMM dimensions");
            auto sequences = [&](std::size_t count, std::size_t bound, bool nonempty) {
                std::vector<std::vector<int>> result;
                std::size_t total = 0;
                for (std::size_t i = 0; i < count; ++i) {
                    const auto length = read<std::size_t>();
                    if (length > 10000 || length > 65536-total || (nonempty && length == 0))
                        throw std::invalid_argument("sequence length");
                    total += length;
                    auto sequence = vector<int>(length);
                    for (int value : sequence)
                        if (value < 0 || static_cast<std::size_t>(value) >= bound)
                            throw std::invalid_argument("sequence value");
                    result.push_back(std::move(sequence));
                }
                return result;
            };
            const auto train = sequences(training_count, symbols, true);
            const auto labels = sequences(training_count, states, true);
            for (std::size_t i = 0; i < training_count; ++i)
                if (train[i].size() != labels[i].size())
                    throw std::invalid_argument("training label length");
            const auto observations = sequences(test_count, symbols, false);
            assignment::DiscreteHMM model;
            model.fit(train, labels, states, symbols, seed);
            const auto paths = model.predict(observations);
            std::cout << paths.size() << '\n';
            for (const auto& path : paths) output(path);
        } else {
            throw std::invalid_argument("unknown task");
        }
        std::string extra;
        if (std::cin >> extra) throw std::invalid_argument("trailing input");
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
