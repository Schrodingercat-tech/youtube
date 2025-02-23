#pragma once

#include <iostream>
#include <chrono>
#include "omp.h"
#include "thread"
#include "string"
#include <vector>
#include <stdexcept>
#include "random"


#define COMMON_H 


#ifdef COMMON_H

template<typename... Args>
void print(const Args&... args) {
    ((std::cout << args << " "), ...) << std::endl;  // Fold expression for variadic printing
}

// Specialized print function for 1D vectors
template<typename T>
void print(const std::vector<T>& vec) {
    for (const auto& element : vec) {
        std::cout << element << " ";
    }
    std::cout << std::endl;
}

// Specialized print function for N-dimensional vectors (recursive)
template<typename T>
void print(const std::vector<std::vector<T>>& vec) {
    for (const auto& sub_vec : vec) {
        print(sub_vec);  // Recursively print 1D vectors inside
    }
}

// Function to print 3D vector as an example (recursively)
template<typename T>
void print(const std::vector<std::vector<std::vector<T>>>& vec) {
    for (const auto& sub_vec2D : vec) {
        print(sub_vec2D);  // Recursively print 2D vectors
    }
}

// life time of a function
struct TimeIt {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point start;
    std::string message;

    TimeIt(std::string&& message = "") : message(message), start(Clock::now()) {}

    ~TimeIt() {
        auto end = Clock::now();
        auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        long long ns = duration_ns.count();

        if (!message.empty()) {
            print("\n", message);
        }

        if (ns >= 3600000000000) {
            std::cout << "\nTime elapsed: " << ns / 3600000000000.0 << " hours.\n";

        }
        else if (ns >= 60000000000) {
            std::cout << "\nTime elapsed: " << ns / 60000000000.0 << " minutes.\n";
        }
        else if (ns >= 1000000000) {
            std::cout << "\nTime elapsed: " << ns / 1000000000.0 << " seconds.\n";
        }
        else if (ns >= 1000000) {
            std::cout << "\nTime elapsed: " << ns / 1000000.0 << " milliseconds.\n";
        }
        else if (ns >= 1000) {
            std::cout << "\nTime elapsed: " << ns / 1000.0 << " microseconds.\n";
        }
        else {
            std::cout << "\nTime elapsed: " << ns << " nanoseconds.\n";
        }
    }


};

// Single random number generator
template <typename T>
float generateRandom(T min, T max) {
    thread_local std::mt19937 generator(std::random_device{}());
    std::uniform_real_distribution<float> distribution(static_cast<float>(min), static_cast<float>(max));
    return distribution(generator);
}

// Vector of random numbers generator (with different signature)
template <typename T>
std::vector<float> generateRandom(T min, T max, int range) {
    std::vector<float> result(range);
#pragma omp parallel for if(range > 100000)
    for (int i = 0; i < range; ++i) {
        result[i] = generateRandom(min, max);
    }

    return result;
}

#endif // COMMON_H
