#include <cmath>
#include <iostream>
#include <chrono>

int main() {
    // Start the timer
    auto start = std::chrono::high_resolution_clock::now();

    std::cout << sqrt(42) << std::endl;

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    std::cout << "Execution time: " << duration << " ns" << std::endl;
}

