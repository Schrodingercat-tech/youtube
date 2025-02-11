#include "functional"
#include "omp.h"
#include "thread"
#include "cmath"
#include "iostream"
#include "random"

// random number generator
template <typename T>
float generateRandom(T min, T max) {
	thread_local std::mt19937 generator(std::random_device{}()); // rng is generated in the same thread
	std::uniform_real_distribution<> distribution(min, max);
	return distribution(generator);
}


// Monte Carlo integration
float MonteCarloIntegration(
	std::function<float(float)> f, // some function
	const float a, const float b, // lower and upper bound
	const long N // samples to simulate
) {
	double  f_mean = 0.0;
	const float volume = (b - a); // 1D volume
	omp_set_num_threads(std::thread::hardware_concurrency()); // enable to manually set the number of threads
#pragma omp parallel for reduction(+:f_mean)  if(N>1000000)
	for (long i = 0; i < N; ++i) {
		float x = generateRandom(a, b); // a<=x<=b x is a random i.i.d 
		f_mean += f(x); // sum of f(x)

	}
	return volume * static_cast<float>(f_mean / N); // volume * f_mean 
}


// this is just example function to test the monte carlo integration

// Function to evaluate f(x) = (x^2 + 1) / (x^4 + 1)
float f(float x) {
	return (x * x + 1) / (x * x * x * x + 1);
}

// Analytical solution F(x) = (1 / sqrt(2)) * arctan((x - 1/x) / sqrt(2))
float analytical_solution(float x) {
	float u = x - 1 / x;
	return (1 / sqrt(2)) * atan(u / sqrt(2));
}

int main(){

	float a = -10.0, b = 10.0;
	long N; // 10000000
	std::cout << "Simulate N: ";
	std::cin >> N;
	float fun = MonteCarloIntegration(f, a, b, N);
	std::cout << "simulated value : " << fun << std::endl;
	std::cout << "Analytical solution: " << analytical_solution(b) - analytical_solution(a) << std::endl;
}