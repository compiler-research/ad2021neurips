// bin/clang-9  -Xclang -load -Xclang ./etc/cling/plugins/lib/clad.so  -Xclang -add-plugin -Xclang clad -Xclang -plugin-arg-clad -Xclang -fdump-derived-fn   -I./etc/cling/plugins/include/ -lstdc++ benchmarking_gauss_exec.cpp  -o benchmarking_gauss_exec  -ldl -lrt -x c++ -std=c++11 -O0 -lm -ldl


#include<iostream>
#include<string.h>
#include <chrono>
using namespace std::chrono;
#include "./tools/cling/tools/plugins/clad/clad-prefix/src/clad/include/clad/Differentiator/Differentiator.h"

double sum(double* y, double* yh, int dim) { 
     double loss = 0.0;
     for (int i = 0; i < dim; i++) 
        loss += -1 * (y[i] * std::log(yh[i])) + (1 - y[i]) * std::log(1 - yh[i]);
     return loss; 
}

double exponential_pdf(double x, double lambda, double x0) {
    if ((x-x0) < 0)
      return 0.0;
    return lambda * std::exp (-lambda * (x-x0));
}

double* Clad(double* y, double* yh, int dim) {
    auto result_y = new double[dim]{};
    auto result_yh = new double[dim]{};
    auto sum_grad = clad::gradient(sum);
    auto start = high_resolution_clock::now();
    sum_grad.execute(y, yh, dim, result_y, result_yh);

    auto stop = high_resolution_clock::now();

    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Clad dimension: "<<dim<<" Clad duration: "<< duration.count() <<" microseconds "<<"result_y: " <<result_y[0]<< std::endl; 
    return result_y;
}

double* Numerical(double* p, int dim, double eps = 1e-8) {
     double* result = new double[dim]{};
     for (int i = 0; i < dim; i++) {
         double pi = p[i];
         p[i] = pi + eps;
         double v1 = sum(p, p, dim);
         p[i] = pi - eps;
         double v2 = sum(p,p, dim);
         result[i] = (v1 - v2)/(2 * eps);
         p[i] = pi;
     }
     std::cout << "Numerical dp:  " << result[0] << std::endl;
     return result;
}


double gaus(double* x, double* p, double sigma, int dim) {
   double t = 0;
   for (int i = 0; i< dim; i++)
       t += (x[i] - p[i]) * (x[i] - p[i]);
   t = -t / (2*sigma*sigma);
   return std::pow(2*M_PI, -dim/2.0) * std::pow(sigma, -0.5) * std::exp(t);
};

double* CladG(double* x, double* p, double sigma, int dim) {
    auto result_x = new double[dim]{};
    auto result_p = new double[dim]{};
    auto result_s = new double[dim]{};
    auto result_d = new double[dim]{};
    
    auto sum_grad = clad::gradient(gaus);
    auto start = high_resolution_clock::now();
    sum_grad.execute(x, p, sigma, dim, result_x, result_p, result_s, result_d);
    auto stop = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(stop - start);
    std::cout << "Clad dimension: "<<dim<<" Clad duration (microseconds): "<< duration.count() << std::endl;
    return result_x;
}

double* NumericalG(double* x, double* p, double sigma, int dim, double eps = 1e-8) {
    double* result = new double[dim]{};
    for (int i = 0; i < dim; i++) {
        double pi = p[i];
        p[i] = pi + eps;
        double v1 = gaus(x, p, sigma, dim);
        p[i] = pi - eps;
        double v2 = gaus(x, p, sigma, dim);
        result[i] = (v1 - v2)/(2 * eps);
        p[i] = pi;
    }
    std::cout << "NumericalG dx: " << result[0] << ' ' << "dy: " << result[1] << std::endl;

    return result;
}


int main(int argc, char *argv[]) {

    double dims[25] = {5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480, 40960, 81920, 163840, 327680, 655360, 1310720, 2621440, 5242880, 10485760, 20971520, 41943040, 83886080};
    double dim;
    for (int i = 0; i < 25; i++) {
        //dim = std::stod(argv[2]);
        dim = dims[i];
        double p[int(dim)], pp[int(dim)];
        for (int i = 0; i < dim; i++) {
            p[i] = i;
            pp[i] = i;
        }

       CladG(p,pp,0.5,dim);
//        NumericalG(p,pp,0.5,dim);
//        Numerical(p, dim);
//        Clad(p, pp, dim);
        
        //auto start2 = high_resolution_clock::now();
        //Numerical(p,p,0.5,dim);
        //auto stop2 = high_resolution_clock::now();
        //auto duration2 = duration_cast<nanoseconds>(stop2 - start2);
        //std::cout << "Numerical dimension: "<<dim<<" Numerical duration: "<< duration2.count() << std::endl;

    }
}
