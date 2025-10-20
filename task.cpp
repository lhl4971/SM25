#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sys/time.h>
#include "bin/conjugate_gradient.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

double getCurrentTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void save_to_file(std::vector<std::vector<double>> mat, std::string filename) {
    std::ofstream file(filename.c_str());
    if (!file.is_open()) {
        std::cerr << "Error: cannot open " << filename << "\n";
        return;
    }

    int M = mat.size() - 1;
    int N = mat[0].size() - 1;

    for (int i = 0; i <= M; ++i) {
        for (int j = 0; j <= N; ++j) {
            file << mat[i][j];
            if (j < N) file << ",";
        }
        file << "\n";
    }
    file.close();
    std::cout << "Saved solution to " << filename << "\n";
}

bool region(double x, double y) {
    return (x > 0 && y > 0 && y < 3 && (y + 3 * x < 9));
}

double f_func(double x, double y) {
    return 1.0;
}

int main(int argc, char *argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " M N" << std::endl;
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);

    std::ofstream debug_log("debug.log", std::ios::app);
    debug_log << "===== Test case M=" << M << ", N=" << N << " =====" << std::endl;

    double start = getCurrentTime();

    PoissonSolver solver(
        M, N,
        0.0, 3.0, 0.0, 3.0,
        region, f_func
    );
    solver.solve();
    
    std::cout << "Total time: " << getCurrentTime() - start << " seconds.\n";
    
    std::string filename = "solution/solution_M_" + std::to_string(M) + "_N_" + std::to_string(N) + ".csv";
    save_to_file(solver.get_solution(), filename);

    return 0;
}