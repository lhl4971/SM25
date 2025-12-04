#include <iostream>
#include <fstream>
#include <cstdlib>
#include <sys/time.h>
#include "include/conjugate_gradient.hpp"
#include "include/timer.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

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
    Timer timer;
    timer.start("total_time");
    timer.start("init");

    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " M N" << std::endl;
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);

    std::ofstream debug_log("debug.log", std::ios::app);
    debug_log << "===== Test case M=" << M << ", N=" << N << " =====" << std::endl;

    PoissonSolver solver(
        M, N,
        0.0, 3.0, 0.0, 3.0,
        region, f_func, timer
    );
    timer.stop("init");
    solver.solve();
    timer.start("finalize");

    std::string filename = "solution/solution_M_" + std::to_string(M) + "_N_" + std::to_string(N) + ".csv";
    save_to_file(solver.get_solution(), filename);

    timer.stop("finalize");
    timer.stop("total_time");
    
    // Print timing summary
    std::cout << "\n========== Timing Summary ==========\n";
    std::cout << "Initialization time     : " << timer.get("init")      << " sec\n";
    std::cout << "Laplace operator time   : " << timer.get("laplace")   << " sec\n";
    std::cout << "Update operator time    : " << timer.get("update")    << " sec\n";
    std::cout << "Reduction time          : " << timer.get("reduction") << " sec\n";
    std::cout << "Finalize time           : " << timer.get("finalize")  << " sec\n";
    std::cout << "Total runtime           : " << timer.get("total_time")<< " sec\n";
    std::cout << "====================================\n\n";

    return 0;
}
