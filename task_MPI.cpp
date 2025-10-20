#include <mpi.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <cstdlib>
#include <vector>
#include <cassert>
#include <sys/time.h>
#include "include/mpi_conjugate_gradient.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

#define X_MIN 0.0
#define X_MAX 3.0
#define Y_MIN 0.0
#define Y_MAX 3.0

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
    std::cout << "Saved solution to " << filename << std::endl << std::endl;
}

bool region(double x, double y) {
    return (x > 0 && y > 0 && y < 3 && (y + 3 * x < 9));
}

double f_func(double x, double y) {
    return 1.0;
}

class DomainDecomposer {
private:
    int M, N, P;
    int px, py;

    std::vector<int> part_x;
    std::vector<int> part_y;

public:
    DomainDecomposer(
        int M_, int N_, int P_
    ): M(M_), N(N_), P(P_)
    {
        assert((P & (P - 1)) == 0 && "P must be power of 2!");
        block_division();
        part_x = divide_grid_1D(M, px);
        part_y = divide_grid_1D(N, py);
    }

    void block_division() {
        int log2p = std::round(std::log2(P));
        if (log2p & 1) { 
            if (M < N) { 
                px = 1 << (log2p / 2);
                py = 1 << (log2p / 2 + 1);
            } else { 
                px = 1 << (log2p / 2 + 1);
                py = 1 << (log2p / 2);
            }
        } else {
            px = 1 << (log2p / 2);
            py = 1 << (log2p / 2);
        }
    }

    std::vector<int> divide_grid_1D(int total_nodes, int blocks) {
        std::vector<int> parts = std::vector<int>(blocks, 0);
        int base = total_nodes / blocks;
        int remainder = total_nodes % blocks;

        for (int i = 1; i < blocks; ++i) {
            if (i < blocks - remainder)
                parts[i] = parts[i-1] + base;
            else
                parts[i] = parts[i-1] + base + 1;
        }
        return parts;
    }

    void print_info() const {
        std::cout << "==== Domain Decomposition Summary ====" << std::endl;
        std::cout << "Global grid: " << M << " x " << N << std::endl;
        std::cout << "Total processes: " << P << "    Grid layout: " << px << " x " << py << std::endl;

        std::cout << "X partitions (" << px << "): ";
        for (int i = 0; i < px; ++i)
            std::cout << std::setw(4) << part_x[i];
        std::cout << std::endl;

        std::cout << "Y partitions (" << py << "): ";
        for (int i = 0; i < py; ++i)
            std::cout << std::setw(4) << part_y[i];
        std::cout << std::endl;
        std::cout << "======================================" << std::endl;
    }

    int get_px() const { return px; }
    int get_py() const { return py; }
    std::vector<int> get_part_x() const { return part_x; }
    std::vector<int> get_part_y() const { return part_y; }
};

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int world_size = 0, world_rank = 0;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (argc < 3) {
        if (world_rank == 0) std::cerr << "Usage: " << argv[0] << " M N" << std::endl;
        MPI_Finalize();
        return 1;
    }
    
    int global_M = atoi(argv[1]);
    int global_N = atoi(argv[2]);

    DomainDecomposer decomp(global_M, global_N, world_size);
    if (world_rank == 0) decomp.print_info();

    int dims[2] = {decomp.get_px(), decomp.get_py()};
    int periods[2] = {0, 0};
    int reorder = 1;
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, reorder, &cart_comm);

    const auto& part_x = decomp.get_part_x();
    const auto& part_y = decomp.get_part_y();

    int coords[2];
    MPI_Cart_coords(cart_comm, world_rank, 2, coords);

    int x_start, x_end, y_start, y_end;
    x_start = part_x[coords[0]];
    x_end   = (coords[0] == dims[0] - 1) ? global_M : part_x[coords[0] + 1] - 1;
    y_start = part_y[coords[1]];
    y_end   = (coords[1] == dims[1] - 1) ? global_N : part_y[coords[1] + 1] - 1;
    
    std::ostringstream oss;
    oss << "[Rank " << std::setw(2) << world_rank << "] "
        << "coords=(" << coords[0] << "," << coords[1] << ") "
        << "x:[" << std::setw(4) << x_start << "," << std::setw(4) << x_end << "] "
        << "y:[" << std::setw(4) << y_start << "," << std::setw(4) << y_end << "] "
        << std::endl;

    for (int p = 0; p < world_size; ++p) {
        MPI_Barrier(MPI_COMM_WORLD);
        if (world_rank == p) std::cout << oss.str();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    
    int left, right, bottom, top;
    MPI_Cart_shift(cart_comm, 0, 1, &left, &right);
    MPI_Cart_shift(cart_comm, 1, 1, &bottom, &top);

    if (left   != MPI_PROC_NULL) x_start--;
    if (right  != MPI_PROC_NULL) x_end++;
    if (bottom != MPI_PROC_NULL) y_start--;
    if (top    != MPI_PROC_NULL) y_end++;

    if (world_rank == 0) {
        std::ofstream debug_log("debug.log", std::ios::app);
        debug_log << "===== Test case M=" << global_M << ", N=" << global_N << " =====" << std::endl;
    }

    double hx = (X_MAX - X_MIN) / global_M;
    double hy = (Y_MAX - Y_MIN) / global_N;

    double start = getCurrentTime();

    MPIPoissonSolver solver(
        x_end - x_start, y_end - y_start,
        X_MIN + hx * x_start, X_MIN + hx * x_end,
        Y_MIN + hy * y_start, Y_MIN + hy * y_end,
        region, f_func,
        world_rank, cart_comm
    );
    solver.solve();
    
    if (world_rank == 0)
        std::cout << "Total time: " << getCurrentTime() - start << " seconds.\n";

    struct Domain2D {
        int x_min, x_max, y_min, y_max, size;
    };
    Domain2D domain = {
        x_start + 1, x_end - 1,
        y_start + 1, y_end - 1,
        (x_end - x_start - 1) * (y_end - y_start - 1)
    };

    std::vector<Domain2D> domains;
    if (world_rank == 0) domains.resize(world_size);
    MPI_Gather(&domain, sizeof(Domain2D), MPI_BYTE, domains.data(), sizeof(Domain2D), MPI_BYTE, 0, cart_comm);

    // Get u_local and reshape to 1D
    const auto& u_local = solver.get_solution();
    int nx = u_local.size() - 2;
    int ny = u_local[0].size() - 2;

    std::vector<double> sendbuf(nx * ny);
    #pragma omp parallel for collapse(2)
    for (int i = 1; i <= nx; ++i)
        for (int j = 1; j <= ny; ++j)
            sendbuf[(i - 1) * ny + (j - 1)] = u_local[i][j];

    // Compute the data size and offset for all ranks
    std::vector<int> recv_counts, displs;
    int total_size = 0;
    if (world_rank == 0) {
        recv_counts.resize(world_size);
        displs.resize(world_size);
        for (int i = 0; i < world_size; ++i) {
            recv_counts[i] = domains[i].size;
            displs[i] = total_size;
            total_size += recv_counts[i];
        }
    }

    std::vector<double> recv_buf(total_size);
    MPI_Gatherv(sendbuf.data(), sendbuf.size(), MPI_DOUBLE,
    recv_buf.data(), recv_counts.data(), displs.data(), MPI_DOUBLE, 0, cart_comm);

    // Concate all data
    if (world_rank == 0) {
        std::vector<std::vector<double>> U(global_M + 1, std::vector<double>(global_N + 1, 0.0));

        int offset = 0;
        for (int p = 0; p < world_size; ++p) {
            domain = domains[p];
            int nx = domain.x_max - domain.x_min + 1;
            int ny = domain.y_max - domain.y_min + 1;

            #pragma omp parallel for schedule(static) collapse(2)
            for (int i = 0; i < nx; ++i)
                for (int j = 0; j < ny; ++j)
                    U[domain.x_min + i][domain.y_min + j] = recv_buf[offset + i * ny + j];

            offset += nx * ny;
        }

        std::string filename = "solution/solution_M_" + std::to_string(global_M) + "_N_" + std::to_string(global_N) + ".csv";
        save_to_file(U, filename);
    }

    MPI_Finalize();
    return 0;
};
