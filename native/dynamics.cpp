#define PY_SSIZE_T_CLEAN

#include <Python.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <complex>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

using cdouble = std::complex<double>;
using Matrix = std::vector<cdouble>;
constexpr double kPi = 3.141592653589793238462643383279502884;
// Pade exponentials materialize an n-by-n propagator and perform dense
// matrix products.  Above this bound a sparse RHS is both cheaper and more
// memory-stable for the high-level truncations this backend is meant to keep.
constexpr int kMaxDenseExponentialDimension = 256;

static_assert(sizeof(cdouble) == sizeof(double) * 2, "complex128 ABI mismatch");

bool finite_value(cdouble value) {
    return std::isfinite(value.real()) && std::isfinite(value.imag());
}

inline void sine_cosine(double angle, double& sine, double& cosine) {
#if defined(__linux__) && (defined(__GNUC__) || defined(__clang__))
    ::sincos(angle, &sine, &cosine);
#else
    sine = std::sin(angle);
    cosine = std::cos(angle);
#endif
}

inline cdouble exp_minus_i(cdouble phase) {
    if (phase.imag() == 0.0) {
        double sine = 0.0;
        double cosine = 0.0;
        sine_cosine(-phase.real(), sine, cosine);
        return cdouble(cosine, sine);
    }
    return std::exp(cdouble(0.0, -1.0) * phase);
}

inline cdouble exp_complex(cdouble value) {
    const double scale = std::exp(value.real());
    double sine = 0.0;
    double cosine = 0.0;
    sine_cosine(value.imag(), sine, cosine);
    return cdouble(scale * cosine, scale * sine);
}

struct BufferView {
    Py_buffer view{};
    bool acquired{false};

    ~BufferView() {
        if (acquired) {
            PyBuffer_Release(&view);
        }
    }

    bool acquire(PyObject* object, const char* name, int ndim, Py_ssize_t itemsize) {
        if (PyObject_GetBuffer(
                object,
                &view,
                PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES | PyBUF_C_CONTIGUOUS) < 0) {
            return false;
        }
        acquired = true;
        if (view.ndim != ndim || view.itemsize != itemsize || view.shape == nullptr) {
            PyErr_Format(
                PyExc_ValueError,
                "%s must be a contiguous %d-D array with itemsize %zd",
                name,
                ndim,
                itemsize);
            return false;
        }
        return true;
    }

    template <typename T>
    const T* data() const {
        return static_cast<const T*>(view.buf);
    }
};

struct SparseMatrixView {
    const cdouble* data{nullptr};
    const std::int64_t* indices{nullptr};
    const std::int64_t* indptr{nullptr};
    std::int64_t nnz{0};
    bool diagonal{false};
    bool diagonal_single{false};
};

struct OwnedSparseMatrix {
    std::vector<cdouble> data;
    std::vector<std::int64_t> indices;
    std::vector<std::int64_t> indptr;
    bool diagonal{false};
    bool diagonal_single{false};

    SparseMatrixView view() const {
        return {
            data.empty() ? nullptr : data.data(),
            indices.empty() ? nullptr : indices.data(),
            indptr.empty() ? nullptr : indptr.data(),
            static_cast<std::int64_t>(data.size()),
            diagonal,
            diagonal_single,
        };
    }
};

struct LindbladJumpEntry {
    int row{0};
    int column{0};
    int source_row{0};
    int source_column{0};
    cdouble value{0.0, 0.0};
};

struct LindbladJumpPattern {
    std::vector<LindbladJumpEntry> entries;
    std::vector<std::int64_t> row_ptr;
    bool ready{false};
};

// A banded operator stores one contiguous row vector for each exact diagonal
// offset.  This is the natural representation for oscillator/ladder
// Hamiltonians and avoids CSR column-index indirection in the hot loop.
struct BandedMatrixView {
    const cdouble* data{nullptr};
    const std::int64_t* offsets{nullptr};
    int band_count{0};
};

struct BandedEntry {
    int row{0};
    int column{0};
    int band{0};
    std::size_t value_index{0};
};

// A solve-local pool amortizes worker startup across all Lindblad RK stages.
// Tasks only touch disjoint output rows and execute without the Python GIL.
class RowWorkerPool {
public:
    explicit RowWorkerPool(int worker_count) : worker_count_(worker_count) {
        try {
            workers_.reserve(static_cast<std::size_t>(worker_count_));
            for (int index = 0; index < worker_count_; ++index) {
                workers_.emplace_back([this, index]() { worker_loop(index); });
            }
        } catch (...) {
            stop();
            throw;
        }
    }

    ~RowWorkerPool() { stop(); }
    RowWorkerPool(const RowWorkerPool&) = delete;
    RowWorkerPool& operator=(const RowWorkerPool&) = delete;

    int size() const { return worker_count_; }

    void execute(std::function<void(int)> task) {
        std::unique_lock<std::mutex> lock(mutex_);
        task_ = std::move(task);
        remaining_ = worker_count_;
        failure_ = nullptr;
        ++generation_;
        wake_.notify_all();
        completed_.wait(lock, [this]() { return remaining_ == 0; });
        task_ = nullptr;
        const std::exception_ptr failure = failure_;
        lock.unlock();
        if (failure) {
            std::rethrow_exception(failure);
        }
    }

private:
    void worker_loop(int index) {
        std::size_t observed_generation = 0;
        std::unique_lock<std::mutex> lock(mutex_);
        for (;;) {
            wake_.wait(lock, [this, observed_generation]() {
                return stopping_ || generation_ != observed_generation;
            });
            if (stopping_) {
                return;
            }
            observed_generation = generation_;
            lock.unlock();
            std::exception_ptr failure;
            try {
                task_(index);
            } catch (...) {
                failure = std::current_exception();
            }
            lock.lock();
            if (failure && !failure_) {
                failure_ = failure;
            }
            if (--remaining_ == 0) {
                completed_.notify_one();
            }
        }
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopping_ = true;
        }
        wake_.notify_all();
        for (std::thread& worker : workers_) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

    int worker_count_{1};
    int remaining_{0};
    bool stopping_{false};
    std::size_t generation_{0};
    std::vector<std::thread> workers_;
    std::mutex mutex_;
    std::condition_variable wake_;
    std::condition_variable completed_;
    std::function<void(int)> task_;
    std::exception_ptr failure_;
};

// The adaptive integrator reuses these buffers for every accepted and rejected
// step.  Keeping them outside `integrate_interval` avoids nine heap allocations
// per output interval, which is significant for long trace grids.
struct Workspace {
    std::vector<cdouble> hmat;
    std::vector<cdouble> scales;
    std::vector<cdouble> k1;
    std::vector<cdouble> k2;
    std::vector<cdouble> k3;
    std::vector<cdouble> k4;
    std::vector<cdouble> k5;
    std::vector<cdouble> k6;
    std::vector<cdouble> k7;
    std::vector<cdouble> trial;
    std::vector<cdouble> fourth;
    std::vector<cdouble> interaction_phases;
    std::vector<cdouble> interaction_row_phases;
    std::vector<cdouble> interaction_rotated_state;
    std::vector<cdouble> carrier_phases;
    std::unique_ptr<RowWorkerPool> row_workers;
    bool scales_valid{false};
    double scales_time{0.0};
    bool carrier_phase_valid{false};
    double carrier_phase_time{0.0};
    bool interaction_phase_valid{false};
    double interaction_phase_time{0.0};
    bool fsal_valid{false};
    double fsal_time{0.0};
    bool step_valid{false};
    double next_step{0.0};

    void resize(std::size_t state_size, std::size_t matrix_size) {
        k1.resize(state_size);
        k2.resize(state_size);
        k3.resize(state_size);
        k4.resize(state_size);
        k5.resize(state_size);
        k6.resize(state_size);
        k7.resize(state_size);
        trial.resize(state_size);
        fourth.resize(state_size);
        hmat.resize(matrix_size);
    }

    void resize_scales(std::size_t control_count) {
        scales.resize(control_count);
    }

    void resize_interaction_phases(std::size_t phase_count) {
        interaction_phases.resize(phase_count);
    }

    void resize_interaction_dense(std::size_t dimension, std::size_t state_size) {
        interaction_row_phases.resize(dimension);
        interaction_rotated_state.resize(state_size);
    }

    void resize_carrier_phases(std::size_t phase_count) {
        carrier_phases.resize(phase_count);
    }
};

// Reusable dense scratch for the small projected matrix exponential.  A
// Krylov worker owns one instance, avoiding repeated heap traffic while its
// Arnoldi basis grows from one checked dimension to the next.
struct MatrixExponentialWorkspace {
    Matrix a;
    Matrix a2;
    Matrix a4;
    Matrix a6;
    Matrix identity;
    Matrix tmp;
    Matrix u_inner;
    Matrix v;
    Matrix v_inner;
    Matrix u;
    Matrix lhs;
    Matrix rhs;
    Matrix squared;
};

// Scratch storage for one scalar Arnoldi action.  A worker owns one instance
// and reuses its allocations across all intervals and batch columns.
struct KrylovWorkspace {
    std::vector<cdouble> basis;
    std::vector<cdouble> hessenberg;
    std::vector<cdouble> work;
    std::vector<cdouble> previous;
    std::vector<cdouble> candidate;
    std::vector<cdouble> small_matrix;
    std::vector<cdouble> small_exponential;
    MatrixExponentialWorkspace exponential_workspace;

    void prepare(int n, int max_dimension) {
        // Every Arnoldi column is overwritten before it is read and every
        // Hessenberg entry used by the projected exponential is written while
        // that column is generated.  Resize without value-initialising the
        // large work arrays; this removes an O(n*m) memset for every action
        // while retaining the same arithmetic and residual checks.
        basis.resize(
            static_cast<std::size_t>(max_dimension + 1)
                * static_cast<std::size_t>(n));
        hessenberg.resize(
            static_cast<std::size_t>(max_dimension + 1)
                * static_cast<std::size_t>(max_dimension));
        // Entries below the active upper-Hessenberg band are copied into the
        // projected matrix while it grows, so they must be reset for every
        // action.  This is only O(m^2) (m <= 96), unlike clearing the O(n*m)
        // Arnoldi basis.
        std::fill(hessenberg.begin(), hessenberg.end(), cdouble(0.0, 0.0));
        work.resize(static_cast<std::size_t>(n));
        previous.resize(static_cast<std::size_t>(n));
        candidate.resize(static_cast<std::size_t>(n));
        small_matrix.reserve(static_cast<std::size_t>(max_dimension) * max_dimension);
        small_exponential.reserve(static_cast<std::size_t>(max_dimension) * max_dimension);
        const std::size_t projected_size =
            static_cast<std::size_t>(max_dimension) * max_dimension;
        exponential_workspace.a.reserve(projected_size);
        exponential_workspace.a2.reserve(projected_size);
        exponential_workspace.a4.reserve(projected_size);
        exponential_workspace.a6.reserve(projected_size);
        exponential_workspace.identity.reserve(projected_size);
        exponential_workspace.tmp.reserve(projected_size);
        exponential_workspace.u_inner.reserve(projected_size);
        exponential_workspace.v.reserve(projected_size);
        exponential_workspace.v_inner.reserve(projected_size);
        exponential_workspace.u.reserve(projected_size);
        exponential_workspace.lhs.reserve(projected_size);
        exponential_workspace.rhs.reserve(projected_size);
        exponential_workspace.squared.reserve(projected_size);
    }
};

struct Problem {
    const cdouble* h0{nullptr};
    const cdouble* controls{nullptr};
    const cdouble* iq{nullptr};
    const double* t_axis{nullptr};
    const double* lo_freqs{nullptr};
    const std::int64_t* control_modes{nullptr};
    const cdouble* iq_polynomial{nullptr};
    int n{0};
    int control_count{0};
    int sample_count{0};
    int batch_count{0};
    int mode{0};
    int coefficient_order{1};
    int polynomial_interval_count{0};
    double atol{1e-8};
    double rtol{1e-6};
    std::int64_t max_steps{2000000};
    std::int64_t attempted_steps{0};
    std::int64_t rhs_evaluations{0};
    std::int64_t exponential_evaluations{0};
    std::int64_t exponential_cache_hits{0};
    std::int64_t krylov_evaluations{0};
    std::int64_t krylov_iterations{0};
    bool krylov_used{false};
    bool piecewise_krylov_used{false};
    // 0=off, 1=automatic for high-dimensional sparse problems, 2=explicit.
    int sparse_expm_mode{1};
    // 0=off, 1=automatic, 2=explicit.  This is an execution policy only;
    // it never changes the represented differential equation.
    int parallel_mode{1};
    std::int64_t parallel_workers{1};
    std::int64_t diagonal_interval_evaluations{0};
    std::int64_t zero_interval_evaluations{0};
    bool whole_trace_exponential{false};
    bool interval_exponential{false};
    double max_error{0.0};
    double max_trial_error{0.0};
    double integration_seconds{0.0};
    double source_dt{0.0};
    double inv_source_dt{0.0};
    double t0{0.0};
    double t_last{0.0};
    // ``run_integration`` sets this before each output interval.  The public
    // native contract uses the same strictly increasing grid for traces and
    // output, so all RK stage times normally lie in this one source interval.
    // The fallback indexer below also handles arbitrary positive interval
    // widths for direct-extension callers.
    int source_interval{-1};
    double max_frequency{0.0};
    std::vector<double> angular_freqs;
    std::vector<double> unique_angular_freqs;
    std::vector<std::int64_t> angular_frequency_indices;
    // Per-interval value deltas are precomputed once for the source grid.  The
    // numerical interpolation remains exactly linear, but each RHS
    // call avoids a second complex subtraction.  ``fraction`` below is the
    // dimensionless position within the interval, so these are deltas rather
    // than derivatives (no extra 1/dt factor).
    std::vector<cdouble> slopes;
    bool sparse{false};
    bool banded{false};
    bool fused_sparse{false};
    bool interaction_picture{false};
    bool interaction_dense{false};
    bool lindblad{false};
    bool lindblad_diagonal_only{false};
    bool lindblad_diagonal_exact{false};
    int lindblad_collapse_count{0};
    std::vector<OwnedSparseMatrix> lindblad_collapse;
    std::vector<OwnedSparseMatrix> lindblad_products;
    std::vector<LindbladJumpPattern> lindblad_jump_patterns;
    std::vector<cdouble> lindblad_collapse_diagonal;
    std::vector<cdouble> lindblad_product_diagonal;
    // Immutable operator values prepared once for a globally constant sparse
    // Hamiltonian.  The entries retain the original term order (including
    // duplicate columns), so the cached action is mathematically identical to
    // the uncached h0-plus-controls traversal.
    OwnedSparseMatrix constant_sparse;
    std::vector<cdouble> constant_banded_values;
    std::vector<cdouble> constant_fused_values;
    bool constant_operator_ready{false};
    bool diagonal_only{false};
    bool h0_diagonal_only{false};
    std::vector<cdouble> h0_diagonal_values;
    std::vector<cdouble> diagonal_h0;
    std::vector<cdouble> diagonal_controls;
    SparseMatrixView h0_sparse;
    std::vector<SparseMatrixView> controls_sparse;
    const cdouble* h0_banded{nullptr};
    const cdouble* controls_banded{nullptr};
    const std::int64_t* band_offsets{nullptr};
    int band_count{0};
    std::vector<BandedEntry> banded_entries;
    std::vector<std::int64_t> banded_control_ptr;
    std::vector<int> banded_control_indices;
    const cdouble* fused_static{nullptr};
    const cdouble* fused_controls{nullptr};
    const std::int64_t* fused_indices{nullptr};
    const std::int64_t* fused_indptr{nullptr};
    std::int64_t fused_nnz{0};
    std::vector<std::int64_t> fused_control_ptr;
    std::vector<int> fused_control_indices;
    const cdouble* interaction_energies{nullptr};
    std::vector<cdouble> interaction_deltas;
    std::vector<double> interaction_unique_deltas;
    std::vector<std::int64_t> interaction_phase_indices;

    int mode_for_control(int control) const {
        if (control_modes != nullptr
            && control >= 0
            && control < control_count) {
            return static_cast<int>(control_modes[control]);
        }
        return mode == 2 ? 0 : mode;
    }

    bool has_rf_control() const {
        for (int control = 0; control < control_count; ++control) {
            if (mode_for_control(control) == 0) {
                return true;
            }
        }
        return false;
    }

    // The coherent kernels operate on ket vectors while the direct Lindblad
    // kernel operates on column-major density matrices.  Keeping this in one
    // helper lets the checked Krylov action reuse its workspace for either
    // physical state representation without constructing a Liouvillian.
    int state_dimension() const {
        if (lindblad) {
            const std::size_t dimension = static_cast<std::size_t>(n) * n;
            if (dimension > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                return 0;
            }
            return static_cast<int>(dimension);
        }
        return n;
    }

    void prepare_frequency_cache() {
        unique_angular_freqs.clear();
        angular_frequency_indices.clear();
        if (control_count <= 0 || angular_freqs.empty()) {
            return;
        }
        angular_frequency_indices.resize(static_cast<std::size_t>(control_count));
        std::unordered_map<double, std::int64_t> lookup;
        lookup.reserve(static_cast<std::size_t>(control_count));
        for (int control = 0; control < control_count; ++control) {
            const double frequency = angular_freqs[static_cast<std::size_t>(control)];
            const auto found = lookup.find(frequency);
            if (found == lookup.end()) {
                const std::int64_t index =
                    static_cast<std::int64_t>(unique_angular_freqs.size());
                lookup.emplace(frequency, index);
                unique_angular_freqs.push_back(frequency);
                angular_frequency_indices[static_cast<std::size_t>(control)] = index;
            } else {
                angular_frequency_indices[static_cast<std::size_t>(control)] = found->second;
            }
        }
    }

    void detect_diagonal_dense() {
        diagonal_only = true;
        h0_diagonal_only = true;
        h0_diagonal_values.assign(static_cast<std::size_t>(n), cdouble(0.0, 0.0));
        diagonal_h0.assign(static_cast<std::size_t>(n), cdouble(0.0, 0.0));
        for (int row = 0; row < n; ++row) {
            for (int column = 0; column < n; ++column) {
                const cdouble value = h0[static_cast<std::size_t>(row) * n + column];
                if (row == column) {
                    h0_diagonal_values[static_cast<std::size_t>(row)] = value;
                    diagonal_h0[static_cast<std::size_t>(row)] = value;
                } else if (value != cdouble(0.0, 0.0)) {
                    h0_diagonal_only = false;
                    diagonal_only = false;
                }
            }
        }
        diagonal_controls.assign(
            static_cast<std::size_t>(control_count) * n,
            cdouble(0.0, 0.0));
        for (int control = 0; control < control_count; ++control) {
            const cdouble* matrix = controls
                + static_cast<std::size_t>(control) * n * n;
            for (int row = 0; row < n; ++row) {
                for (int column = 0; column < n; ++column) {
                    const cdouble value = matrix[static_cast<std::size_t>(row) * n + column];
                    if (row == column) {
                        diagonal_controls[static_cast<std::size_t>(control) * n + row] = value;
                    } else if (value != cdouble(0.0, 0.0)) {
                        diagonal_only = false;
                    }
                }
            }
        }
        if (!diagonal_only) {
            diagonal_h0.clear();
            diagonal_controls.clear();
        }
        if (!h0_diagonal_only) {
            h0_diagonal_values.clear();
        }
    }

    static bool collect_diagonal_sparse(
        const SparseMatrixView& matrix,
        int dimension,
        cdouble* target) {
        for (int row = 0; row < dimension; ++row) {
            const std::int64_t begin = matrix.indptr[row];
            const std::int64_t end = matrix.indptr[row + 1];
            for (std::int64_t entry = begin; entry < end; ++entry) {
                const int column = static_cast<int>(matrix.indices[entry]);
                const cdouble value = matrix.data[entry];
                if (column == row) {
                    target[row] += value;
                } else if (value != cdouble(0.0, 0.0)) {
                    return false;
                }
            }
        }
        return true;
    }

    void detect_diagonal_sparse() {
        diagonal_only = true;
        h0_diagonal_only = true;
        h0_diagonal_values.assign(static_cast<std::size_t>(n), cdouble(0.0, 0.0));
        diagonal_h0.assign(static_cast<std::size_t>(n), cdouble(0.0, 0.0));
        if (!collect_diagonal_sparse(h0_sparse, n, h0_diagonal_values.data())) {
            h0_diagonal_only = false;
            diagonal_only = false;
        } else {
            diagonal_h0 = h0_diagonal_values;
        }
        diagonal_controls.assign(
            static_cast<std::size_t>(control_count) * n,
            cdouble(0.0, 0.0));
        for (int control = 0; control < control_count && diagonal_only; ++control) {
            if (!collect_diagonal_sparse(
                    controls_sparse[static_cast<std::size_t>(control)],
                    n,
                    diagonal_controls.data() + static_cast<std::size_t>(control) * n)) {
                diagonal_only = false;
            }
        }
        if (!diagonal_only) {
            diagonal_h0.clear();
            diagonal_controls.clear();
        }
        if (!h0_diagonal_only) {
            h0_diagonal_values.clear();
        }
    }

    void detect_lindblad_diagonal() {
        lindblad_diagonal_only = lindblad && diagonal_only;
        if (!lindblad_diagonal_only) {
            lindblad_collapse_diagonal.clear();
            lindblad_product_diagonal.clear();
            return;
        }
        for (const OwnedSparseMatrix& collapse : lindblad_collapse) {
            if (!collapse.diagonal) {
                lindblad_diagonal_only = false;
                break;
            }
        }
        if (!lindblad_diagonal_only) {
            lindblad_collapse_diagonal.clear();
            lindblad_product_diagonal.clear();
            return;
        }
        lindblad_collapse_diagonal.assign(
            static_cast<std::size_t>(lindblad_collapse_count) * n,
            cdouble(0.0, 0.0));
        lindblad_product_diagonal.assign(
            static_cast<std::size_t>(lindblad_collapse_count) * n,
            cdouble(0.0, 0.0));
        for (int collapse_index = 0;
             collapse_index < lindblad_collapse_count;
             ++collapse_index) {
            const OwnedSparseMatrix& collapse = lindblad_collapse[
                static_cast<std::size_t>(collapse_index)];
            const OwnedSparseMatrix& product = lindblad_products[
                static_cast<std::size_t>(collapse_index)];
            for (int row = 0; row < n; ++row) {
                for (std::int64_t entry = collapse.indptr[static_cast<std::size_t>(row)];
                     entry < collapse.indptr[static_cast<std::size_t>(row) + 1];
                     ++entry) {
                    lindblad_collapse_diagonal[
                        static_cast<std::size_t>(collapse_index) * n + row] +=
                        collapse.data[static_cast<std::size_t>(entry)];
                }
                for (std::int64_t entry = product.indptr[static_cast<std::size_t>(row)];
                     entry < product.indptr[static_cast<std::size_t>(row) + 1];
                     ++entry) {
                    lindblad_product_diagonal[
                        static_cast<std::size_t>(collapse_index) * n + row] +=
                        product.data[static_cast<std::size_t>(entry)];
                }
            }
        }
    }

    void detect_diagonal_banded() {
        diagonal_only = true;
        h0_diagonal_only = true;
        h0_diagonal_values.assign(static_cast<std::size_t>(n), cdouble(0.0, 0.0));
        diagonal_h0.assign(static_cast<std::size_t>(n), cdouble(0.0, 0.0));
        diagonal_controls.assign(
            static_cast<std::size_t>(control_count) * n,
            cdouble(0.0, 0.0));
        for (int band = 0; band < band_count && diagonal_only; ++band) {
            const std::int64_t offset = band_offsets[band];
            for (int row = 0; row < n; ++row) {
                const int column = row + static_cast<int>(offset);
                if (column < 0 || column >= n) {
                    continue;
                }
                const std::size_t band_index = static_cast<std::size_t>(band) * n + row;
                const cdouble static_value = h0_banded[band_index];
                if (offset == 0) {
                    h0_diagonal_values[static_cast<std::size_t>(row)] += static_value;
                    diagonal_h0[static_cast<std::size_t>(row)] += static_value;
                } else if (static_value != cdouble(0.0, 0.0)) {
                    h0_diagonal_only = false;
                    diagonal_only = false;
                    break;
                }
                for (int control = 0; control < control_count; ++control) {
                    const cdouble value = controls_banded[
                        (static_cast<std::size_t>(control) * band_count + band) * n + row];
                    if (offset == 0) {
                        diagonal_controls[static_cast<std::size_t>(control) * n + row] += value;
                    } else if (value != cdouble(0.0, 0.0)) {
                        diagonal_only = false;
                        break;
                    }
                }
            }
        }
        if (!diagonal_only) {
            diagonal_h0.clear();
            diagonal_controls.clear();
        }
        if (!h0_diagonal_only) {
            h0_diagonal_values.clear();
        }
    }

    void prepare_banded_entries() {
        banded_entries.clear();
        banded_control_ptr.clear();
        banded_control_indices.clear();
        if (!banded || band_count <= 0) {
            return;
        }
        banded_entries.reserve(static_cast<std::size_t>(band_count) * n);
        for (int band = 0; band < band_count; ++band) {
            const int offset = static_cast<int>(band_offsets[band]);
            for (int row = 0; row < n; ++row) {
                const int column = row + offset;
                if (column < 0 || column >= n) {
                    continue;
                }
                const std::size_t index = static_cast<std::size_t>(band) * n + row;
                bool active = h0_banded[index] != cdouble(0.0, 0.0);
                for (int control = 0; control < control_count && !active; ++control) {
                    active = controls_banded[
                        (static_cast<std::size_t>(control) * band_count + band) * n + row]
                        != cdouble(0.0, 0.0);
                }
                if (active) {
                    banded_entries.push_back({row, column, band, index});
                }
            }
        }
        banded_control_ptr.resize(banded_entries.size() + 1, 0);
        for (std::size_t entry = 0; entry < banded_entries.size(); ++entry) {
            const BandedEntry& item = banded_entries[entry];
            for (int control = 0; control < control_count; ++control) {
                const cdouble value = controls_banded[
                    (static_cast<std::size_t>(control) * band_count + item.band)
                        * n + item.row];
                if (value != cdouble(0.0, 0.0)) {
                    banded_control_indices.push_back(control);
                }
            }
            banded_control_ptr[entry + 1] =
                static_cast<std::int64_t>(banded_control_indices.size());
        }
    }

    void prepare_fused_control_entries() {
        fused_control_ptr.clear();
        fused_control_indices.clear();
        if (!fused_sparse || fused_nnz <= 0) {
            return;
        }
        fused_control_ptr.resize(static_cast<std::size_t>(fused_nnz) + 1, 0);
        for (std::int64_t entry = 0; entry < fused_nnz; ++entry) {
            for (int control = 0; control < control_count; ++control) {
                const cdouble value = fused_controls[
                    static_cast<std::size_t>(control) * fused_nnz + entry];
                if (value != cdouble(0.0, 0.0)) {
                    fused_control_indices.push_back(control);
                }
            }
            fused_control_ptr[static_cast<std::size_t>(entry) + 1] =
                static_cast<std::int64_t>(fused_control_indices.size());
        }
    }

    void prepare_interaction_deltas() {
        interaction_deltas.clear();
        interaction_unique_deltas.clear();
        interaction_phase_indices.clear();
        if (!interaction_picture || fused_nnz <= 0) {
            return;
        }
        interaction_deltas.resize(static_cast<std::size_t>(fused_nnz));
        interaction_phase_indices.resize(static_cast<std::size_t>(fused_nnz));
        std::unordered_map<double, std::int64_t> phase_lookup;
        phase_lookup.reserve(static_cast<std::size_t>(fused_nnz));
        for (int row = 0; row < n; ++row) {
            for (std::int64_t entry = fused_indptr[row];
                 entry < fused_indptr[row + 1]; ++entry) {
                const int column = static_cast<int>(fused_indices[entry]);
                const cdouble delta =
                    interaction_energies[row] - interaction_energies[column];
                interaction_deltas[static_cast<std::size_t>(entry)] = delta;
                // The Python preparation supplies real eigenvalues.  Keep the
                // real key so repeated transition frequencies share one phase
                // evaluation; a defensive complex fallback remains in the
                // RHS for direct extension callers.
                const double key = delta.real();
                const auto found = phase_lookup.find(key);
                if (found == phase_lookup.end()) {
                    const std::int64_t phase_index =
                        static_cast<std::int64_t>(interaction_unique_deltas.size());
                    phase_lookup.emplace(key, phase_index);
                    interaction_unique_deltas.push_back(key);
                    interaction_phase_indices[static_cast<std::size_t>(entry)] = phase_index;
                } else {
                    interaction_phase_indices[static_cast<std::size_t>(entry)] = found->second;
                }
            }
        }
    }

    bool detect_full_interaction_pattern() const {
        if (fused_nnz != static_cast<std::int64_t>(n) * n
            || fused_indptr == nullptr
            || fused_indices == nullptr) {
            return false;
        }
        for (int row = 0; row < n; ++row) {
            if (fused_indptr[row + 1] - fused_indptr[row] != n) {
                return false;
            }
            for (int column = 0; column < n; ++column) {
                const std::int64_t entry = fused_indptr[row] + column;
                if (fused_indices[entry] != column) {
                    return false;
                }
            }
        }
        return true;
    }

    void detect_diagonal_fused() {
        diagonal_only = true;
        h0_diagonal_only = true;
        h0_diagonal_values.assign(static_cast<std::size_t>(n), cdouble(0.0, 0.0));
        diagonal_h0.assign(static_cast<std::size_t>(n), cdouble(0.0, 0.0));
        diagonal_controls.assign(
            static_cast<std::size_t>(control_count) * n,
            cdouble(0.0, 0.0));
        for (int row = 0; row < n && diagonal_only; ++row) {
            const std::int64_t begin = fused_indptr[row];
            const std::int64_t end = fused_indptr[row + 1];
            for (std::int64_t entry = begin; entry < end; ++entry) {
                const int column = static_cast<int>(fused_indices[entry]);
                const cdouble static_value = fused_static[entry];
                if (column == row) {
                    h0_diagonal_values[static_cast<std::size_t>(row)] += static_value;
                    diagonal_h0[static_cast<std::size_t>(row)] += static_value;
                } else if (static_value != cdouble(0.0, 0.0)) {
                    h0_diagonal_only = false;
                    diagonal_only = false;
                    break;
                }
                for (int control = 0; control < control_count; ++control) {
                    const cdouble value = fused_controls[
                        static_cast<std::size_t>(control) * fused_nnz + entry];
                    if (column == row) {
                        diagonal_controls[static_cast<std::size_t>(control) * n + row] += value;
                    } else if (value != cdouble(0.0, 0.0)) {
                        diagonal_only = false;
                        break;
                    }
                }
            }
        }
        if (!diagonal_only) {
            diagonal_h0.clear();
            diagonal_controls.clear();
        }
        if (!h0_diagonal_only) {
            h0_diagonal_values.clear();
        }
    }

    void prepare_slopes() {
        slopes.clear();
        if (sample_count <= 1 || control_count <= 0) {
            return;
        }
        slopes.resize(
            static_cast<std::size_t>(control_count)
                * static_cast<std::size_t>(sample_count - 1));
        for (int control = 0; control < control_count; ++control) {
            for (int sample = 0; sample + 1 < sample_count; ++sample) {
                slopes[
                    static_cast<std::size_t>(control) * (sample_count - 1) + sample] =
                    iq[control * sample_count + sample + 1]
                        - iq[control * sample_count + sample];
            }
        }
    }

    bool has_constant_coefficients() const {
        if (sample_count <= 1 || control_count <= 0) {
            return true;
        }
        for (int control = 0; control < control_count; ++control) {
            const cdouble first = iq[static_cast<std::size_t>(control) * sample_count];
            const double angular_frequency = angular_freqs.empty()
                ? 2.0 * kPi * lo_freqs[control]
                : angular_freqs[static_cast<std::size_t>(control)];
            if (mode_for_control(control) == 1) {
                if (iq_polynomial != nullptr && polynomial_interval_count > 0) {
                    const std::size_t stride = static_cast<std::size_t>(coefficient_order + 1);
                    for (int interval = 0; interval < polynomial_interval_count; ++interval) {
                        const std::size_t base = (
                            static_cast<std::size_t>(control) * polynomial_interval_count
                            + static_cast<std::size_t>(interval)) * stride;
                        for (int power = 1; power <= coefficient_order; ++power) {
                            if (iq_polynomial[base + static_cast<std::size_t>(power)]
                                != cdouble(0.0, 0.0)) {
                                return false;
                            }
                        }
                    }
                }
                for (int sample = 1; sample < sample_count; ++sample) {
                    if (iq[static_cast<std::size_t>(control) * sample_count + sample]
                        != first) {
                        return false;
                    }
                }
            } else if (angular_frequency == 0.0) {
                if (iq_polynomial != nullptr && polynomial_interval_count > 0) {
                    const std::size_t stride = static_cast<std::size_t>(coefficient_order + 1);
                    for (int interval = 0; interval < polynomial_interval_count; ++interval) {
                        const std::size_t base = (
                            static_cast<std::size_t>(control) * polynomial_interval_count
                            + static_cast<std::size_t>(interval)) * stride;
                        for (int power = 1; power <= coefficient_order; ++power) {
                            if (iq_polynomial[base + static_cast<std::size_t>(power)]
                                != cdouble(0.0, 0.0)) {
                                return false;
                            }
                        }
                    }
                }
                for (int sample = 1; sample < sample_count; ++sample) {
                    if (iq[static_cast<std::size_t>(control) * sample_count + sample].real()
                        != first.real()) {
                        return false;
                    }
                }
            } else {
                // A non-zero carrier is constant only when the envelope is
                // identically zero; otherwise the RF coefficient oscillates.
                if (iq_polynomial != nullptr && polynomial_interval_count > 0) {
                    const std::size_t stride = static_cast<std::size_t>(coefficient_order + 1);
                    for (int interval = 0; interval < polynomial_interval_count; ++interval) {
                        const std::size_t base = (
                            static_cast<std::size_t>(control) * polynomial_interval_count
                            + static_cast<std::size_t>(interval)) * stride;
                        for (int power = 0; power <= coefficient_order; ++power) {
                            if (iq_polynomial[base + static_cast<std::size_t>(power)]
                                != cdouble(0.0, 0.0)) {
                                return false;
                            }
                        }
                    }
                } else {
                    for (int sample = 0; sample < sample_count; ++sample) {
                        if (iq[static_cast<std::size_t>(control) * sample_count + sample]
                            != cdouble(0.0, 0.0)) {
                            return false;
                        }
                    }
                }
            }
        }
        return true;
    }

    cdouble constant_coefficient(int control) const {
        const cdouble value = iq[static_cast<std::size_t>(control) * sample_count];
        if (mode_for_control(control) == 1) {
            return value;
        }
        const double angular_frequency = angular_freqs.empty()
            ? 2.0 * kPi * lo_freqs[control]
            : angular_freqs[static_cast<std::size_t>(control)];
        return angular_frequency == 0.0
            ? cdouble(value.real(), 0.0)
            : cdouble(0.0, 0.0);
    }

    bool interval_constant_scales(
        int sample,
        std::vector<cdouble>& scales_out) const {
        if (sample < 0 || sample + 1 >= sample_count) {
            return false;
        }
        scales_out.resize(static_cast<std::size_t>(control_count));
        for (int control = 0; control < control_count; ++control) {
            const std::size_t base =
                static_cast<std::size_t>(control) * sample_count;
            const cdouble left = iq[base + sample];
            const cdouble right = iq[base + sample + 1];
            const double angular_frequency = angular_freqs.empty()
                ? 2.0 * kPi * lo_freqs[control]
                : angular_freqs[static_cast<std::size_t>(control)];
            if (iq_polynomial != nullptr && polynomial_interval_count > 0) {
                const std::size_t stride = static_cast<std::size_t>(coefficient_order + 1);
                const std::size_t base = (
                    static_cast<std::size_t>(control) * polynomial_interval_count
                    + static_cast<std::size_t>(std::min(sample, polynomial_interval_count - 1)))
                    * stride;
                bool constant = true;
                for (int power = 1; power <= coefficient_order; ++power) {
                    if (iq_polynomial[base + static_cast<std::size_t>(power)]
                        != cdouble(0.0, 0.0)) {
                        constant = false;
                        break;
                    }
                }
                const cdouble value = iq_polynomial[base];
                if (mode_for_control(control) == 1) {
                    if (!constant) {
                        return false;
                    }
                    scales_out[static_cast<std::size_t>(control)] = value;
                } else if (value == cdouble(0.0, 0.0) && constant) {
                    scales_out[static_cast<std::size_t>(control)] = cdouble(0.0, 0.0);
                } else if (angular_frequency == 0.0 && constant) {
                    scales_out[static_cast<std::size_t>(control)] =
                        cdouble(value.real(), 0.0);
                } else {
                    return false;
                }
                continue;
            }
            if (mode_for_control(control) == 1) {
                if (left != right) {
                    return false;
                }
                scales_out[static_cast<std::size_t>(control)] = left;
            } else if (left == cdouble(0.0, 0.0)
                       && right == cdouble(0.0, 0.0)) {
                // A zero envelope is exactly zero even in the presence of a
                // carrier, so the interval Hamiltonian is constant.
                scales_out[static_cast<std::size_t>(control)] = cdouble(0.0, 0.0);
            } else if (angular_frequency == 0.0
                       && left.real() == right.real()) {
                scales_out[static_cast<std::size_t>(control)] =
                    cdouble(left.real(), 0.0);
            } else {
                return false;
            }
        }
        return true;
    }

    void build_hamiltonian_from_scales(
        const std::vector<cdouble>& scales,
        std::vector<cdouble>& matrix) const {
        matrix.assign(static_cast<std::size_t>(n) * n, cdouble(0.0, 0.0));
        if (!sparse && !banded && !fused_sparse) {
            std::copy(h0, h0 + static_cast<std::size_t>(n) * n, matrix.begin());
            for (int control = 0; control < control_count; ++control) {
                const cdouble* source = controls
                    + static_cast<std::size_t>(control) * n * n;
                const cdouble scale = scales[static_cast<std::size_t>(control)];
                if (scale == cdouble(0.0, 0.0)) {
                    continue;
                }
                for (int index = 0; index < n * n; ++index) {
                    matrix[static_cast<std::size_t>(index)] += scale * source[index];
                }
            }
            return;
        }
        if (sparse) {
            for (int row = 0; row < n; ++row) {
                for (std::int64_t entry = h0_sparse.indptr[row];
                     entry < h0_sparse.indptr[row + 1]; ++entry) {
                    const int column = static_cast<int>(h0_sparse.indices[entry]);
                    matrix[static_cast<std::size_t>(row) * n + column] +=
                        h0_sparse.data[entry];
                }
            }
            for (int control = 0; control < control_count; ++control) {
                const SparseMatrixView& source = controls_sparse[
                    static_cast<std::size_t>(control)];
                const cdouble scale = scales[static_cast<std::size_t>(control)];
                if (scale == cdouble(0.0, 0.0)) {
                    continue;
                }
                for (int row = 0; row < n; ++row) {
                    for (std::int64_t entry = source.indptr[row];
                         entry < source.indptr[row + 1]; ++entry) {
                        const int column = static_cast<int>(source.indices[entry]);
                        matrix[static_cast<std::size_t>(row) * n + column] +=
                            scale * source.data[entry];
                    }
                }
            }
            return;
        }
        if (banded) {
            for (const BandedEntry& entry : banded_entries) {
                const std::size_t target =
                    static_cast<std::size_t>(entry.row) * n + entry.column;
                matrix[target] += h0_banded[entry.value_index];
                for (int control = 0; control < control_count; ++control) {
                    const cdouble scale = scales[static_cast<std::size_t>(control)];
                    if (scale == cdouble(0.0, 0.0)) {
                        continue;
                    }
                    const std::size_t source =
                        (static_cast<std::size_t>(control) * band_count + entry.band)
                            * n + entry.row;
                    matrix[target] += scale * controls_banded[source];
                }
            }
            return;
        }
        for (int row = 0; row < n; ++row) {
            for (std::int64_t entry = fused_indptr[row];
                 entry < fused_indptr[row + 1]; ++entry) {
                const int column = static_cast<int>(fused_indices[entry]);
                cdouble value = fused_static[entry];
                for (int control = 0; control < control_count; ++control) {
                    const cdouble scale = scales[static_cast<std::size_t>(control)];
                    if (scale == cdouble(0.0, 0.0)) {
                        continue;
                    }
                    value += scale
                        * fused_controls[
                            static_cast<std::size_t>(control) * fused_nnz + entry];
                }
                matrix[static_cast<std::size_t>(row) * n + column] += value;
            }
        }
    }

    void build_constant_hamiltonian(std::vector<cdouble>& matrix) const {
        matrix.assign(static_cast<std::size_t>(n) * n, cdouble(0.0, 0.0));
        std::vector<cdouble> constant_scales(
            static_cast<std::size_t>(control_count), cdouble(0.0, 0.0));
        for (int control = 0; control < control_count; ++control) {
            constant_scales[static_cast<std::size_t>(control)] =
                constant_coefficient(control);
        }
        if (!sparse && !banded && !fused_sparse) {
            std::copy(
                h0,
                h0 + static_cast<std::size_t>(n) * n,
                matrix.begin());
            for (int control = 0; control < control_count; ++control) {
                const cdouble scale = constant_scales[static_cast<std::size_t>(control)];
                if (scale == cdouble(0.0, 0.0)) {
                    continue;
                }
                const cdouble* source = controls
                    + static_cast<std::size_t>(control) * n * n;
                for (int index = 0; index < n * n; ++index) {
                    matrix[static_cast<std::size_t>(index)] += scale * source[index];
                }
            }
            return;
        }
        if (sparse) {
            for (int row = 0; row < n; ++row) {
                for (std::int64_t entry = h0_sparse.indptr[row];
                     entry < h0_sparse.indptr[row + 1]; ++entry) {
                    const int column = static_cast<int>(h0_sparse.indices[entry]);
                    matrix[static_cast<std::size_t>(row) * n + column] +=
                        h0_sparse.data[entry];
                }
            }
            for (int control = 0; control < control_count; ++control) {
                const cdouble scale = constant_scales[static_cast<std::size_t>(control)];
                if (scale == cdouble(0.0, 0.0)) {
                    continue;
                }
                const SparseMatrixView& source = controls_sparse[
                    static_cast<std::size_t>(control)];
                for (int row = 0; row < n; ++row) {
                    for (std::int64_t entry = source.indptr[row];
                         entry < source.indptr[row + 1]; ++entry) {
                        const int column = static_cast<int>(source.indices[entry]);
                        matrix[static_cast<std::size_t>(row) * n + column] +=
                            scale * source.data[entry];
                    }
                }
            }
            return;
        }
        if (banded) {
            for (const BandedEntry& entry : banded_entries) {
                const std::size_t target =
                    static_cast<std::size_t>(entry.row) * n + entry.column;
                matrix[target] += h0_banded[entry.value_index];
                for (int control = 0; control < control_count; ++control) {
                    const cdouble scale = constant_scales[static_cast<std::size_t>(control)];
                    if (scale == cdouble(0.0, 0.0)) {
                        continue;
                    }
                    const std::size_t source =
                        (static_cast<std::size_t>(control) * band_count + entry.band)
                            * n + entry.row;
                    matrix[target] += scale * controls_banded[source];
                }
            }
            return;
        }
        for (int row = 0; row < n; ++row) {
            for (std::int64_t entry = fused_indptr[row];
                 entry < fused_indptr[row + 1]; ++entry) {
                const int column = static_cast<int>(fused_indices[entry]);
                cdouble value = fused_static[entry];
                for (int control = 0; control < control_count; ++control) {
                    const cdouble scale = constant_scales[static_cast<std::size_t>(control)];
                    if (scale == cdouble(0.0, 0.0)) {
                        continue;
                    }
                    value += scale
                        * fused_controls[
                            static_cast<std::size_t>(control) * fused_nnz + entry];
                }
                matrix[static_cast<std::size_t>(row) * n + column] += value;
            }
        }
    }

    void prepare_constant_operator(const std::vector<cdouble>& scales) {
        constant_operator_ready = false;
        constant_sparse = OwnedSparseMatrix();
        constant_banded_values.clear();
        constant_fused_values.clear();
        if (!(sparse || banded || fused_sparse)) {
            return;
        }
        if (sparse) {
            constant_sparse.indptr.resize(static_cast<std::size_t>(n) + 1);
            std::size_t estimated_nnz = static_cast<std::size_t>(h0_sparse.nnz);
            for (int control = 0; control < control_count; ++control) {
                if (scales[static_cast<std::size_t>(control)] != cdouble(0.0, 0.0)) {
                    estimated_nnz += static_cast<std::size_t>(
                        controls_sparse[static_cast<std::size_t>(control)].nnz);
                }
            }
            constant_sparse.data.reserve(estimated_nnz);
            constant_sparse.indices.reserve(estimated_nnz);
            for (int row = 0; row < n; ++row) {
                auto& data = constant_sparse.data;
                auto& indices = constant_sparse.indices;
                for (std::int64_t entry = h0_sparse.indptr[row];
                     entry < h0_sparse.indptr[row + 1]; ++entry) {
                    data.push_back(h0_sparse.data[entry]);
                    indices.push_back(h0_sparse.indices[entry]);
                }
                for (int control = 0; control < control_count; ++control) {
                    const cdouble scale = scales[static_cast<std::size_t>(control)];
                    if (scale == cdouble(0.0, 0.0)) {
                        continue;
                    }
                    const SparseMatrixView& source = controls_sparse[
                        static_cast<std::size_t>(control)];
                    for (std::int64_t entry = source.indptr[row];
                         entry < source.indptr[row + 1]; ++entry) {
                        data.push_back(scale * source.data[entry]);
                        indices.push_back(source.indices[entry]);
                    }
                }
                constant_sparse.indptr[static_cast<std::size_t>(row + 1)] =
                    static_cast<std::int64_t>(data.size());
            }
            constant_sparse.diagonal = false;
            constant_sparse.diagonal_single = false;
        } else if (banded) {
            constant_banded_values.resize(banded_entries.size(), cdouble(0.0, 0.0));
            for (std::size_t index = 0; index < banded_entries.size(); ++index) {
                const BandedEntry& entry = banded_entries[index];
                cdouble value = h0_banded[entry.value_index];
                for (int control = 0; control < control_count; ++control) {
                    const cdouble scale = scales[static_cast<std::size_t>(control)];
                    if (scale != cdouble(0.0, 0.0)) {
                        value += scale * controls_banded[
                            (static_cast<std::size_t>(control) * band_count + entry.band)
                                * n + entry.row];
                    }
                }
                constant_banded_values[index] = value;
            }
        } else {
            constant_fused_values.resize(static_cast<std::size_t>(fused_nnz));
            for (int row = 0; row < n; ++row) {
                for (std::int64_t entry = fused_indptr[row];
                     entry < fused_indptr[row + 1]; ++entry) {
                    cdouble value = fused_static[entry];
                    for (int control = 0; control < control_count; ++control) {
                        const cdouble scale = scales[static_cast<std::size_t>(control)];
                        if (scale != cdouble(0.0, 0.0)) {
                            value += scale * fused_controls[
                                static_cast<std::size_t>(control) * fused_nnz + entry];
                        }
                    }
                    constant_fused_values[static_cast<std::size_t>(entry)] = value;
                }
            }
        }
        constant_operator_ready = true;
    }

    // Apply the constant Hamiltonian to one vector without materialising a
    // dense matrix.  This is used by the high-dimensional Krylov exponential
    // action and preserves the exact CSR/banded/fused arithmetic order.
    void apply_constant_single(
        const cdouble* input,
        cdouble* output,
        const std::vector<cdouble>& scales) const {
        std::fill(output, output + n, cdouble(0.0, 0.0));
        if (constant_operator_ready) {
            if (sparse) {
                for (int row = 0; row < n; ++row) {
                    for (std::int64_t entry = constant_sparse.indptr[row];
                         entry < constant_sparse.indptr[row + 1]; ++entry) {
                        output[row] += constant_sparse.data[entry]
                            * input[static_cast<int>(constant_sparse.indices[entry])];
                    }
                }
            } else if (banded) {
                for (std::size_t index = 0; index < banded_entries.size(); ++index) {
                    const BandedEntry& entry = banded_entries[index];
                    output[entry.row] += constant_banded_values[index]
                        * input[entry.column];
                }
            } else if (fused_sparse) {
                for (int row = 0; row < n; ++row) {
                    for (std::int64_t entry = fused_indptr[row];
                         entry < fused_indptr[row + 1]; ++entry) {
                        output[row] += constant_fused_values[static_cast<std::size_t>(entry)]
                            * input[static_cast<int>(fused_indices[entry])];
                    }
                }
            }
            for (int row = 0; row < n; ++row) {
                output[row] *= cdouble(0.0, -1.0);
            }
            return;
        }
        if (!sparse && !banded && !fused_sparse) {
            for (int row = 0; row < n; ++row) {
                cdouble value(0.0, 0.0);
                for (int column = 0; column < n; ++column) {
                    const std::size_t matrix_index =
                        static_cast<std::size_t>(row) * n + column;
                    cdouble matrix_value = h0[matrix_index];
                    for (int control = 0; control < control_count; ++control) {
                        const cdouble scale = scales[static_cast<std::size_t>(control)];
                        if (scale != cdouble(0.0, 0.0)) {
                            matrix_value += scale * controls[
                                static_cast<std::size_t>(control) * n * n + matrix_index];
                        }
                    }
                    value += matrix_value * input[column];
                }
                output[row] = cdouble(0.0, -1.0) * value;
            }
            return;
        }
        if (sparse) {
            for (int row = 0; row < n; ++row) {
                for (std::int64_t entry = h0_sparse.indptr[row];
                     entry < h0_sparse.indptr[row + 1]; ++entry) {
                    output[row] += h0_sparse.data[entry]
                        * input[static_cast<int>(h0_sparse.indices[entry])];
                }
                for (int control = 0; control < control_count; ++control) {
                    const cdouble scale = scales[static_cast<std::size_t>(control)];
                    if (scale == cdouble(0.0, 0.0)) {
                        continue;
                    }
                    const SparseMatrixView& matrix = controls_sparse[
                        static_cast<std::size_t>(control)];
                    for (std::int64_t entry = matrix.indptr[row];
                         entry < matrix.indptr[row + 1]; ++entry) {
                        output[row] += scale * matrix.data[entry]
                            * input[static_cast<int>(matrix.indices[entry])];
                    }
                }
            }
        } else if (banded) {
            for (const BandedEntry& entry : banded_entries) {
                cdouble value = h0_banded[entry.value_index];
                for (int control = 0; control < control_count; ++control) {
                    const cdouble scale = scales[static_cast<std::size_t>(control)];
                    if (scale == cdouble(0.0, 0.0)) {
                        continue;
                    }
                    value += scale * controls_banded[
                        (static_cast<std::size_t>(control) * band_count + entry.band)
                            * n + entry.row];
                }
                output[entry.row] += value * input[entry.column];
            }
        } else {
            for (int row = 0; row < n; ++row) {
                for (std::int64_t entry = fused_indptr[row];
                     entry < fused_indptr[row + 1]; ++entry) {
                    cdouble value = fused_static[entry];
                    for (int control = 0; control < control_count; ++control) {
                        const cdouble scale = scales[static_cast<std::size_t>(control)];
                        if (scale != cdouble(0.0, 0.0)) {
                            value += scale * fused_controls[
                                static_cast<std::size_t>(control) * fused_nnz + entry];
                        }
                    }
                    output[row] += value
                        * input[static_cast<int>(fused_indices[entry])];
                }
            }
        }
        for (int row = 0; row < n; ++row) {
            output[row] *= cdouble(0.0, -1.0);
        }
    }

    // Apply a constant physical Lindblad generator to one vectorized density
    // matrix.  The storage is rho[row + column*n], matching the Python
    // column-major flattening.  No Kronecker product is formed and every
    // summation follows the same CSR row order as rhs_lindblad_rows().
    void apply_lindblad_constant_single(
        const cdouble* input,
        cdouble* output,
        const std::vector<cdouble>& scales) const {
        const std::size_t density_size = static_cast<std::size_t>(n) * n;
        std::fill(output, output + density_size, cdouble(0.0, 0.0));
        auto add_left = [&](const SparseMatrixView& matrix, cdouble coefficient) {
            if (coefficient == cdouble(0.0, 0.0)) {
                return;
            }
            for (int row = 0; row < n; ++row) {
                for (std::int64_t entry = matrix.indptr[row];
                     entry < matrix.indptr[row + 1]; ++entry) {
                    const int source_row = static_cast<int>(matrix.indices[entry]);
                    const cdouble value = coefficient * matrix.data[entry];
                    for (int column = 0; column < n; ++column) {
                        output[static_cast<std::size_t>(row)
                               + static_cast<std::size_t>(column) * n] += value
                            * input[static_cast<std::size_t>(source_row)
                                    + static_cast<std::size_t>(column) * n];
                    }
                }
            }
        };
        auto add_right_adjoint = [&](const SparseMatrixView& matrix, cdouble coefficient) {
            if (coefficient == cdouble(0.0, 0.0)) {
                return;
            }
            for (int column = 0; column < n; ++column) {
                for (std::int64_t entry = matrix.indptr[column];
                     entry < matrix.indptr[column + 1]; ++entry) {
                    const int source_column = static_cast<int>(matrix.indices[entry]);
                    const cdouble value = coefficient * std::conj(matrix.data[entry]);
                    for (int row = 0; row < n; ++row) {
                        output[static_cast<std::size_t>(row)
                               + static_cast<std::size_t>(column) * n] += value
                            * input[static_cast<std::size_t>(row)
                                    + static_cast<std::size_t>(source_column) * n];
                    }
                }
            }
        };
        auto add_jump = [&](const SparseMatrixView& collapse) {
            for (int row = 0; row < n; ++row) {
                for (int column = 0; column < n; ++column) {
                    const std::size_t target = static_cast<std::size_t>(row)
                        + static_cast<std::size_t>(column) * n;
                    for (std::int64_t left_entry = collapse.indptr[row];
                         left_entry < collapse.indptr[row + 1]; ++left_entry) {
                        const int source_row = static_cast<int>(collapse.indices[left_entry]);
                        const cdouble left_value = collapse.data[left_entry];
                        for (std::int64_t right_entry = collapse.indptr[column];
                             right_entry < collapse.indptr[column + 1]; ++right_entry) {
                            const int source_column = static_cast<int>(collapse.indices[right_entry]);
                            const cdouble value = left_value
                                * std::conj(collapse.data[right_entry]);
                            output[target] += value * input[
                                static_cast<std::size_t>(source_row)
                                + static_cast<std::size_t>(source_column) * n];
                        }
                    }
                }
            }
        };
        auto add_jump_pattern = [&](const LindbladJumpPattern& pattern) {
            if (!pattern.ready) {
                return;
            }
            for (const LindbladJumpEntry& item : pattern.entries) {
                output[static_cast<std::size_t>(item.row)
                       + static_cast<std::size_t>(item.column) * n] += item.value
                    * input[static_cast<std::size_t>(item.source_row)
                            + static_cast<std::size_t>(item.source_column) * n];
            }
        };

        const cdouble minus_i(0.0, -1.0);
        if (constant_operator_ready && sparse) {
            // The constant sparse bundle keeps static entries followed by
            // controls in their original order, so combining the two passes
            // removes repeated scale checks without changing accumulation.
            const SparseMatrixView matrix = constant_sparse.view();
            add_left(matrix, minus_i);
            add_right_adjoint(matrix, -minus_i);
        } else {
            add_left(h0_sparse, minus_i);
            add_right_adjoint(h0_sparse, -minus_i);
            for (int control = 0; control < control_count; ++control) {
                const cdouble scale = scales[static_cast<std::size_t>(control)];
                if (scale == cdouble(0.0, 0.0)) {
                    continue;
                }
                const SparseMatrixView& matrix = controls_sparse[static_cast<std::size_t>(control)];
                add_left(matrix, minus_i * scale);
                add_right_adjoint(matrix, minus_i * (-std::conj(scale)));
            }
        }
        for (std::size_t collapse_index = 0;
             collapse_index < lindblad_collapse.size(); ++collapse_index) {
            const LindbladJumpPattern& pattern = lindblad_jump_patterns[
                collapse_index];
            if (pattern.ready) {
                add_jump_pattern(pattern);
            } else {
                add_jump(lindblad_collapse[collapse_index].view());
            }
            add_left(lindblad_products[collapse_index].view(), cdouble(-0.5, 0.0));
            add_right_adjoint(lindblad_products[collapse_index].view(), cdouble(-0.5, 0.0));
        }
    }

    bool interpolation_coordinates(double t, int& left, double& fraction) const {
        if (t < t0 || t > t_last) {
            return false;
        }
        if (sample_count <= 1) {
            left = 0;
            fraction = 0.0;
            return true;
        }
        left = -1;
        fraction = 0.0;
        if (source_interval >= 0 && source_interval + 1 < sample_count) {
            const double interval_start = t_axis[source_interval];
            const double interval_end = t_axis[source_interval + 1];
            const double tolerance = 16.0 * std::numeric_limits<double>::epsilon()
                * std::max(1.0, std::abs(interval_end));
            if (t >= interval_start - tolerance && t <= interval_end + tolerance) {
                left = source_interval;
                const double interval_dt = interval_end - interval_start;
                fraction = interval_dt > 0.0
                    ? (t - interval_start) / interval_dt
                    : 0.0;
                fraction = std::min(1.0, std::max(0.0, fraction));
                // QuTiP's order-0 interpolator selects the most recent
                // sample at an exact knot.  The interval-local fast path is
                // otherwise allowed to reuse the preceding interval.
                if (iq_polynomial != nullptr
                    && coefficient_order == 0
                    && source_interval + 1 < sample_count - 1
                    && fraction >= 1.0) {
                    left = source_interval + 1;
                    fraction = 0.0;
                }
            }
        }
        if (left < 0) {
            // Use a real interval lookup here so nonuniform grids and direct
            // extension callers retain their exact local linear interpolation
            // semantics.
            const double* first = t_axis;
            const double* last = t_axis + sample_count;
            const double* upper = std::upper_bound(first, last, t);
            if (upper == first) {
                left = 0;
            } else if (upper == last) {
                left = sample_count - 1;
            } else {
                left = static_cast<int>(upper - first) - 1;
            }
            const int right = std::min(sample_count - 1, left + 1);
            if (right == left) {
                fraction = 0.0;
            } else {
                const double interval_dt = t_axis[right] - t_axis[left];
                fraction = interval_dt > 0.0
                    ? (t - t_axis[left]) / interval_dt
                    : 0.0;
                fraction = std::min(1.0, std::max(0.0, fraction));
            }
        }
        return true;
    }

    cdouble interpolated_value(int control, int left, double fraction) const {
        if (left >= sample_count - 1) {
            return iq[static_cast<std::size_t>(control) * sample_count
                       + static_cast<std::size_t>(sample_count - 1)];
        }
        const int right = std::min(sample_count - 1, left + 1);
        if (iq_polynomial != nullptr && polynomial_interval_count > 0) {
            const int order = std::max(0, coefficient_order);
            const std::size_t base = (
                static_cast<std::size_t>(control) * polynomial_interval_count
                + static_cast<std::size_t>(std::min(
                    left,
                    polynomial_interval_count - 1)))
                * static_cast<std::size_t>(order + 1);
            cdouble value = iq_polynomial[base + static_cast<std::size_t>(order)];
            for (int power = order - 1; power >= 0; --power) {
                value = value * fraction
                    + iq_polynomial[base + static_cast<std::size_t>(power)];
            }
            return value;
        }
        const cdouble left_value = iq[control * sample_count + left];
        cdouble value = left_value;
        if (right != left) {
            if (!slopes.empty()) {
                value += fraction * slopes[
                    static_cast<std::size_t>(control) * (sample_count - 1) + left];
            } else {
                const cdouble right_value = iq[control * sample_count + right];
                value += fraction * (right_value - left_value);
            }
        }
        return value;
    }

    cdouble physical_coefficient(int control, cdouble value, double t) const {
        if (mode_for_control(control) == 1) {
            return value;
        }
        const double angular_frequency = angular_freqs.empty()
            ? 2.0 * kPi * lo_freqs[control]
            : angular_freqs[static_cast<std::size_t>(control)];
        if (angular_frequency == 0.0) {
            return cdouble(value.real(), 0.0);
        }
        const double phase = angular_frequency * t;
        double sine = 0.0;
        double cosine = 0.0;
        sine_cosine(phase, sine, cosine);
        return value.real() * cosine - value.imag() * sine;
    }

    void fill_carrier_phases(double t, Workspace& workspace) const {
        if (!has_rf_control() || unique_angular_freqs.empty()) {
            return;
        }
        if (workspace.carrier_phase_valid && workspace.carrier_phase_time == t) {
            return;
        }
        for (std::size_t index = 0; index < unique_angular_freqs.size(); ++index) {
            const double frequency = unique_angular_freqs[index];
            double sine = 0.0;
            double cosine = 0.0;
            sine_cosine(frequency * t, sine, cosine);
            workspace.carrier_phases[index] = cdouble(cosine, sine);
        }
        workspace.carrier_phase_valid = true;
        workspace.carrier_phase_time = t;
    }

    cdouble physical_coefficient_with_phase(
        int control,
        cdouble value,
        const Workspace& workspace,
        double time) const {
        if (mode_for_control(control) == 1) {
            return value;
        }
        const double angular_frequency = angular_freqs.empty()
            ? 2.0 * kPi * lo_freqs[control]
            : angular_freqs[static_cast<std::size_t>(control)];
        if (angular_frequency == 0.0) {
            return cdouble(value.real(), 0.0);
        }
        if (!workspace.carrier_phases.empty()
            && static_cast<std::size_t>(control) < angular_frequency_indices.size()) {
            const std::size_t phase_index = static_cast<std::size_t>(
                angular_frequency_indices[static_cast<std::size_t>(control)]);
            if (phase_index < workspace.carrier_phases.size()) {
                const cdouble phase = workspace.carrier_phases[phase_index];
                return value.real() * phase.real() - value.imag() * phase.imag();
            }
        }
        return physical_coefficient(control, value, time);
    }

    void fill_scales(double t, Workspace& workspace) const {
        if (workspace.scales_valid && workspace.scales_time == t) {
            return;
        }
        int left = 0;
        double fraction = 0.0;
        if (!interpolation_coordinates(t, left, fraction)) {
            std::fill(workspace.scales.begin(), workspace.scales.end(), cdouble(0.0, 0.0));
            workspace.scales_valid = true;
            workspace.scales_time = t;
            return;
        }
        fill_carrier_phases(t, workspace);
        if (sample_count == 1) {
            for (int control = 0; control < control_count; ++control) {
                workspace.scales[static_cast<std::size_t>(control)] = physical_coefficient_with_phase(
                    control,
                    iq[static_cast<std::size_t>(control) * sample_count],
                    workspace,
                    t);
            }
            workspace.scales_valid = true;
            workspace.scales_time = t;
            return;
        }
        for (int control = 0; control < control_count; ++control) {
            workspace.scales[static_cast<std::size_t>(control)] = physical_coefficient_with_phase(
                control,
                interpolated_value(control, left, fraction),
                workspace,
                t);
        }
        workspace.scales_valid = true;
        workspace.scales_time = t;
    }

    cdouble coefficient(int control, double t) const {
        int left = 0;
        double fraction = 0.0;
        if (!interpolation_coordinates(t, left, fraction)) {
            return cdouble(0.0, 0.0);
        }
        const cdouble value = sample_count == 1
            ? iq[static_cast<std::size_t>(control) * sample_count]
            : interpolated_value(control, left, fraction);
        return physical_coefficient(control, value, t);
    }

    void build_hamiltonian(double t, Workspace& workspace) const {
        std::copy(h0, h0 + static_cast<std::size_t>(n) * n, workspace.hmat.begin());
        fill_scales(t, workspace);
        for (int control = 0; control < control_count; ++control) {
            const cdouble scale = workspace.scales[static_cast<std::size_t>(control)];
            if (scale == cdouble(0.0, 0.0)) {
                continue;
            }
            const cdouble* operator_data = controls + static_cast<std::size_t>(control) * n * n;
            for (int index = 0; index < n * n; ++index) {
                workspace.hmat[index] += scale * operator_data[index];
            }
        }
    }

    void add_sparse_matrix(
        const SparseMatrixView& matrix,
        cdouble scale,
        const std::vector<cdouble>& state,
        std::vector<cdouble>& derivative) const {
        if (scale == cdouble(0.0, 0.0)) {
            return;
        }
        if (matrix.diagonal_single) {
            for (int row = 0; row < n; ++row) {
                const cdouble value = scale * matrix.data[row];
                for (int batch = 0; batch < batch_count; ++batch) {
                    derivative[row * batch_count + batch] +=
                        value * state[row * batch_count + batch];
                }
            }
            return;
        }
        if (matrix.diagonal) {
            for (int row = 0; row < n; ++row) {
                const std::int64_t begin = matrix.indptr[row];
                const std::int64_t end = matrix.indptr[row + 1];
                for (std::int64_t entry = begin; entry < end; ++entry) {
                    const cdouble value = scale * matrix.data[entry];
                    for (int batch = 0; batch < batch_count; ++batch) {
                        derivative[row * batch_count + batch] +=
                            value * state[row * batch_count + batch];
                    }
                }
            }
            return;
        }
        for (int row = 0; row < n; ++row) {
            const std::int64_t begin = matrix.indptr[row];
            const std::int64_t end = matrix.indptr[row + 1];
            for (std::int64_t entry = begin; entry < end; ++entry) {
                const int column = static_cast<int>(matrix.indices[entry]);
                const cdouble value = scale * matrix.data[entry];
                for (int batch = 0; batch < batch_count; ++batch) {
                    derivative[row * batch_count + batch] +=
                        value * state[column * batch_count + batch];
                }
            }
        }
    }

    void rhs_banded(
        double t,
        const std::vector<cdouble>& state,
        std::vector<cdouble>& derivative,
        Workspace& workspace) {
        std::fill(derivative.begin(), derivative.end(), cdouble(0.0, 0.0));
        fill_scales(t, workspace);

        // Fuse H0 and all controls at each exact band position.  A state
        // element is loaded once for the complete time-dependent coefficient,
        // instead of once per operator as in the generic CSR path.
        for (std::size_t entry_index = 0;
             entry_index < banded_entries.size();
             ++entry_index) {
            const BandedEntry& entry = banded_entries[entry_index];
            cdouble value = h0_banded[entry.value_index];
            if (!banded_control_ptr.empty()) {
                const std::int64_t begin = banded_control_ptr[
                    entry_index];
                const std::int64_t end = banded_control_ptr[
                    entry_index + 1];
                for (std::int64_t active = begin; active < end; ++active) {
                    const int control = banded_control_indices[
                        static_cast<std::size_t>(active)];
                    const cdouble scale = workspace.scales[
                        static_cast<std::size_t>(control)];
                    if (scale == cdouble(0.0, 0.0)) {
                        continue;
                    }
                    const std::size_t control_index =
                        (static_cast<std::size_t>(control) * band_count + entry.band)
                            * n + entry.row;
                    value += scale * controls_banded[control_index];
                }
            } else {
                for (int control = 0; control < control_count; ++control) {
                    const cdouble scale = workspace.scales[
                        static_cast<std::size_t>(control)];
                    if (scale == cdouble(0.0, 0.0)) {
                        continue;
                    }
                    const std::size_t control_index =
                        (static_cast<std::size_t>(control) * band_count + entry.band)
                            * n + entry.row;
                    value += scale * controls_banded[control_index];
                }
            }
            if (value == cdouble(0.0, 0.0)) {
                continue;
            }
            for (int batch = 0; batch < batch_count; ++batch) {
                derivative[entry.row * batch_count + batch] +=
                    value * state[entry.column * batch_count + batch];
            }
        }
        for (cdouble& value : derivative) {
            value *= cdouble(0.0, -1.0);
        }
    }

    void rhs_fused_sparse(
        double t,
        const std::vector<cdouble>& state,
        std::vector<cdouble>& derivative,
        Workspace& workspace) {
        std::fill(derivative.begin(), derivative.end(), cdouble(0.0, 0.0));
        fill_scales(t, workspace);

        // The union CSR pattern is built once during preparation.  Each
        // nonzero position is visited once, while the static and control
        // contributions are accumulated in their original deterministic order.
        for (int row = 0; row < n; ++row) {
            const std::int64_t begin = fused_indptr[row];
            const std::int64_t end = fused_indptr[row + 1];
            for (std::int64_t entry = begin; entry < end; ++entry) {
                cdouble value = fused_static[entry];
                if (!fused_control_ptr.empty()) {
                    const std::int64_t active_begin = fused_control_ptr[
                        static_cast<std::size_t>(entry)];
                    const std::int64_t active_end = fused_control_ptr[
                        static_cast<std::size_t>(entry) + 1];
                    for (std::int64_t active = active_begin; active < active_end; ++active) {
                        const int control = fused_control_indices[
                            static_cast<std::size_t>(active)];
                        const cdouble scale = workspace.scales[
                            static_cast<std::size_t>(control)];
                        if (scale == cdouble(0.0, 0.0)) {
                            continue;
                        }
                        value += scale
                            * fused_controls[
                                static_cast<std::size_t>(control) * fused_nnz + entry];
                    }
                } else {
                    for (int control = 0; control < control_count; ++control) {
                        const cdouble scale = workspace.scales[
                            static_cast<std::size_t>(control)];
                        if (scale == cdouble(0.0, 0.0)) {
                            continue;
                        }
                        value += scale
                            * fused_controls[
                                static_cast<std::size_t>(control) * fused_nnz + entry];
                    }
                }
                if (value == cdouble(0.0, 0.0)) {
                    continue;
                }
                const int column = static_cast<int>(fused_indices[entry]);
                for (int batch = 0; batch < batch_count; ++batch) {
                    derivative[row * batch_count + batch] +=
                        value * state[column * batch_count + batch];
                }
            }
        }
        for (cdouble& value : derivative) {
            value *= cdouble(0.0, -1.0);
        }
    }

    void rhs_interaction_dense(
        double t,
        const std::vector<cdouble>& state,
        std::vector<cdouble>& derivative,
        Workspace& workspace) {
        std::fill(derivative.begin(), derivative.end(), cdouble(0.0, 0.0));
        fill_scales(t, workspace);
        const double elapsed = t - t0;
        // H_I[row, column] = exp(+i E_row tau) H[row, column]
        // exp(-i E_column tau).  Apply the two diagonal factors around the
        // dense control matvec so only 2*n trigonometric evaluations are
        // needed, rather than one per matrix entry.
        if (!workspace.interaction_phase_valid
            || workspace.interaction_phase_time != t) {
            for (int level = 0; level < n; ++level) {
                const double angle = interaction_energies[level].real() * elapsed;
                double sine = 0.0;
                double cosine = 0.0;
                sine_cosine(angle, sine, cosine);
                workspace.interaction_row_phases[static_cast<std::size_t>(level)] =
                    cdouble(cosine, sine);
            }
            workspace.interaction_phase_valid = true;
            workspace.interaction_phase_time = t;
        }
        for (int column = 0; column < n; ++column) {
            const cdouble phase = std::conj(
                workspace.interaction_row_phases[static_cast<std::size_t>(column)]);
            for (int batch = 0; batch < batch_count; ++batch) {
                workspace.interaction_rotated_state[
                    static_cast<std::size_t>(column) * batch_count + batch] = phase
                    * state[static_cast<std::size_t>(column) * batch_count + batch];
            }
        }
        for (int row = 0; row < n; ++row) {
            const cdouble row_phase =
                workspace.interaction_row_phases[static_cast<std::size_t>(row)];
            const std::int64_t row_begin = fused_indptr[row];
            for (int column = 0; column < n; ++column) {
                const std::int64_t entry = row_begin + column;
                cdouble matrix_value(0.0, 0.0);
                for (int control = 0; control < control_count; ++control) {
                    const cdouble scale = workspace.scales[
                        static_cast<std::size_t>(control)];
                    if (scale == cdouble(0.0, 0.0)) {
                        continue;
                    }
                    matrix_value += scale
                        * fused_controls[
                            static_cast<std::size_t>(control) * fused_nnz + entry];
                }
                for (int batch = 0; batch < batch_count; ++batch) {
                    derivative[static_cast<std::size_t>(row) * batch_count + batch] +=
                        matrix_value * workspace.interaction_rotated_state[
                            static_cast<std::size_t>(column) * batch_count + batch];
                }
            }
            for (int batch = 0; batch < batch_count; ++batch) {
                derivative[static_cast<std::size_t>(row) * batch_count + batch] *=
                    cdouble(0.0, -1.0) * row_phase;
            }
        }
    }

    void rhs_interaction_fused_sparse(
        double t,
        const std::vector<cdouble>& state,
        std::vector<cdouble>& derivative,
        Workspace& workspace) {
        std::fill(derivative.begin(), derivative.end(), cdouble(0.0, 0.0));
        fill_scales(t, workspace);
        const double elapsed = t - t0;
        if (!workspace.interaction_phase_valid
            || workspace.interaction_phase_time != t) {
            for (std::size_t phase_index = 0;
                 phase_index < interaction_unique_deltas.size();
                 ++phase_index) {
                const double angle = interaction_unique_deltas[phase_index] * elapsed;
                double sine = 0.0;
                double cosine = 0.0;
                sine_cosine(angle, sine, cosine);
                workspace.interaction_phases[phase_index] = cdouble(cosine, sine);
            }
            workspace.interaction_phase_valid = true;
            workspace.interaction_phase_time = t;
        }
        for (int row = 0; row < n; ++row) {
            const std::int64_t begin = fused_indptr[row];
            const std::int64_t end = fused_indptr[row + 1];
            for (std::int64_t entry = begin; entry < end; ++entry) {
                cdouble value(0.0, 0.0);
                if (!fused_control_ptr.empty()) {
                    const std::int64_t active_begin = fused_control_ptr[
                        static_cast<std::size_t>(entry)];
                    const std::int64_t active_end = fused_control_ptr[
                        static_cast<std::size_t>(entry) + 1];
                    for (std::int64_t active = active_begin; active < active_end; ++active) {
                        const int control = fused_control_indices[
                            static_cast<std::size_t>(active)];
                        const cdouble scale = workspace.scales[
                            static_cast<std::size_t>(control)];
                        if (scale == cdouble(0.0, 0.0)) {
                            continue;
                        }
                        value += scale
                            * fused_controls[
                                static_cast<std::size_t>(control) * fused_nnz + entry];
                    }
                } else {
                    for (int control = 0; control < control_count; ++control) {
                        const cdouble scale = workspace.scales[
                            static_cast<std::size_t>(control)];
                        if (scale == cdouble(0.0, 0.0)) {
                            continue;
                        }
                        value += scale
                            * fused_controls[
                                static_cast<std::size_t>(control) * fused_nnz + entry];
                    }
                }
                if (value == cdouble(0.0, 0.0)) {
                    continue;
                }
                const cdouble delta = interaction_deltas[
                    static_cast<std::size_t>(entry)];
                if (delta != cdouble(0.0, 0.0)) {
                    const std::int64_t phase_index = interaction_phase_indices[
                        static_cast<std::size_t>(entry)];
                    if (delta.imag() == 0.0
                        && phase_index >= 0
                        && static_cast<std::size_t>(phase_index)
                            < workspace.interaction_phases.size()) {
                        value *= workspace.interaction_phases[
                            static_cast<std::size_t>(phase_index)];
                    } else {
                        value *= std::exp(cdouble(0.0, 1.0) * delta * elapsed);
                    }
                }
                const int column = static_cast<int>(fused_indices[entry]);
                for (int batch = 0; batch < batch_count; ++batch) {
                    derivative[row * batch_count + batch] +=
                        value * state[column * batch_count + batch];
                }
            }
        }
        for (cdouble& value : derivative) {
            value *= cdouble(0.0, -1.0);
        }
    }

    // The matrix-free Lindblad path keeps rho as an n-by-n column-major
    // matrix (with batch columns interleaved in the final dimension).  These
    // helpers apply sparse operators from the left or right without forming a
    // Kronecker-product Liouvillian of dimension n^2 by n^2.
    void add_lindblad_left(
        const SparseMatrixView& operator_view,
        const std::vector<cdouble>& state,
        std::vector<cdouble>& derivative,
        cdouble coefficient,
        int row_begin = 0,
        int row_end = -1) const {
        if (coefficient == cdouble(0.0, 0.0)) {
            return;
        }
        row_end = row_end < 0 ? n : row_end;
        const std::size_t batch_stride = static_cast<std::size_t>(batch_count);
        for (int row = row_begin; row < row_end; ++row) {
            for (std::int64_t entry = operator_view.indptr[row];
                 entry < operator_view.indptr[row + 1]; ++entry) {
                const int source_row =
                    static_cast<int>(operator_view.indices[entry]);
                const cdouble value = coefficient * operator_view.data[entry];
                for (int column = 0; column < n; ++column) {
                    const std::size_t target_base =
                        (static_cast<std::size_t>(row)
                         + static_cast<std::size_t>(column) * n) * batch_stride;
                    const std::size_t source_base =
                        (static_cast<std::size_t>(source_row)
                         + static_cast<std::size_t>(column) * n) * batch_stride;
                    for (std::size_t batch = 0; batch < batch_stride; ++batch) {
                        derivative[target_base + batch] +=
                            value * state[source_base + batch];
                    }
                }
            }
        }
    }

    void add_lindblad_right_adjoint(
        const SparseMatrixView& operator_view,
        const std::vector<cdouble>& state,
        std::vector<cdouble>& derivative,
        cdouble coefficient,
        int row_begin = 0,
        int row_end = -1) const {
        if (coefficient == cdouble(0.0, 0.0)) {
            return;
        }
        row_end = row_end < 0 ? n : row_end;
        const std::size_t batch_stride = static_cast<std::size_t>(batch_count);
        for (int column = 0; column < n; ++column) {
            for (std::int64_t entry = operator_view.indptr[column];
                 entry < operator_view.indptr[column + 1]; ++entry) {
                const int source_column =
                    static_cast<int>(operator_view.indices[entry]);
                const cdouble value = coefficient
                    * std::conj(operator_view.data[entry]);
                for (int row = row_begin; row < row_end; ++row) {
                    const std::size_t target_base =
                        (static_cast<std::size_t>(row)
                         + static_cast<std::size_t>(column) * n) * batch_stride;
                    const std::size_t source_base =
                        (static_cast<std::size_t>(row)
                         + static_cast<std::size_t>(source_column) * n)
                        * batch_stride;
                    for (std::size_t batch = 0; batch < batch_stride; ++batch) {
                        derivative[target_base + batch] +=
                            value * state[source_base + batch];
                    }
                }
            }
        }
    }

    void add_lindblad_jump(
        const SparseMatrixView& collapse,
        const std::vector<cdouble>& state,
        std::vector<cdouble>& derivative,
        int row_begin = 0,
        int row_end = -1) const {
        row_end = row_end < 0 ? n : row_end;
        const std::size_t batch_stride = static_cast<std::size_t>(batch_count);
        for (int row = row_begin; row < row_end; ++row) {
            for (int column = 0; column < n; ++column) {
                const std::size_t target_base =
                    (static_cast<std::size_t>(row)
                     + static_cast<std::size_t>(column) * n) * batch_stride;
                for (std::int64_t left_entry = collapse.indptr[row];
                     left_entry < collapse.indptr[row + 1]; ++left_entry) {
                    const int source_row =
                        static_cast<int>(collapse.indices[left_entry]);
                    const cdouble left_value = collapse.data[left_entry];
                    for (std::int64_t right_entry = collapse.indptr[column];
                         right_entry < collapse.indptr[column + 1];
                         ++right_entry) {
                        const int source_column =
                            static_cast<int>(collapse.indices[right_entry]);
                        const cdouble value =
                            left_value * std::conj(collapse.data[right_entry]);
                        const std::size_t source_base =
                            (static_cast<std::size_t>(source_row)
                             + static_cast<std::size_t>(source_column) * n)
                            * batch_stride;
                        for (std::size_t batch = 0; batch < batch_stride; ++batch) {
                            derivative[target_base + batch] +=
                                value * state[source_base + batch];
                        }
                    }
                }
            }
        }
    }

    void add_lindblad_jump(
        const LindbladJumpPattern& pattern,
        const std::vector<cdouble>& state,
        std::vector<cdouble>& derivative,
        int row_begin = 0,
        int row_end = -1) const {
        if (!pattern.ready) {
            return;
        }
        row_end = row_end < 0 ? n : row_end;
        const std::size_t batch_stride = static_cast<std::size_t>(batch_count);
        for (int row = row_begin; row < row_end; ++row) {
            const std::int64_t begin = pattern.row_ptr[static_cast<std::size_t>(row)];
            const std::int64_t end = pattern.row_ptr[static_cast<std::size_t>(row) + 1];
            for (std::int64_t entry = begin; entry < end; ++entry) {
                const LindbladJumpEntry& item = pattern.entries[
                    static_cast<std::size_t>(entry)];
                const std::size_t target_base = (
                    static_cast<std::size_t>(item.row)
                    + static_cast<std::size_t>(item.column) * n) * batch_stride;
                const std::size_t source_base = (
                    static_cast<std::size_t>(item.source_row)
                    + static_cast<std::size_t>(item.source_column) * n) * batch_stride;
                for (std::size_t batch = 0; batch < batch_stride; ++batch) {
                    derivative[target_base + batch] += item.value
                        * state[source_base + batch];
                }
            }
        }
    }

    void rhs_lindblad_rows(
        const std::vector<cdouble>& state,
        std::vector<cdouble>& derivative,
        const Workspace& workspace,
        int row_begin,
        int row_end) const {
        const cdouble minus_i(0.0, -1.0);

        // Directly accumulate -i(H rho - rho H^dagger).
        add_lindblad_left(
            h0_sparse, state, derivative, minus_i, row_begin, row_end);
        add_lindblad_right_adjoint(
            h0_sparse, state, derivative, -minus_i, row_begin, row_end);
        for (int control = 0; control < control_count; ++control) {
            const cdouble scale = workspace.scales[
                static_cast<std::size_t>(control)];
            if (scale == cdouble(0.0, 0.0)) {
                continue;
            }
            const SparseMatrixView& operator_view = controls_sparse[
                static_cast<std::size_t>(control)];
            add_lindblad_left(
                operator_view,
                state,
                derivative,
                minus_i * scale,
                row_begin,
                row_end);
            add_lindblad_right_adjoint(
                operator_view,
                state,
                derivative,
                minus_i * (-std::conj(scale)),
                row_begin,
                row_end);
        }

        // C rho C^dagger - 1/2 {C^dagger C, rho}; products are precomputed
        // once during parsing and all three contributions stream into the
        // same output buffer.
        for (std::size_t collapse_index = 0;
             collapse_index < lindblad_collapse.size();
             ++collapse_index) {
            const SparseMatrixView collapse =
                lindblad_collapse[collapse_index].view();
            const SparseMatrixView product =
                lindblad_products[collapse_index].view();
            const LindbladJumpPattern& jump_pattern = lindblad_jump_patterns[
                collapse_index];
            if (jump_pattern.ready) {
                add_lindblad_jump(
                    jump_pattern, state, derivative, row_begin, row_end);
            } else {
                add_lindblad_jump(
                    collapse, state, derivative, row_begin, row_end);
            }
            add_lindblad_left(
                product,
                state,
                derivative,
                cdouble(-0.5, 0.0),
                row_begin,
                row_end);
            add_lindblad_right_adjoint(
                product,
                state,
                derivative,
                cdouble(-0.5, 0.0),
                row_begin,
                row_end);
        }
    }

    void rhs_lindblad(
        double t,
        const std::vector<cdouble>& state,
        std::vector<cdouble>& derivative,
        Workspace& workspace) {
        std::fill(derivative.begin(), derivative.end(), cdouble(0.0, 0.0));
        fill_scales(t, workspace);

        int worker_count = 1;
        const std::size_t work_size =
            static_cast<std::size_t>(n) * n * batch_count;
        if (parallel_mode != 0
            && batch_count > 1
            && n >= 64
            && work_size >= 4096
            && (parallel_mode == 2 || batch_count >= 16)) {
            unsigned int hardware = std::thread::hardware_concurrency();
            if (hardware == 0) {
                hardware = 2;
            }
            worker_count = std::min(
                std::min(batch_count, static_cast<int>(hardware)),
                8);
        }
        parallel_workers = std::max(
            parallel_workers,
            static_cast<std::int64_t>(worker_count));
        if (worker_count == 1) {
            rhs_lindblad_rows(state, derivative, workspace, 0, n);
            return;
        }

        const int rows_per_worker = (n + worker_count - 1) / worker_count;
        if (workspace.row_workers == nullptr
            || workspace.row_workers->size() != worker_count) {
            workspace.row_workers = std::make_unique<RowWorkerPool>(worker_count);
        }
        workspace.row_workers->execute(
            [this, &state, &derivative, &workspace, rows_per_worker](int worker_index) {
                const int row_begin = std::min(n, worker_index * rows_per_worker);
                const int row_end = std::min(n, row_begin + rows_per_worker);
                    rhs_lindblad_rows(
                        state, derivative, workspace, row_begin, row_end);
            });
    }


    void rhs(
        double t,
        const std::vector<cdouble>& state,
        std::vector<cdouble>& derivative,
        Workspace& workspace) {
        ++rhs_evaluations;
        if (lindblad) {
            rhs_lindblad(t, state, derivative, workspace);
            return;
        }
        if (interaction_picture) {
            if (interaction_dense) {
                rhs_interaction_dense(t, state, derivative, workspace);
            } else {
                rhs_interaction_fused_sparse(t, state, derivative, workspace);
            }
            return;
        }
        if (fused_sparse) {
            rhs_fused_sparse(t, state, derivative, workspace);
            return;
        }
        if (banded) {
            rhs_banded(t, state, derivative, workspace);
            return;
        }
        if (sparse) {
            std::fill(derivative.begin(), derivative.end(), cdouble(0.0, 0.0));
            add_sparse_matrix(h0_sparse, cdouble(1.0, 0.0), state, derivative);
            fill_scales(t, workspace);
            for (int control = 0; control < control_count; ++control) {
                const cdouble scale = workspace.scales[
                    static_cast<std::size_t>(control)];
                if (scale == cdouble(0.0, 0.0)) {
                    continue;
                }
                add_sparse_matrix(
                    controls_sparse[static_cast<std::size_t>(control)],
                    scale,
                    state,
                    derivative);
            }
            for (cdouble& value : derivative) {
                value *= cdouble(0.0, -1.0);
            }
            return;
        }

        if (batch_count == 1) {
            // A single state does not benefit from materialising H(t).  Apply
            // each fixed operator directly and keep only one scalar per drive
            // in the reusable workspace.
            fill_scales(t, workspace);
            for (int row = 0; row < n; ++row) {
                cdouble value(0.0, 0.0);
                for (int column = 0; column < n; ++column) {
                    const std::size_t matrix_index = static_cast<std::size_t>(row) * n + column;
                    cdouble matrix_value = h0[matrix_index];
                    for (int control = 0; control < control_count; ++control) {
                        const cdouble scale = workspace.scales[
                            static_cast<std::size_t>(control)];
                        if (scale == cdouble(0.0, 0.0)) {
                            continue;
                        }
                        matrix_value += scale
                            * controls[static_cast<std::size_t>(control) * n * n + matrix_index];
                    }
                    value += matrix_value * state[column];
                }
                derivative[row] = cdouble(0.0, -1.0) * value;
            }
            return;
        }

        build_hamiltonian(t, workspace);
        for (int row = 0; row < n; ++row) {
            for (int batch = 0; batch < batch_count; ++batch) {
                cdouble value(0.0, 0.0);
                for (int column = 0; column < n; ++column) {
                    value += workspace.hmat[row * n + column] * state[column * batch_count + batch];
                }
                derivative[row * batch_count + batch] = cdouble(0.0, -1.0) * value;
            }
        }
    }
};

void matrix_identity(int n, Matrix& result) {
    result.resize(static_cast<std::size_t>(n) * n);
    std::fill(result.begin(), result.end(), cdouble(0.0, 0.0));
    for (int index = 0; index < n; ++index) {
        result[static_cast<std::size_t>(index) * n + index] = cdouble(1.0, 0.0);
    }
}

void matrix_add_scaled(
    const Matrix& left,
    const Matrix& right,
    cdouble right_scale,
    Matrix& result) {
    result.resize(left.size());
    for (std::size_t index = 0; index < left.size(); ++index) {
        result[index] = left[index] + right_scale * right[index];
    }
}

void matrix_multiply(const Matrix& left, const Matrix& right, int n, Matrix& result) {
    result.resize(static_cast<std::size_t>(n) * n);
    std::fill(result.begin(), result.end(), cdouble(0.0, 0.0));
    for (int row = 0; row < n; ++row) {
        for (int pivot = 0; pivot < n; ++pivot) {
            const cdouble value = left[static_cast<std::size_t>(row) * n + pivot];
            if (value == cdouble(0.0, 0.0)) {
                continue;
            }
            for (int column = 0; column < n; ++column) {
                result[static_cast<std::size_t>(row) * n + column] += value
                    * right[static_cast<std::size_t>(pivot) * n + column];
            }
        }
    }
}

double matrix_one_norm(const Matrix& matrix, int n) {
    double norm = 0.0;
    for (int column = 0; column < n; ++column) {
        double sum = 0.0;
        for (int row = 0; row < n; ++row) {
            sum += std::abs(matrix[static_cast<std::size_t>(row) * n + column]);
        }
        norm = std::max(norm, sum);
    }
    return norm;
}

bool solve_linear_system(Matrix lhs, Matrix& rhs, int n) {
    // Gaussian elimination with partial pivoting solves all RHS columns at
    // once.  The matrices are small (the constant path is intended for the
    // low-dimensional gate Hamiltonians), so this avoids a heavyweight BLAS
    // dependency while retaining the Pade algorithm's numerical stability.
    for (int pivot = 0; pivot < n; ++pivot) {
        int pivot_row = pivot;
        double pivot_abs = std::abs(lhs[static_cast<std::size_t>(pivot) * n + pivot]);
        for (int row = pivot + 1; row < n; ++row) {
            const double candidate = std::abs(lhs[static_cast<std::size_t>(row) * n + pivot]);
            if (candidate > pivot_abs) {
                pivot_abs = candidate;
                pivot_row = row;
            }
        }
        if (!(pivot_abs > std::numeric_limits<double>::epsilon())) {
            return false;
        }
        if (pivot_row != pivot) {
            for (int column = pivot; column < n; ++column) {
                std::swap(
                    lhs[static_cast<std::size_t>(pivot) * n + column],
                    lhs[static_cast<std::size_t>(pivot_row) * n + column]);
            }
            for (int column = 0; column < n; ++column) {
                std::swap(
                    rhs[static_cast<std::size_t>(pivot) * n + column],
                    rhs[static_cast<std::size_t>(pivot_row) * n + column]);
            }
        }
        const cdouble diagonal = lhs[static_cast<std::size_t>(pivot) * n + pivot];
        for (int row = pivot + 1; row < n; ++row) {
            const cdouble factor = lhs[static_cast<std::size_t>(row) * n + pivot] / diagonal;
            if (factor == cdouble(0.0, 0.0)) {
                continue;
            }
            lhs[static_cast<std::size_t>(row) * n + pivot] = cdouble(0.0, 0.0);
            for (int column = pivot + 1; column < n; ++column) {
                lhs[static_cast<std::size_t>(row) * n + column] -= factor
                    * lhs[static_cast<std::size_t>(pivot) * n + column];
            }
            for (int column = 0; column < n; ++column) {
                rhs[static_cast<std::size_t>(row) * n + column] -= factor
                    * rhs[static_cast<std::size_t>(pivot) * n + column];
            }
        }
    }
    for (int pivot = n - 1; pivot >= 0; --pivot) {
        const cdouble diagonal = lhs[static_cast<std::size_t>(pivot) * n + pivot];
        for (int column = 0; column < n; ++column) {
            cdouble value = rhs[static_cast<std::size_t>(pivot) * n + column];
            for (int next = pivot + 1; next < n; ++next) {
                value -= lhs[static_cast<std::size_t>(pivot) * n + next]
                    * rhs[static_cast<std::size_t>(next) * n + column];
            }
            rhs[static_cast<std::size_t>(pivot) * n + column] = value / diagonal;
        }
    }
    return true;
}

bool matrix_exponential(
    const Matrix& input,
    int n,
    Matrix& result,
    MatrixExponentialWorkspace* reusable = nullptr) {
    if (n < 1 || input.size() != static_cast<std::size_t>(n) * n) {
        return false;
    }
    const double norm = matrix_one_norm(input, n);
    if (!std::isfinite(norm)) {
        return false;
    }
    if (norm == 0.0) {
        matrix_identity(n, result);
        return true;
    }

    // Higham's theta for the [13/13] Pade approximant.
    constexpr double theta13 = 5.371920351148152;
    int squarings = 0;
    if (norm > theta13) {
        squarings = static_cast<int>(std::ceil(std::log2(norm / theta13)));
        if (squarings < 0) {
            squarings = 0;
        }
    }
    if (squarings > 1024) {
        return false;
    }
    const double scale = std::ldexp(1.0, -squarings);
    MatrixExponentialWorkspace local_workspace;
    MatrixExponentialWorkspace& workspace = reusable != nullptr
        ? *reusable
        : local_workspace;
    Matrix& a = workspace.a;
    Matrix& a2 = workspace.a2;
    Matrix& a4 = workspace.a4;
    Matrix& a6 = workspace.a6;
    Matrix& identity = workspace.identity;
    Matrix& tmp = workspace.tmp;
    Matrix& u_inner = workspace.u_inner;
    Matrix& v = workspace.v;
    Matrix& v_inner = workspace.v_inner;
    Matrix& u = workspace.u;
    Matrix& lhs = workspace.lhs;
    Matrix& rhs = workspace.rhs;
    Matrix& squared = workspace.squared;
    a.resize(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        a[index] = input[index] * scale;
    }
    matrix_multiply(a, a, n, a2);
    matrix_multiply(a2, a2, n, a4);
    matrix_multiply(a4, a2, n, a6);

    constexpr double b0 = 64764752532480000.0;
    constexpr double b1 = 32382376266240000.0;
    constexpr double b2 = 7771770303897600.0;
    constexpr double b3 = 1187353796428800.0;
    constexpr double b4 = 129060195264000.0;
    constexpr double b5 = 10559470521600.0;
    constexpr double b6 = 670442572800.0;
    constexpr double b7 = 33522128640.0;
    constexpr double b8 = 1323241920.0;
    constexpr double b9 = 40840800.0;
    constexpr double b10 = 960960.0;
    constexpr double b11 = 16380.0;
    constexpr double b12 = 182.0;
    constexpr double b13 = 1.0;

    matrix_identity(n, identity);
    u_inner.resize(a6.size());
    for (std::size_t index = 0; index < a6.size(); ++index) {
        u_inner[index] = b13 * a6[index] + b11 * a4[index] + b9 * a2[index];
    }
    matrix_multiply(a6, u_inner, n, tmp);
    for (std::size_t index = 0; index < tmp.size(); ++index) {
        tmp[index] += b7 * a6[index] + b5 * a4[index] + b3 * a2[index]
            + b1 * identity[index];
    }
    matrix_multiply(a, tmp, n, u);

    v_inner.resize(a6.size());
    for (std::size_t index = 0; index < v_inner.size(); ++index) {
        v_inner[index] = b12 * a6[index] + b10 * a4[index] + b8 * a2[index];
    }
    matrix_multiply(a6, v_inner, n, v);
    for (std::size_t index = 0; index < v.size(); ++index) {
        v[index] += b6 * a6[index] + b4 * a4[index] + b2 * a2[index]
            + b0 * identity[index];
    }
    matrix_add_scaled(v, u, cdouble(-1.0, 0.0), lhs);
    matrix_add_scaled(v, u, cdouble(1.0, 0.0), rhs);
    if (!solve_linear_system(lhs, rhs, n)) {
        return false;
    }
    // All callers pass an output vector distinct from the workspace RHS, so
    // swap avoids copying the projected matrix before the squaring phase.
    result.swap(rhs);
    for (int index = 0; index < squarings; ++index) {
        matrix_multiply(result, result, n, squared);
        result.swap(squared);
    }
    for (const cdouble value : result) {
        if (!finite_value(value)) {
            return false;
        }
    }
    return true;
}

double vector_two_norm(const std::vector<cdouble>& values) {
    long double sum = 0.0L;
    for (const cdouble value : values) {
        const long double magnitude = static_cast<long double>(std::abs(value));
        sum += magnitude * magnitude;
    }
    return std::sqrt(static_cast<double>(sum));
}

cdouble vector_inner_product(
    const cdouble* left,
    const cdouble* right,
    int size) {
    cdouble result(0.0, 0.0);
    for (int index = 0; index < size; ++index) {
        result += std::conj(left[index]) * right[index];
    }
    return result;
}

// Compute exp(-i H dt) v with an Arnoldi basis generated exclusively through
// the sparse operator action.  Acceptance requires both the standard Arnoldi
// residual estimate and agreement with the preceding Krylov dimension.  If
// either certificate cannot be met within the bounded basis, the caller
// falls back to the adaptive RK path rather than returning an unchecked result.
bool krylov_exponential_action(
    Problem& problem,
    const std::vector<cdouble>& input,
    const std::vector<cdouble>& scales,
    double dt,
    double tolerance,
    int max_dimension,
    std::vector<cdouble>& output,
    int& used_dimension,
    double& residual,
    KrylovWorkspace* reusable_workspace = nullptr) {
    const int n = problem.state_dimension();
    output.assign(static_cast<std::size_t>(std::max(0, n)), cdouble(0.0, 0.0));
    used_dimension = 0;
    residual = std::numeric_limits<double>::infinity();
    if (n < 1 || input.size() != static_cast<std::size_t>(n)
        || !std::isfinite(dt)
        || !std::isfinite(tolerance)
        || tolerance <= 0.0
        || max_dimension < 1) {
        return false;
    }
    const double input_norm = vector_two_norm(input);
    if (!std::isfinite(input_norm)) {
        return false;
    }
    if (input_norm == 0.0 || dt == 0.0) {
        output = input;
        used_dimension = 0;
        residual = 0.0;
        return true;
    }
    max_dimension = std::min(max_dimension, n);
    KrylovWorkspace local_workspace;
    KrylovWorkspace& scratch = reusable_workspace != nullptr
        ? *reusable_workspace
        : local_workspace;
    scratch.prepare(n, max_dimension);
    const std::size_t basis_stride = static_cast<std::size_t>(n);
    auto& basis = scratch.basis;
    for (int index = 0; index < n; ++index) {
        basis[static_cast<std::size_t>(index)] =
            input[static_cast<std::size_t>(index)] / input_norm;
    }
    auto& hessenberg = scratch.hessenberg;
    auto& work = scratch.work;
    auto& previous = scratch.previous;
    auto& candidate = scratch.candidate;
    bool have_previous = false;
    const double scaled_tolerance = std::max(
        tolerance + problem.rtol * input_norm,
        32.0 * std::numeric_limits<double>::epsilon() * input_norm);

    for (int column = 0; column < max_dimension; ++column) {
        if (problem.lindblad) {
            problem.apply_lindblad_constant_single(
                basis.data() + static_cast<std::size_t>(column) * basis_stride,
                work.data(),
                scales);
        } else {
            problem.apply_constant_single(
                basis.data() + static_cast<std::size_t>(column) * basis_stride,
                work.data(),
                scales);
        }
        for (int pass = 0; pass < 2; ++pass) {
            for (int row = 0; row <= column; ++row) {
                const cdouble coefficient = vector_inner_product(
                    basis.data() + static_cast<std::size_t>(row) * basis_stride,
                    work.data(),
                    n);
                if (pass == 0) {
                    hessenberg[static_cast<std::size_t>(row) * max_dimension + column] =
                        coefficient;
                } else {
                    hessenberg[static_cast<std::size_t>(row) * max_dimension + column] +=
                        coefficient;
                }
                for (int index = 0; index < n; ++index) {
                    work[static_cast<std::size_t>(index)] -= coefficient
                        * basis[static_cast<std::size_t>(row) * basis_stride + index];
                }
            }
        }
        const double subdiagonal = vector_two_norm(work);
        hessenberg[static_cast<std::size_t>(column + 1) * max_dimension + column] =
            cdouble(subdiagonal, 0.0);
        const int dimension = column + 1;
        const bool breakdown = subdiagonal <=
            64.0 * std::numeric_limits<double>::epsilon();

        // The residual acceptance rule requires at least four basis vectors,
        // so projected exponentials for dimensions 1-3 cannot accept an
        // action unless the Krylov space has already broken down.  Generate
        // those vectors without paying for three guaranteed-unused Pade
        // evaluations.
        if (dimension < 4 && !breakdown) {
            if (subdiagonal <= 0.0 || !std::isfinite(subdiagonal)) {
                return false;
            }
            for (int index = 0; index < n; ++index) {
                basis[static_cast<std::size_t>(dimension) * basis_stride + index] =
                    work[static_cast<std::size_t>(index)] / subdiagonal;
            }
            continue;
        }

        auto& small_matrix = scratch.small_matrix;
        small_matrix.resize(static_cast<std::size_t>(dimension) * dimension);
        for (int row = 0; row < dimension; ++row) {
            for (int col = 0; col < dimension; ++col) {
                small_matrix[static_cast<std::size_t>(row) * dimension + col] =
                    hessenberg[static_cast<std::size_t>(row) * max_dimension + col]
                    * dt;
            }
        }
        auto& small_exponential = scratch.small_exponential;
        if (!matrix_exponential(
                small_matrix,
                dimension,
                small_exponential,
                &scratch.exponential_workspace)) {
            return false;
        }
        for (int index = 0; index < n; ++index) {
            cdouble value(0.0, 0.0);
            for (int row = 0; row < dimension; ++row) {
                value += basis[static_cast<std::size_t>(row) * basis_stride + index]
                    * small_exponential[static_cast<std::size_t>(row) * dimension];
            }
            candidate[static_cast<std::size_t>(index)] = input_norm * value;
        }
        residual = input_norm * std::abs(
            subdiagonal * small_exponential[static_cast<std::size_t>(dimension - 1) * dimension]
            * dt);
        double difference = std::numeric_limits<double>::infinity();
        if (have_previous) {
            difference = 0.0;
            for (int index = 0; index < n; ++index) {
                difference = std::max(
                    difference,
                    std::abs(candidate[static_cast<std::size_t>(index)]
                             - previous[static_cast<std::size_t>(index)]));
            }
        }
        // A tiny subdiagonal is an exact Krylov breakdown: the generated
        // subspace is invariant and the current exponential is complete.
        if (breakdown || (dimension >= 4
                          && residual <= scaled_tolerance
                          && (!have_previous || difference <= scaled_tolerance))) {
            output = candidate;
            used_dimension = dimension;
            return true;
        }
        previous = candidate;
        have_previous = true;
        if (subdiagonal <= 0.0 || !std::isfinite(subdiagonal)) {
            return false;
        }
        for (int index = 0; index < n; ++index) {
            basis[static_cast<std::size_t>(dimension) * basis_stride + index] =
                work[static_cast<std::size_t>(index)] / subdiagonal;
        }
    }
    return false;
}

int choose_parallel_workers(
    const Problem& problem,
    int task_count,
    int action_count = 1) {
    if (task_count < 2 || problem.parallel_mode == 0) {
        return 1;
    }
    unsigned int hardware = std::thread::hardware_concurrency();
    if (hardware == 0) {
        hardware = 2;
    }
    int workers = std::min(task_count, static_cast<int>(hardware));
    const int state_dimension = std::max(1, problem.state_dimension());
    // Automatic mode avoids thread and cache contention for small batches.  An
    // explicit ``parallel='on'`` request still uses all available workers.
    if (problem.parallel_mode == 1) {
        // Thread creation and per-worker scratch dominate small Krylov
        // batches.  Use a conservative work estimate so ``auto`` cannot turn
        // a short final-state request into a slowdown; explicit ``on`` keeps
        // the reproducible all-workers override for benchmarking.
        const std::size_t estimated_work =
            static_cast<std::size_t>(std::max(1, state_dimension))
            * static_cast<std::size_t>(std::max(1, task_count))
            * static_cast<std::size_t>(std::max(1, action_count));
        if (task_count < 16 || estimated_work < 1'000'000u) {
            return 1;
        }
    }
    // Each worker owns an Arnoldi basis.  Keep aggregate scratch bounded for
    // very large Hilbert spaces instead of trading a speedup for paging or
    // allocation failure.
    constexpr std::size_t kMaxParallelScratchBytes = 256u * 1024u * 1024u;
    const std::size_t per_worker = (
        static_cast<std::size_t>(std::min(state_dimension, 96) + 1)
            * static_cast<std::size_t>(state_dimension)
        + 96u * 96u
        + 4u * static_cast<std::size_t>(state_dimension))
        * sizeof(cdouble);
    if (per_worker > 0) {
        const int memory_workers = static_cast<int>(
            std::max<std::size_t>(1, kMaxParallelScratchBytes / per_worker));
        workers = std::min(workers, memory_workers);
    }
    return std::max(1, workers);
}

void apply_matrix_to_state(
    const Matrix& matrix,
    int n,
    int batch_count,
    const std::vector<cdouble>& state,
    std::vector<cdouble>& next) {
    for (int row = 0; row < n; ++row) {
        const std::size_t row_base = static_cast<std::size_t>(row) * batch_count;
        std::fill(
            next.begin() + static_cast<std::ptrdiff_t>(row_base),
            next.begin() + static_cast<std::ptrdiff_t>(row_base + batch_count),
            cdouble(0.0, 0.0));
        // Keep the column summation order unchanged for every batch state,
        // but move the independent batch loop innermost so each propagator
        // element is loaded once and reused across contiguous states.
        for (int column = 0; column < n; ++column) {
            const cdouble value = matrix[static_cast<std::size_t>(row) * n + column];
            if (value == cdouble(0.0, 0.0)) {
                continue;
            }
            const std::size_t source_base = static_cast<std::size_t>(column) * batch_count;
            for (int batch = 0; batch < batch_count; ++batch) {
                next[row_base + static_cast<std::size_t>(batch)] += value
                    * state[source_base + static_cast<std::size_t>(batch)];
            }
        }
    }
}

void apply_diagonal_to_state(
    const std::vector<cdouble>& diagonal,
    int n,
    int batch_count,
    double dt,
    const std::vector<cdouble>& state,
    std::vector<cdouble>& next) {
    for (int row = 0; row < n; ++row) {
        const cdouble factor = exp_minus_i(
            diagonal[static_cast<std::size_t>(row)] * dt);
        for (int batch = 0; batch < batch_count; ++batch) {
            next[static_cast<std::size_t>(row) * batch_count + batch] = factor
                * state[static_cast<std::size_t>(row) * batch_count + batch];
        }
    }
}

bool integrate_interval(
    Problem& problem,
    double start,
    double end,
    std::vector<cdouble>& state,
    Workspace& workspace,
    std::string& error_message) {
    if (end <= start) {
        return true;
    }

    const std::size_t state_size = state.size();
    // Workspace storage is owned by the caller and reused across all intervals.
    auto& k1 = workspace.k1;
    auto& k2 = workspace.k2;
    auto& k3 = workspace.k3;
    auto& k4 = workspace.k4;
    auto& k5 = workspace.k5;
    auto& k6 = workspace.k6;
    auto& k7 = workspace.k7;
    auto& trial = workspace.trial;
    auto& fourth = workspace.fourth;

    double t = start;
    double step = workspace.step_valid ? workspace.next_step : (end - start);
    if (!(step > 0.0) || !std::isfinite(step)) {
        step = end - start;
    }
    bool have_fsal = workspace.fsal_valid
        && std::abs(workspace.fsal_time - start)
            <= 8.0 * std::numeric_limits<double>::epsilon()
                * std::max(1.0, std::abs(start));
    if (!problem.interaction_picture && problem.has_rf_control() && problem.max_frequency > 0.0) {
        // Give the first trial step enough resolution for a lab-frame carrier;
        // the embedded error estimate then expands or contracts it as needed.
        step = std::min(step, 0.1 / problem.max_frequency);
    }

    while (t < end) {
        if (++problem.attempted_steps > problem.max_steps) {
            error_message = "native integrator exceeded max_steps";
            return false;
        }
        const double remaining = end - t;
        step = std::min(step, remaining);
        if (step <= std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(t))) {
            error_message = "native integrator reached a step smaller than machine precision";
            return false;
        }

        if (!have_fsal) {
            problem.rhs(t, state, k1, workspace);
            have_fsal = true;
        }
        for (std::size_t index = 0; index < state_size; ++index) {
            trial[index] = state[index] + step * (1.0 / 5.0) * k1[index];
        }
        problem.rhs(t + step * (1.0 / 5.0), trial, k2, workspace);

        for (std::size_t index = 0; index < state_size; ++index) {
            trial[index] = state[index] + step * (3.0 / 40.0 * k1[index] + 9.0 / 40.0 * k2[index]);
        }
        problem.rhs(t + step * (3.0 / 10.0), trial, k3, workspace);

        for (std::size_t index = 0; index < state_size; ++index) {
            trial[index] = state[index] + step * (
                44.0 / 45.0 * k1[index]
                - 56.0 / 15.0 * k2[index]
                + 32.0 / 9.0 * k3[index]);
        }
        problem.rhs(t + step * (4.0 / 5.0), trial, k4, workspace);

        for (std::size_t index = 0; index < state_size; ++index) {
            trial[index] = state[index] + step * (
                19372.0 / 6561.0 * k1[index]
                - 25360.0 / 2187.0 * k2[index]
                + 64448.0 / 6561.0 * k3[index]
                - 212.0 / 729.0 * k4[index]);
        }
        problem.rhs(t + step * (8.0 / 9.0), trial, k5, workspace);

        for (std::size_t index = 0; index < state_size; ++index) {
            trial[index] = state[index] + step * (
                9017.0 / 3168.0 * k1[index]
                - 355.0 / 33.0 * k2[index]
                + 46732.0 / 5247.0 * k3[index]
                + 49.0 / 176.0 * k4[index]
                - 5103.0 / 18656.0 * k5[index]);
        }
        problem.rhs(t + step, trial, k6, workspace);

        for (std::size_t index = 0; index < state_size; ++index) {
            trial[index] = state[index] + step * (
                35.0 / 384.0 * k1[index]
                + 500.0 / 1113.0 * k3[index]
                + 125.0 / 192.0 * k4[index]
                - 2187.0 / 6784.0 * k5[index]
                + 11.0 / 84.0 * k6[index]);
        }
        problem.rhs(t + step, trial, k7, workspace);

        // ``trial`` already contains the candidate fifth-order estimate from
        // the stage-7 RHS call above.  Recomputing that same linear
        // combination here only to form the embedded fourth-order estimate
        // adds a full state-sized pass; preserve the existing buffer and
        // compute the fourth-order estimate alone.
        for (std::size_t index = 0; index < state_size; ++index) {
            fourth[index] = state[index] + step * (
                5179.0 / 57600.0 * k1[index]
                + 7571.0 / 16695.0 * k3[index]
                + 393.0 / 640.0 * k4[index]
                - 92097.0 / 339200.0 * k5[index]
                + 187.0 / 2100.0 * k6[index]
                + 1.0 / 40.0 * k7[index]);
        }

        double error_norm = 0.0;
        for (std::size_t index = 0; index < state_size; ++index) {
            const double scale = problem.atol + problem.rtol * std::max(std::abs(state[index]), std::abs(trial[index]));
            error_norm = std::max(error_norm, std::abs(trial[index] - fourth[index]) / scale);
        }
        if (!std::isfinite(error_norm)) {
            error_message = "native integrator produced a non-finite error estimate";
            return false;
        }
        problem.max_trial_error = std::max(problem.max_trial_error, error_norm);

        if (error_norm <= 1.0) {
            problem.max_error = std::max(problem.max_error, error_norm);
            state.swap(trial);
            t += step;
            // Dormand-Prince 5(4) is FSAL: k7 is the derivative at the new
            // accepted state and can serve as k1 on the next trial step.
            k1.swap(k7);
            have_fsal = true;
            workspace.fsal_valid = true;
            workspace.fsal_time = t;
            if (std::abs(t - end) <= 8.0 * std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(end))) {
                t = end;
            }
            double factor = error_norm == 0.0
                ? 5.0
                : 0.9 * std::pow(error_norm, -0.2);
            factor = std::min(5.0, std::max(0.2, factor));
            step *= factor;
        } else {
            // The state and time are unchanged after a rejected trial, so the
            // existing k1 remains valid for the smaller retry step.
            if (step <= std::numeric_limits<double>::epsilon() * std::max(1.0, std::abs(t))) {
                error_message = "native integrator could not satisfy the requested tolerance";
                return false;
            }
            const double factor = std::max(0.1, 0.9 * std::pow(error_norm, -0.2));
            step *= factor;
        }
    }
    workspace.fsal_valid = have_fsal;
    workspace.fsal_time = t;
    workspace.step_valid = true;
    workspace.next_step = step;
    return true;
}

// Integral of a linearly interpolated complex envelope multiplied by a
// carrier.  For RF mode the physical coefficient is
//   Re[(z0 + (z1-z0)x/dt) exp(i*w*(t0+x))].
// Evaluating this expression analytically removes every adaptive RHS call for
// diagonal Hamiltonians while preserving the exact piecewise-linear model.
cdouble integrate_rf_linear(
    cdouble z0,
    cdouble z1,
    double t0,
    double t1,
    double angular_frequency) {
    const double dt = t1 - t0;
    if (!(dt > 0.0)) {
        return cdouble(0.0, 0.0);
    }
    if (angular_frequency == 0.0) {
        return cdouble(0.5 * dt * (z0.real() + z1.real()), 0.0);
    }

    const double x = angular_frequency * dt;
    const cdouble iw(0.0, angular_frequency);
    cdouble i0;
    cdouble i1;
    // Series evaluation avoids cancellation in exp(i*x)-1 for short steps.
    if (std::abs(x) < 1e-3) {
        const double x2 = x * x;
        const double x3 = x2 * x;
        const double x4 = x2 * x2;
        i0 = dt * cdouble(
            1.0 - x2 / 6.0 + x4 / 120.0,
            x / 2.0 - x3 / 24.0);
        i1 = dt * dt * cdouble(
            0.5 - x2 / 8.0 + x4 / 144.0,
            x / 3.0 - x3 / 30.0);
    } else {
        double sine = 0.0;
        double cosine = 0.0;
        sine_cosine(x, sine, cosine);
        const cdouble phase(cosine, sine);
        i0 = (phase - cdouble(1.0, 0.0)) / iw;
        i1 = (phase * (cdouble(0.0, x) - cdouble(1.0, 0.0))
            + cdouble(1.0, 0.0)) / (iw * iw);
    }
    double carrier_sine = 0.0;
    double carrier_cosine = 0.0;
    sine_cosine(angular_frequency * t0, carrier_sine, carrier_cosine);
    const cdouble carrier_at_start(carrier_cosine, carrier_sine);
    const cdouble slope = (z1 - z0) / dt;
    const cdouble integral = carrier_at_start * (z0 * i0 + slope * i1);
    return cdouble(integral.real(), 0.0);
}

cdouble integrate_rf_polynomial(
    const cdouble* coefficients,
    int order,
    double t0,
    double t1,
    double angular_frequency) {
    const double dt = t1 - t0;
    if (!(dt > 0.0) || coefficients == nullptr) {
        return cdouble(0.0, 0.0);
    }
    if (angular_frequency == 0.0) {
        cdouble integral(0.0, 0.0);
        for (int power = 0; power <= order; ++power) {
            integral += coefficients[power] * (dt / static_cast<double>(power + 1));
        }
        return cdouble(integral.real(), 0.0);
    }
    const double x = angular_frequency * dt;
    cdouble carrier;
    double carrier_sine = 0.0;
    double carrier_cosine = 0.0;
    sine_cosine(angular_frequency * t0, carrier_sine, carrier_cosine);
    carrier = cdouble(carrier_cosine, carrier_sine);
    cdouble scaled_integral(0.0, 0.0);
    // The normalized-coordinate moments are evaluated by a short convergent
    // series for small |x| and by integration-by-parts recurrence otherwise.
    for (int power = 0; power <= order; ++power) {
        cdouble moment(0.0, 0.0);
        if (std::abs(x) < 0.25) {
            cdouble term(1.0, 0.0);
            for (int k = 0; k <= 24; ++k) {
                if (k > 0) {
                    term *= cdouble(0.0, x) / static_cast<double>(k);
                }
                moment += term / static_cast<double>(power + k + 1);
                if (std::abs(term) < 2.0e-18) {
                    break;
                }
            }
            moment *= dt;
        } else {
            cdouble i0;
            double sine = 0.0;
            double cosine = 0.0;
            sine_cosine(x, sine, cosine);
            const cdouble phase(cosine, sine);
            const cdouble iw(0.0, angular_frequency);
            i0 = (phase - cdouble(1.0, 0.0)) / iw;
            if (power == 0) {
                moment = i0;
            } else {
                cdouble previous = i0;
                for (int p = 1; p <= power; ++p) {
                    const double endpoint_power = std::pow(dt, static_cast<double>(p));
                    moment = phase * endpoint_power / iw
                        - static_cast<double>(p) * previous / iw;
                    previous = moment;
                }
            }
            moment /= std::pow(dt, static_cast<double>(power));
        }
        scaled_integral += coefficients[power] * moment;
    }
    const cdouble integral = carrier * scaled_integral;
    return cdouble(integral.real(), 0.0);
}

cdouble integrate_control_interval(
    const Problem& problem,
    int control,
    int sample,
    double start,
    double end) {
    const double dt = end - start;
    if (problem.iq_polynomial != nullptr
        && problem.polynomial_interval_count > 0
        && sample < problem.polynomial_interval_count) {
        const std::size_t base = (
            static_cast<std::size_t>(control) * problem.polynomial_interval_count
            + static_cast<std::size_t>(sample))
            * static_cast<std::size_t>(problem.coefficient_order + 1);
        const cdouble* coefficients = problem.iq_polynomial + base;
        if (problem.mode_for_control(control) == 1) {
            cdouble integral(0.0, 0.0);
            for (int power = 0; power <= problem.coefficient_order; ++power) {
                integral += coefficients[power]
                    * (dt / static_cast<double>(power + 1));
            }
            return integral;
        }
        const double angular_frequency = problem.angular_freqs.empty()
            ? 2.0 * kPi * problem.lo_freqs[control]
            : problem.angular_freqs[static_cast<std::size_t>(control)];
        return integrate_rf_polynomial(
            coefficients,
            problem.coefficient_order,
            start,
            end,
            angular_frequency);
    }
    const cdouble left = problem.iq[
        static_cast<std::size_t>(control) * problem.sample_count + sample];
    const cdouble right = problem.iq[
        static_cast<std::size_t>(control) * problem.sample_count
        + std::min(sample + 1, problem.sample_count - 1)];
    if (problem.mode_for_control(control) == 1) {
        return 0.5 * dt * (left + right);
    }
    const double angular_frequency = problem.angular_freqs.empty()
        ? 2.0 * kPi * problem.lo_freqs[control]
        : problem.angular_freqs[static_cast<std::size_t>(control)];
    return integrate_rf_linear(left, right, start, end, angular_frequency);
}

bool run_diagonal_integration(
    Problem& problem,
    std::vector<cdouble>& state,
    bool store_trajectory,
    std::vector<cdouble>& trajectory,
    std::string& error_message) {
    (void)error_message;
    if (store_trajectory) {
        trajectory.reserve(
            static_cast<std::size_t>(problem.sample_count) * state.size());
        trajectory.insert(trajectory.end(), state.begin(), state.end());
    }
    std::vector<cdouble> integrals(
        static_cast<std::size_t>(problem.control_count),
        cdouble(0.0, 0.0));

    if (!store_trajectory) {
        // With no requested samples, diagonal entries commute over the whole
        // trace.  Accumulate the scalar phase first and evaluate one
        // exponential per level instead of one per level and interval.
        std::vector<cdouble> cumulative_phase(
            static_cast<std::size_t>(problem.n),
            cdouble(0.0, 0.0));
        Py_BEGIN_ALLOW_THREADS
        for (int sample = 0; sample + 1 < problem.sample_count; ++sample) {
            const double start = problem.t_axis[sample];
            const double end = problem.t_axis[sample + 1];
            const double dt = end - start;
            for (int control = 0; control < problem.control_count; ++control) {
                integrals[static_cast<std::size_t>(control)] =
                    integrate_control_interval(problem, control, sample, start, end);
            }
            for (int level = 0; level < problem.n; ++level) {
                cdouble phase = problem.diagonal_h0[static_cast<std::size_t>(level)] * dt;
                for (int control = 0; control < problem.control_count; ++control) {
                    phase += problem.diagonal_controls[
                        static_cast<std::size_t>(control) * problem.n + level]
                        * integrals[static_cast<std::size_t>(control)];
                }
                cumulative_phase[static_cast<std::size_t>(level)] += phase;
            }
        }
        for (int level = 0; level < problem.n; ++level) {
            const cdouble factor = exp_minus_i(
                cumulative_phase[static_cast<std::size_t>(level)]);
            for (int batch = 0; batch < problem.batch_count; ++batch) {
                state[static_cast<std::size_t>(level) * problem.batch_count + batch] *= factor;
            }
        }
        Py_END_ALLOW_THREADS
        problem.attempted_steps = std::max<std::int64_t>(0, problem.sample_count - 1);
        problem.rhs_evaluations = 0;
        problem.max_error = 0.0;
        problem.max_trial_error = 0.0;
        return true;
    }

    // The diagonal entries commute at all times.  The ordered exponential is
    // therefore the scalar exponential of the integrated diagonal value.
    Py_BEGIN_ALLOW_THREADS
    for (int sample = 0; sample + 1 < problem.sample_count; ++sample) {
        const double start = problem.t_axis[sample];
        const double end = problem.t_axis[sample + 1];
        const double dt = end - start;
        for (int control = 0; control < problem.control_count; ++control) {
            integrals[static_cast<std::size_t>(control)] =
                integrate_control_interval(problem, control, sample, start, end);
        }
        for (int level = 0; level < problem.n; ++level) {
            cdouble phase = problem.diagonal_h0[static_cast<std::size_t>(level)] * dt;
            for (int control = 0; control < problem.control_count; ++control) {
                phase += problem.diagonal_controls[
                    static_cast<std::size_t>(control) * problem.n + level]
                    * integrals[static_cast<std::size_t>(control)];
            }
            const cdouble factor = exp_minus_i(phase);
            for (int batch = 0; batch < problem.batch_count; ++batch) {
                state[static_cast<std::size_t>(level) * problem.batch_count + batch] *= factor;
            }
        }
        if (store_trajectory) {
            trajectory.insert(trajectory.end(), state.begin(), state.end());
        }
    }
    Py_END_ALLOW_THREADS
    problem.attempted_steps = std::max<std::int64_t>(0, problem.sample_count - 1);
    problem.rhs_evaluations = 0;
    problem.max_error = 0.0;
    problem.max_trial_error = 0.0;
    return true;
}

bool run_lindblad_diagonal_integration(
    Problem& problem,
    std::vector<cdouble>& state,
    bool store_trajectory,
    std::vector<cdouble>& trajectory,
    std::string& error_message) {
    (void)error_message;
    if (!problem.lindblad_diagonal_only) {
        return false;
    }
    const std::size_t state_size = state.size();
    const int interval_count = std::max(0, problem.sample_count - 1);
    const std::size_t pair_count =
        static_cast<std::size_t>(problem.n) * problem.n;
    std::vector<cdouble> dissipative_rates(pair_count, cdouble(0.0, 0.0));
    for (int row = 0; row < problem.n; ++row) {
        for (int column = 0; column < problem.n; ++column) {
            cdouble rate(0.0, 0.0);
            for (int collapse = 0;
                 collapse < problem.lindblad_collapse_count;
                 ++collapse) {
                const std::size_t base =
                    static_cast<std::size_t>(collapse) * problem.n;
                rate +=
                    problem.lindblad_collapse_diagonal[base + row]
                        * std::conj(problem.lindblad_collapse_diagonal[base + column])
                    - cdouble(0.5, 0.0) * (
                        problem.lindblad_product_diagonal[base + row]
                        + std::conj(problem.lindblad_product_diagonal[base + column]));
            }
            dissipative_rates[static_cast<std::size_t>(row)
                + static_cast<std::size_t>(column) * problem.n] = rate;
        }
    }

    std::vector<cdouble> integrals(
        static_cast<std::size_t>(problem.control_count), cdouble(0.0, 0.0));
    std::vector<cdouble> interval_left(
        static_cast<std::size_t>(problem.n), cdouble(0.0, 0.0));
    std::vector<cdouble> interval_right(
        static_cast<std::size_t>(problem.n), cdouble(0.0, 0.0));
    std::vector<cdouble> cumulative_left(
        static_cast<std::size_t>(problem.n), cdouble(0.0, 0.0));
    std::vector<cdouble> cumulative_right(
        static_cast<std::size_t>(problem.n), cdouble(0.0, 0.0));
    std::vector<cdouble> working = state;
    std::vector<cdouble> local_trajectory;
    if (store_trajectory) {
        local_trajectory.reserve(
            static_cast<std::size_t>(problem.sample_count) * state_size);
        local_trajectory.insert(
            local_trajectory.end(), working.begin(), working.end());
    }

    auto fill_level_phases = [&](int sample, double start, double end) -> bool {
        const double dt = end - start;
        if (!(dt > 0.0) || !std::isfinite(dt)) {
            return false;
        }
        for (int control = 0; control < problem.control_count; ++control) {
            integrals[static_cast<std::size_t>(control)] =
                integrate_control_interval(problem, control, sample, start, end);
        }
        for (int level = 0; level < problem.n; ++level) {
            cdouble left = problem.diagonal_h0[static_cast<std::size_t>(level)] * dt;
            cdouble right = std::conj(
                problem.diagonal_h0[static_cast<std::size_t>(level)]) * dt;
            for (int control = 0; control < problem.control_count; ++control) {
                const std::size_t base =
                    static_cast<std::size_t>(control) * problem.n;
                const cdouble contribution =
                    problem.diagonal_controls[base + level]
                    * integrals[static_cast<std::size_t>(control)];
                left += contribution;
                right += std::conj(contribution);
            }
            interval_left[static_cast<std::size_t>(level)] = left;
            interval_right[static_cast<std::size_t>(level)] = right;
        }
        return true;
    };

    bool success = true;
    Py_BEGIN_ALLOW_THREADS
    try {
        if (!store_trajectory) {
            double total_duration = 0.0;
            for (int sample = 0; sample < interval_count; ++sample) {
                const double start = problem.t_axis[sample];
                const double end = problem.t_axis[sample + 1];
                if (!fill_level_phases(sample, start, end)) {
                    success = false;
                    break;
                }
                const double dt = end - start;
                total_duration += dt;
                for (int level = 0; level < problem.n; ++level) {
                    cumulative_left[static_cast<std::size_t>(level)] +=
                        interval_left[static_cast<std::size_t>(level)];
                    cumulative_right[static_cast<std::size_t>(level)] +=
                        interval_right[static_cast<std::size_t>(level)];
                }
            }
            if (success) {
                int worker_count = 1;
                if (problem.parallel_mode != 0
                    && problem.n >= 128
                    && pair_count * static_cast<std::size_t>(problem.batch_count) >= 16384
                    && (problem.parallel_mode == 2 || problem.batch_count >= 16)) {
                    unsigned int hardware = std::thread::hardware_concurrency();
                    if (hardware == 0) {
                        hardware = 2;
                    }
                    worker_count = std::min(
                        std::min(problem.n, static_cast<int>(hardware)),
                        8);
                }
                problem.parallel_workers = std::max(
                    problem.parallel_workers,
                    static_cast<std::int64_t>(worker_count));
                std::atomic<bool> factor_failed(false);
                auto apply_rows = [&](int row_begin, int row_end) {
                    for (int row = row_begin;
                         row < row_end
                             && !factor_failed.load(std::memory_order_relaxed);
                         ++row) {
                        for (int column = 0; column < problem.n; ++column) {
                            const std::size_t state_base = (
                                static_cast<std::size_t>(row)
                                + static_cast<std::size_t>(column) * problem.n)
                                * static_cast<std::size_t>(problem.batch_count);
                            bool nonzero = false;
                            for (int batch = 0; batch < problem.batch_count; ++batch) {
                                if (working[state_base + static_cast<std::size_t>(batch)]
                                    != cdouble(0.0, 0.0)) {
                                    nonzero = true;
                                    break;
                                }
                            }
                            if (!nonzero) {
                                continue;
                            }
                            const std::size_t pair_index =
                                static_cast<std::size_t>(row)
                                + static_cast<std::size_t>(column) * problem.n;
                            const cdouble exponent = cdouble(0.0, -1.0) * (
                                cumulative_left[static_cast<std::size_t>(row)]
                                - cumulative_right[static_cast<std::size_t>(column)])
                                + dissipative_rates[pair_index] * total_duration;
                            const cdouble factor = exp_complex(exponent);
                            if (!finite_value(factor)) {
                                factor_failed.store(true, std::memory_order_relaxed);
                                break;
                            }
                            for (int batch = 0; batch < problem.batch_count; ++batch) {
                                working[state_base + static_cast<std::size_t>(batch)] *= factor;
                            }
                        }
                    }
                };
                if (worker_count == 1) {
                    apply_rows(0, problem.n);
                } else {
                    std::vector<std::thread> workers;
                    workers.reserve(static_cast<std::size_t>(worker_count));
                    const int rows_per_worker =
                        (problem.n + worker_count - 1) / worker_count;
                    for (int worker_index = 0; worker_index < worker_count; ++worker_index) {
                        const int row_begin = std::min(
                            problem.n, worker_index * rows_per_worker);
                        const int row_end = std::min(
                            problem.n, row_begin + rows_per_worker);
                        workers.emplace_back(
                            apply_rows, row_begin, row_end);
                    }
                    for (std::thread& thread : workers) {
                        thread.join();
                    }
                }
                if (factor_failed.load(std::memory_order_relaxed)) {
                    success = false;
                }
            }
        } else {
            for (int sample = 0; sample < interval_count; ++sample) {
                const double start = problem.t_axis[sample];
                const double end = problem.t_axis[sample + 1];
                if (!fill_level_phases(sample, start, end)) {
                    success = false;
                    break;
                }
                const double dt = end - start;
                for (int row = 0; row < problem.n; ++row) {
                    for (int column = 0; column < problem.n; ++column) {
                        const std::size_t state_base = (
                            static_cast<std::size_t>(row)
                            + static_cast<std::size_t>(column) * problem.n)
                            * static_cast<std::size_t>(problem.batch_count);
                        bool nonzero = false;
                        for (int batch = 0; batch < problem.batch_count; ++batch) {
                            if (working[state_base + static_cast<std::size_t>(batch)]
                                != cdouble(0.0, 0.0)) {
                                nonzero = true;
                                break;
                            }
                        }
                        if (!nonzero) {
                            continue;
                        }
                        const std::size_t pair_index =
                            static_cast<std::size_t>(row)
                            + static_cast<std::size_t>(column) * problem.n;
                        const cdouble exponent = cdouble(0.0, -1.0) * (
                            interval_left[static_cast<std::size_t>(row)]
                            - interval_right[static_cast<std::size_t>(column)])
                            + dissipative_rates[pair_index] * dt;
                        const cdouble factor = exp_complex(exponent);
                        if (!finite_value(factor)) {
                            success = false;
                            break;
                        }
                        for (int batch = 0; batch < problem.batch_count; ++batch) {
                            working[state_base + static_cast<std::size_t>(batch)] *= factor;
                        }
                    }
                    if (!success) {
                        break;
                    }
                }
                if (!success) {
                    break;
                }
                local_trajectory.insert(
                    local_trajectory.end(), working.begin(), working.end());
            }
        }
    } catch (...) {
        success = false;
    }
    Py_END_ALLOW_THREADS
    if (!success) {
        return false;
    }
    state.swap(working);
    if (store_trajectory) {
        trajectory.swap(local_trajectory);
    }
    problem.lindblad_diagonal_exact = true;
    problem.attempted_steps = static_cast<std::int64_t>(interval_count);
    problem.rhs_evaluations = 0;
    problem.max_error = 0.0;
    problem.max_trial_error = 0.0;
    return true;
}


bool run_constant_exponential(
    Problem& problem,
    std::vector<cdouble>& state,
    bool store_trajectory,
    std::vector<cdouble>& trajectory,
    std::string& error_message) {
    (void)error_message;
    if (problem.n > kMaxDenseExponentialDimension) {
        return false;
    }
    Matrix hamiltonian;
    problem.build_constant_hamiltonian(hamiltonian);
    Matrix propagator;
    if (problem.sample_count > 1) {
        const double first_dt = problem.t_axis[1] - problem.t_axis[0];
        const double total_dt = problem.t_axis[problem.sample_count - 1]
            - problem.t_axis[0];
        bool equal_intervals = true;
        for (int sample = 2; sample < problem.sample_count; ++sample) {
            const double current_dt = problem.t_axis[sample] - problem.t_axis[sample - 1];
            if (current_dt != first_dt) {
                equal_intervals = false;
                break;
            }
        }
        if (store_trajectory && !equal_intervals) {
            return false;
        }
        // For a final-state-only request, exp(A t_k) factors through the
        // same constant generator for every interval.  Applying exp(A T)
        // once is exactly equivalent to all interval applications, including
        // nonuniform output grids.
        const double propagation_dt = store_trajectory || equal_intervals
            ? first_dt
            : total_dt;
        if (!(propagation_dt > 0.0) || !std::isfinite(propagation_dt)) {
            return false;
        }
        Matrix scaled(hamiltonian.size());
        for (std::size_t index = 0; index < hamiltonian.size(); ++index) {
            scaled[index] = cdouble(0.0, -propagation_dt) * hamiltonian[index];
        }
        if (!matrix_exponential(scaled, problem.n, propagator)) {
            return false;
        }
        problem.exponential_evaluations = 1;
        problem.whole_trace_exponential = true;
    } else {
        matrix_identity(problem.n, propagator);
        problem.exponential_evaluations = 0;
    }

    if (store_trajectory) {
        trajectory.reserve(
            static_cast<std::size_t>(problem.sample_count) * state.size());
        trajectory.insert(trajectory.end(), state.begin(), state.end());
    }
    std::vector<cdouble> next(state.size(), cdouble(0.0, 0.0));
    Py_BEGIN_ALLOW_THREADS
    const int application_count = store_trajectory
        ? std::max(0, problem.sample_count - 1)
        : (problem.sample_count > 1 ? 1 : 0);
    for (int sample = 0; sample < application_count; ++sample) {
        for (int row = 0; row < problem.n; ++row) {
            for (int batch = 0; batch < problem.batch_count; ++batch) {
                cdouble value(0.0, 0.0);
                for (int column = 0; column < problem.n; ++column) {
                    value += propagator[static_cast<std::size_t>(row) * problem.n + column]
                        * state[static_cast<std::size_t>(column) * problem.batch_count + batch];
                }
                next[static_cast<std::size_t>(row) * problem.batch_count + batch] = value;
            }
        }
        state.swap(next);
        if (store_trajectory) {
            trajectory.insert(trajectory.end(), state.begin(), state.end());
        }
    }
    Py_END_ALLOW_THREADS
    problem.attempted_steps = std::max<std::int64_t>(0, problem.sample_count - 1);
    problem.rhs_evaluations = 0;
    problem.max_error = 0.0;
    problem.max_trial_error = 0.0;
    return true;
}

bool run_krylov_batch_intervals(
    Problem& problem,
    const std::vector<std::vector<cdouble>>* interval_scales,
    const std::vector<cdouble>& constant_scales,
    std::vector<cdouble>& state,
    bool store_trajectory,
    std::vector<cdouble>& trajectory,
    std::string& error_message) {
    (void)error_message;
    constexpr int kMaxKrylovDimension = 96;
    const int interval_count = std::max(0, problem.sample_count - 1);
    const bool collapse_constant_intervals =
        interval_scales == nullptr && !store_trajectory && interval_count > 1;
    const int action_count = collapse_constant_intervals ? 1 : interval_count;
    std::vector<double> interval_durations(
        static_cast<std::size_t>(action_count), 0.0);
    if (collapse_constant_intervals) {
        const double total_dt = problem.t_axis[problem.sample_count - 1]
            - problem.t_axis[0];
        if (!(total_dt > 0.0) || !std::isfinite(total_dt)) {
            return false;
        }
        interval_durations[0] = total_dt;
    } else {
        for (int sample = 0; sample < interval_count; ++sample) {
            const double dt = problem.t_axis[sample + 1] - problem.t_axis[sample];
            if (!(dt > 0.0) || !std::isfinite(dt)) {
                return false;
            }
            interval_durations[static_cast<std::size_t>(sample)] = dt;
        }
    }

    const int worker_count = choose_parallel_workers(
        problem,
        problem.batch_count,
        action_count);
    problem.parallel_workers = worker_count;
    std::vector<cdouble> result(state.size(), cdouble(0.0, 0.0));
    std::vector<cdouble> local_trajectory;
    if (store_trajectory) {
        local_trajectory.resize(
            static_cast<std::size_t>(problem.sample_count) * state.size());
        std::copy(state.begin(), state.end(), local_trajectory.begin());
    }

    std::vector<std::int64_t> worker_evaluations(
        static_cast<std::size_t>(worker_count), 0);
    std::vector<std::int64_t> worker_iterations(
        static_cast<std::size_t>(worker_count), 0);
    std::atomic<bool> worker_failed(false);
    bool success = true;

    // Each batch column is independent.  Keep one worker per assigned set of
    // columns for the entire call, so interval propagation never recreates
    // threads and no synchronization is needed between columns.
    Py_BEGIN_ALLOW_THREADS
    try {
        auto worker = [&](int worker_index) {
            const int state_dimension = problem.state_dimension();
            std::vector<cdouble> input(static_cast<std::size_t>(state_dimension));
            std::vector<cdouble> output(static_cast<std::size_t>(state_dimension));
            KrylovWorkspace krylov_scratch;
            for (int batch = worker_index;
                 batch < problem.batch_count
                     && !worker_failed.load(std::memory_order_relaxed);
                 batch += worker_count) {
                for (int row = 0; row < state_dimension; ++row) {
                    input[static_cast<std::size_t>(row)] = state[
                        static_cast<std::size_t>(row) * problem.batch_count + batch];
                }
                for (int sample = 0;
                     sample < action_count
                         && !worker_failed.load(std::memory_order_relaxed);
                     ++sample) {
                    const std::vector<cdouble>& scales =
                        interval_scales != nullptr
                            ? (*interval_scales)[static_cast<std::size_t>(sample)]
                            : constant_scales;
                    int used_dimension = 0;
                    double residual = 0.0;
                    bool ok = false;
                    try {
                        ok = krylov_exponential_action(
                            problem,
                            input,
                            scales,
                            interval_durations[static_cast<std::size_t>(sample)],
                            problem.atol,
                            kMaxKrylovDimension,
                            output,
                            used_dimension,
                            residual,
                            &krylov_scratch);
                    } catch (...) {
                        ok = false;
                    }
                    if (!ok) {
                        worker_failed.store(true, std::memory_order_relaxed);
                        break;
                    }
                    worker_evaluations[static_cast<std::size_t>(worker_index)] += 1;
                    worker_iterations[static_cast<std::size_t>(worker_index)] += used_dimension;
                    input.swap(output);
                    if (store_trajectory) {
                        const std::size_t base =
                            static_cast<std::size_t>(sample + 1) * state.size();
                        for (int row = 0; row < state_dimension; ++row) {
                            local_trajectory[
                                base + static_cast<std::size_t>(row)
                                    * problem.batch_count + batch] =
                                input[static_cast<std::size_t>(row)];
                        }
                    }
                }
                if (worker_failed.load(std::memory_order_relaxed)) {
                    break;
                }
                for (int row = 0; row < state_dimension; ++row) {
                    result[static_cast<std::size_t>(row) * problem.batch_count + batch] =
                        input[static_cast<std::size_t>(row)];
                }
            }
        };

        if (worker_count == 1) {
            worker(0);
        } else {
            std::vector<std::thread> workers;
            workers.reserve(static_cast<std::size_t>(worker_count));
            for (int worker_index = 0; worker_index < worker_count; ++worker_index) {
                workers.emplace_back(worker, worker_index);
            }
            for (std::thread& thread : workers) {
                thread.join();
            }
        }
    } catch (...) {
        success = false;
    }
    Py_END_ALLOW_THREADS

    if (worker_failed.load(std::memory_order_relaxed)) {
        success = false;
    }
    for (const std::int64_t value : worker_evaluations) {
        problem.krylov_evaluations += value;
    }
    for (const std::int64_t value : worker_iterations) {
        problem.krylov_iterations += value;
    }
    if (!success) {
        problem.krylov_evaluations = 0;
        problem.krylov_iterations = 0;
        problem.parallel_workers = 1;
        return false;
    }

    state.swap(result);
    if (store_trajectory) {
        trajectory.swap(local_trajectory);
    }
    problem.krylov_used = true;
    problem.attempted_steps = static_cast<std::int64_t>(interval_count);
    problem.rhs_evaluations = 0;
    problem.max_error = 0.0;
    problem.max_trial_error = 0.0;
    return true;
}

bool run_constant_krylov(
    Problem& problem,
    std::vector<cdouble>& state,
    bool store_trajectory,
    std::vector<cdouble>& trajectory,
    std::string& error_message) {
    if (!(problem.sparse || problem.banded || problem.fused_sparse)
        || problem.interaction_picture
        || problem.sparse_expm_mode == 0
        || (problem.state_dimension() <= kMaxDenseExponentialDimension
            && problem.sparse_expm_mode != 2)
        || !problem.has_constant_coefficients()) {
        return false;
    }
    std::vector<cdouble> scales(
        static_cast<std::size_t>(problem.control_count), cdouble(0.0, 0.0));
    for (int control = 0; control < problem.control_count; ++control) {
        scales[static_cast<std::size_t>(control)] =
            problem.constant_coefficient(control);
    }
    problem.prepare_constant_operator(scales);
    return run_krylov_batch_intervals(
        problem,
        nullptr,
        scales,
        state,
        store_trajectory,
        trajectory,
        error_message);
}

bool run_piecewise_constant_krylov(
    Problem& problem,
    std::vector<cdouble>& state,
    bool store_trajectory,
    std::vector<cdouble>& trajectory,
    std::string& error_message) {
    if (!(problem.sparse || problem.banded || problem.fused_sparse)
        || problem.interaction_picture
        || problem.sparse_expm_mode == 0
        || (problem.state_dimension() <= kMaxDenseExponentialDimension
            && problem.sparse_expm_mode != 2)
        || problem.sample_count < 2) {
        return false;
    }

    // Only enter this path when every source interval is exactly constant.
    // If even one interval is genuinely varying, leave the state untouched so
    // the established adaptive RK path can represent the original function.
    std::vector<std::vector<cdouble>> interval_scales(
        static_cast<std::size_t>(problem.sample_count - 1));
    for (int sample = 0; sample + 1 < problem.sample_count; ++sample) {
        problem.source_interval = sample;
        if (!problem.interval_constant_scales(
                sample,
                interval_scales[static_cast<std::size_t>(sample)])) {
            return false;
        }
    }
    if (!run_krylov_batch_intervals(
            problem,
            &interval_scales,
            std::vector<cdouble>(),
            state,
            store_trajectory,
            trajectory,
            error_message)) {
        return false;
    }
    problem.piecewise_krylov_used = true;
    return true;
}
void apply_interaction_frame(
    const Problem& problem,
    std::vector<cdouble>& state,
    double time) {
    const double elapsed = time - problem.t0;
    for (int level = 0; level < problem.n; ++level) {
        const cdouble energy = problem.interaction_energies[level];
        cdouble factor(1.0, 0.0);
        if (energy != cdouble(0.0, 0.0)) {
            // The interaction parser admits real energies only, so evaluate
            // this unit-modulus phase directly instead of invoking the
            // general complex exponential implementation.
            double sine = 0.0;
            double cosine = 0.0;
            sine_cosine(-energy.real() * elapsed, sine, cosine);
            factor = cdouble(cosine, sine);
        }
        for (int batch = 0; batch < problem.batch_count; ++batch) {
            state[static_cast<std::size_t>(level) * problem.batch_count + batch] *= factor;
        }
    }
}

bool run_interaction_integration(
    Problem& problem,
    std::vector<cdouble>& state,
    Workspace& workspace,
    bool store_trajectory,
    std::vector<cdouble>& trajectory,
    std::string& error_message) {
    if (store_trajectory) {
        trajectory.reserve(
            static_cast<std::size_t>(problem.sample_count) * state.size());
        trajectory.insert(trajectory.end(), state.begin(), state.end());
    }
    std::vector<cdouble> lab_state;
    if (store_trajectory) {
        lab_state.resize(state.size());
    }
    std::vector<cdouble> interval_scales;
    bool success = true;
    Py_BEGIN_ALLOW_THREADS
    try {
        for (int sample = 0; sample + 1 < problem.sample_count; ++sample) {
            problem.source_interval = sample;
            bool zero_interval = problem.interval_constant_scales(sample, interval_scales);
            if (zero_interval) {
                for (const cdouble value : interval_scales) {
                    if (value != cdouble(0.0, 0.0)) {
                        zero_interval = false;
                        break;
                    }
                }
            }
            if (zero_interval) {
                ++problem.zero_interval_evaluations;
                // The exact zero-drive segment invalidates the endpoint
                // derivative, but its duration carries useful step-size
                // information into the next active segment.  Retaining that
                // estimate avoids a large rejected trial after every idle
                // region without changing the adaptive error test.
                workspace.fsal_valid = false;
                if (workspace.step_valid) {
                    workspace.next_step = std::min(
                        workspace.next_step,
                        problem.t_axis[sample + 1] - problem.t_axis[sample]);
                }
            } else if (!integrate_interval(
                           problem,
                           problem.t_axis[sample],
                           problem.t_axis[sample + 1],
                           state,
                           workspace,
                           error_message)) {
                success = false;
                break;
            }
            if (store_trajectory) {
                std::copy(state.begin(), state.end(), lab_state.begin());
                apply_interaction_frame(
                    problem,
                    lab_state,
                    problem.t_axis[sample + 1]);
                trajectory.insert(trajectory.end(), lab_state.begin(), lab_state.end());
            }
        }
    } catch (const std::exception& exception) {
        success = false;
        error_message = exception.what();
    } catch (...) {
        success = false;
        error_message = "native interaction-picture integrator failed";
    }
    Py_END_ALLOW_THREADS
    if (success) {
        apply_interaction_frame(problem, state, problem.t_axis[problem.sample_count - 1]);
    }
    return success;
}

bool run_integration(
    Problem& problem,
    std::vector<cdouble>& state,
    Workspace& workspace,
    bool store_trajectory,
    std::vector<cdouble>& trajectory,
    std::string& error_message) {
    const auto integration_started = std::chrono::steady_clock::now();
    struct IntegrationTimer {
        Problem& problem;
        std::chrono::steady_clock::time_point started;
        ~IntegrationTimer() {
            problem.integration_seconds = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
        }
    } timer{problem, integration_started};
    if (problem.lindblad && problem.lindblad_diagonal_only) {
        return run_lindblad_diagonal_integration(
            problem,
            state,
            store_trajectory,
            trajectory,
            error_message);
    }
    if (!problem.lindblad && problem.diagonal_only) {
        return run_diagonal_integration(
            problem,
            state,
            store_trajectory,
            trajectory,
            error_message);
    }
    if (!problem.lindblad && problem.interaction_picture) {
        return run_interaction_integration(
            problem,
            state,
            workspace,
            store_trajectory,
            trajectory,
            error_message);
    }
    if ((problem.sparse || problem.banded || problem.fused_sparse)
        && problem.sparse_expm_mode != 0
        && (problem.sparse_expm_mode == 2 || !problem.lindblad)
        && (problem.state_dimension() > kMaxDenseExponentialDimension
            || problem.sparse_expm_mode == 2)
        && problem.has_constant_coefficients()) {
        if (run_constant_krylov(
                problem,
                state,
                store_trajectory,
                trajectory,
                error_message)) {
            return true;
        }
        // A failed certificate leaves ``state`` and ``trajectory`` untouched;
        // continue with the established adaptive RK implementation.
        problem.krylov_evaluations = 0;
        problem.krylov_iterations = 0;
    }
    if ((problem.sparse || problem.banded || problem.fused_sparse)
        && problem.sparse_expm_mode != 0
        && (problem.sparse_expm_mode == 2 || !problem.lindblad)
        && (problem.state_dimension() > kMaxDenseExponentialDimension
            || problem.sparse_expm_mode == 2)
        && !problem.has_constant_coefficients()) {
        if (run_piecewise_constant_krylov(
                problem,
                state,
                store_trajectory,
                trajectory,
                error_message)) {
            return true;
        }
        // A failed certificate or a varying interval leaves the established
        // adaptive RK path as the exact fallback.
        problem.krylov_evaluations = 0;
        problem.krylov_iterations = 0;
    }
    if (!problem.lindblad && problem.has_constant_coefficients()) {
        if (run_constant_exponential(
                problem,
                state,
                store_trajectory,
                trajectory,
                error_message)) {
            return true;
        }
    }
    if (store_trajectory) {
        trajectory.reserve(
            static_cast<std::size_t>(problem.sample_count) * state.size());
        trajectory.insert(trajectory.end(), state.begin(), state.end());
    }

    std::vector<cdouble> interval_scales;
    std::vector<cdouble> interval_hamiltonian;
    std::vector<cdouble> interval_scaled;
    std::vector<cdouble> interval_propagator;
    std::vector<cdouble> interval_next(state.size(), cdouble(0.0, 0.0));
    // Constant coefficient intervals are common around pulses (for example,
    // long idle regions).  Reusing an exponential for the exact same scales
    // and duration removes repeated O(n^3) setup without changing the
    // represented piecewise-linear Hamiltonian.  Keep the cache bounded so a
    // waveform with many distinct plateaus cannot grow memory indefinitely.
    struct IntervalCacheEntry {
        double dt{0.0};
        std::vector<cdouble> scales;
        std::vector<cdouble> propagator;
    };
    std::vector<IntervalCacheEntry> interval_cache;
    constexpr std::size_t kIntervalCacheLimit = 8;

    bool success = true;
    // The whole numerical loop runs without the Python GIL.  No Python object
    // or callback is touched until all intervals have been integrated.
    Py_BEGIN_ALLOW_THREADS
    try {
        for (int sample = 0; sample + 1 < problem.sample_count; ++sample) {
            problem.source_interval = sample;
            bool handled = false;
            const double interval_dt =
                problem.t_axis[sample + 1] - problem.t_axis[sample];
            if (!problem.lindblad
                && problem.interval_constant_scales(sample, interval_scales)) {
                bool all_zero = true;
                for (const cdouble value : interval_scales) {
                    if (value != cdouble(0.0, 0.0)) {
                        all_zero = false;
                        break;
                    }
                }
                if (all_zero && problem.h0_diagonal_only) {
                    apply_diagonal_to_state(
                        problem.h0_diagonal_values,
                        problem.n,
                        problem.batch_count,
                        interval_dt,
                        state,
                        interval_next);
                    handled = true;
                    ++problem.diagonal_interval_evaluations;
                } else if (problem.n <= kMaxDenseExponentialDimension) {
                    const std::vector<cdouble>* propagator = nullptr;
                    for (IntervalCacheEntry& cached : interval_cache) {
                        if (cached.dt == interval_dt
                            && cached.scales == interval_scales) {
                            propagator = &cached.propagator;
                            ++problem.exponential_cache_hits;
                            break;
                        }
                    }
                    if (propagator == nullptr) {
                        problem.build_hamiltonian_from_scales(
                            interval_scales,
                            interval_hamiltonian);
                        interval_scaled.resize(interval_hamiltonian.size());
                        for (std::size_t index = 0; index < interval_hamiltonian.size(); ++index) {
                            interval_scaled[index] = cdouble(0.0, -interval_dt)
                                * interval_hamiltonian[index];
                        }
                        if (!matrix_exponential(
                                interval_scaled,
                                problem.n,
                                interval_propagator)) {
                            interval_propagator.clear();
                        } else {
                            if (interval_cache.size() >= kIntervalCacheLimit) {
                                interval_cache.erase(interval_cache.begin());
                            }
                            interval_cache.push_back({
                                interval_dt,
                                std::move(interval_scales),
                                std::move(interval_propagator),
                            });
                            propagator = &interval_cache.back().propagator;
                        }
                    }
                    if (propagator != nullptr && !propagator->empty()) {
                        apply_matrix_to_state(
                            *propagator,
                            problem.n,
                            problem.batch_count,
                            state,
                            interval_next);
                        handled = true;
                        ++problem.exponential_evaluations;
                        problem.interval_exponential = true;
                    }
                }
            }
            if (handled) {
                state.swap(interval_next);
                // A specialized interval may change the Hamiltonian at the
                // next sample boundary, so a previous RK endpoint derivative
                // cannot be reused across it.
                workspace.fsal_valid = false;
                if (workspace.step_valid) {
                    workspace.next_step = std::min(
                        workspace.next_step,
                        interval_dt);
                }
            } else if (!integrate_interval(
                           problem,
                           problem.t_axis[sample],
                           problem.t_axis[sample + 1],
                           state,
                           workspace,
                           error_message)) {
                success = false;
                break;
            }
            if (store_trajectory) {
                trajectory.insert(trajectory.end(), state.begin(), state.end());
            }
        }
    } catch (const std::exception& exception) {
        success = false;
        error_message = exception.what();
    } catch (...) {
        success = false;
        error_message = "native integrator failed with an unknown exception";
    }
    Py_END_ALLOW_THREADS
    return success;
}

PyObject* make_native_result(
    const Problem& problem,
    const std::vector<cdouble>& state,
    const std::vector<cdouble>& trajectory,
    bool store_trajectory) {
    PyObject* final_bytes = PyBytes_FromStringAndSize(
        reinterpret_cast<const char*>(state.data()),
        static_cast<Py_ssize_t>(state.size() * sizeof(cdouble)));
    if (final_bytes == nullptr) {
        return nullptr;
    }
    PyObject* trajectory_object = nullptr;
    if (store_trajectory) {
        trajectory_object = PyBytes_FromStringAndSize(
            reinterpret_cast<const char*>(trajectory.data()),
            static_cast<Py_ssize_t>(trajectory.size() * sizeof(cdouble)));
    } else {
        trajectory_object = Py_None;
        Py_INCREF(Py_None);
    }
    if (trajectory_object == nullptr) {
        Py_DECREF(final_bytes);
        return nullptr;
    }
    PyObject* stats = PyDict_New();
    if (stats == nullptr) {
        Py_DECREF(final_bytes);
        Py_DECREF(trajectory_object);
        return nullptr;
    }
    PyObject* value = PyLong_FromLongLong(problem.attempted_steps);
    PyDict_SetItemString(stats, "steps", value);
    Py_DECREF(value);
    value = PyLong_FromLongLong(problem.rhs_evaluations);
    PyDict_SetItemString(stats, "rhs_evaluations", value);
    Py_DECREF(value);
    value = PyFloat_FromDouble(problem.max_error);
    PyDict_SetItemString(stats, "max_error", value);
    Py_DECREF(value);
    value = PyFloat_FromDouble(problem.max_trial_error);
    PyDict_SetItemString(stats, "max_trial_error", value);
    Py_DECREF(value);
    value = PyLong_FromLongLong(problem.exponential_evaluations);
    PyDict_SetItemString(stats, "exponential_evaluations", value);
    Py_DECREF(value);
    value = PyLong_FromLongLong(problem.exponential_cache_hits);
    PyDict_SetItemString(stats, "exponential_cache_hits", value);
    Py_DECREF(value);
    value = PyLong_FromLongLong(problem.krylov_evaluations);
    PyDict_SetItemString(stats, "krylov_evaluations", value);
    Py_DECREF(value);
    value = PyLong_FromLongLong(problem.krylov_iterations);
    PyDict_SetItemString(stats, "krylov_iterations", value);
    Py_DECREF(value);
    value = PyFloat_FromDouble(problem.integration_seconds);
    PyDict_SetItemString(stats, "integration_seconds", value);
    Py_DECREF(value);
    value = PyLong_FromLong(problem.state_dimension());
    PyDict_SetItemString(stats, "state_dimension", value);
    Py_DECREF(value);
    value = PyLong_FromLong(problem.sparse_expm_mode);
    PyDict_SetItemString(stats, "sparse_expm_mode", value);
    Py_DECREF(value);
    value = PyLong_FromLong(problem.parallel_mode);
    PyDict_SetItemString(stats, "parallel_mode", value);
    Py_DECREF(value);
    value = PyLong_FromLongLong(problem.parallel_workers);
    PyDict_SetItemString(stats, "parallel_workers", value);
    Py_DECREF(value);
    value = PyLong_FromLongLong(problem.diagonal_interval_evaluations);
    PyDict_SetItemString(stats, "diagonal_interval_evaluations", value);
    Py_DECREF(value);
    value = PyLong_FromLongLong(problem.zero_interval_evaluations);
    PyDict_SetItemString(stats, "zero_interval_evaluations", value);
    Py_DECREF(value);
    value = PyBool_FromLong(problem.interaction_dense ? 1 : 0);
    PyDict_SetItemString(stats, "interaction_dense", value);
    Py_DECREF(value);
    value = PyBool_FromLong(problem.lindblad ? 1 : 0);
    PyDict_SetItemString(stats, "lindblad", value);
    Py_DECREF(value);
    value = PyBool_FromLong(problem.lindblad_diagonal_exact ? 1 : 0);
    PyDict_SetItemString(stats, "lindblad_diagonal_exact", value);
    Py_DECREF(value);
    if (problem.lindblad) {
        value = PyLong_FromLong(problem.n);
        PyDict_SetItemString(stats, "physical_dimension", value);
        Py_DECREF(value);
        value = PyLong_FromLong(problem.lindblad_collapse_count);
        PyDict_SetItemString(stats, "collapse_count", value);
        Py_DECREF(value);
    }
    value = PyLong_FromLong(problem.mode);
    PyDict_SetItemString(stats, "mode", value);
    Py_DECREF(value);
    const char* integrator_name = problem.sparse
        ? "cpp_dopri5_csr"
        : (problem.banded
            ? "cpp_dopri5_banded"
            : (problem.fused_sparse
                ? "cpp_dopri5_fused_csr"
                : "cpp_dopri5"));
    if (problem.interval_exponential) {
        integrator_name = "cpp_mixed_exact_expm";
    }
    if (problem.whole_trace_exponential) {
        integrator_name = "cpp_constant_expm";
    }
    if (problem.lindblad) {
        integrator_name = problem.lindblad_diagonal_exact
            ? "cpp_lindblad_diagonal_exact"
            : "cpp_lindblad_matrix_free";
    } else if (problem.diagonal_only) {
        integrator_name = "cpp_diagonal_exact";
    }
    if (problem.interaction_picture) {
        integrator_name = "cpp_interaction_exact_csr";
    }
    if (problem.krylov_used) {
        if (problem.lindblad) {
            integrator_name = problem.piecewise_krylov_used
                ? "cpp_lindblad_piecewise_krylov"
                : "cpp_lindblad_constant_krylov";
        } else {
            integrator_name = problem.piecewise_krylov_used
                ? "cpp_piecewise_krylov_csr"
                : "cpp_constant_krylov_csr";
        }
    }
    value = PyUnicode_FromString(integrator_name);
    PyDict_SetItemString(stats, "integrator", value);
    Py_DECREF(value);

    PyObject* result = PyTuple_New(3);
    if (result == nullptr) {
        Py_DECREF(final_bytes);
        Py_DECREF(trajectory_object);
        Py_DECREF(stats);
        return nullptr;
    }
    PyTuple_SET_ITEM(result, 0, final_bytes);
    PyTuple_SET_ITEM(result, 1, trajectory_object);
    PyTuple_SET_ITEM(result, 2, stats);
    return result;
}

bool validate_polynomial_view(
    const BufferView& view,
    int control_count,
    int interval_count,
    int coefficient_order);

bool validate_control_modes_view(
    const BufferView& view,
    int control_count);

PyObject* native_propagate(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* h0_object = nullptr;
    PyObject* controls_object = nullptr;
    PyObject* iq_object = nullptr;
    PyObject* t_axis_object = nullptr;
    PyObject* initial_object = nullptr;
    PyObject* lo_freqs_object = nullptr;
    int mode = 0;
    double atol = 1e-8;
    double rtol = 1e-6;
    long long max_steps = 2000000;
    int store_trajectory = 0;
    int sparse_expm = 1;
    PyObject* iq_polynomial_object = Py_None;
    int coefficient_order = 1;
    PyObject* control_modes_object = Py_None;
    int parallel = 1;
    static const char* keywords[] = {
        "h0", "controls", "iq", "t_axis", "initial_states", "lo_freqs",
        "mode", "atol", "rtol", "max_steps", "store_trajectory", "sparse_expm",
        "iq_polynomial", "coefficient_order", "control_modes", "parallel", nullptr
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOOOOO|iddLiiOiOi",
            const_cast<char**>(keywords),
            &h0_object,
            &controls_object,
            &iq_object,
            &t_axis_object,
            &initial_object,
            &lo_freqs_object,
            &mode,
            &atol,
            &rtol,
            &max_steps,
            &store_trajectory,
            &sparse_expm,
            &iq_polynomial_object,
            &coefficient_order,
            &control_modes_object,
            &parallel)) {
        return nullptr;
    }
    if (mode != 0 && mode != 1 && mode != 2) {
        PyErr_SetString(PyExc_ValueError, "mode must be 0 (RF), 1 (complex envelope), or 2 (mixed)");
        return nullptr;
    }
    if (!std::isfinite(atol) || !std::isfinite(rtol)
        || !(atol > 0.0) || !(rtol > 0.0) || max_steps < 1) {
        PyErr_SetString(
            PyExc_ValueError,
            "atol and rtol must be finite and positive; max_steps must be positive");
        return nullptr;
    }
    if (sparse_expm < 0 || sparse_expm > 2) {
        PyErr_SetString(PyExc_ValueError, "sparse_expm must be 0 (off), 1 (auto), or 2 (on)");
        return nullptr;
    }
    if (coefficient_order < 0 || coefficient_order > 3) {
        PyErr_SetString(PyExc_ValueError, "coefficient_order must be between 0 and 3");
        return nullptr;
    }
    if (parallel < 0 || parallel > 2) {
        PyErr_SetString(PyExc_ValueError, "parallel must be 0 (off), 1 (auto), or 2 (on)");
        return nullptr;
    }

    BufferView h0_view, controls_view, iq_view, t_view, initial_view, lo_view;
    BufferView iq_polynomial_view;
    const bool has_polynomial = iq_polynomial_object != Py_None;
    BufferView control_modes_view;
    const bool has_control_modes = control_modes_object != Py_None;
    if (!h0_view.acquire(h0_object, "h0", 2, sizeof(cdouble))
        || !controls_view.acquire(controls_object, "controls", 3, sizeof(cdouble))
        || !iq_view.acquire(iq_object, "iq", 2, sizeof(cdouble))
        || !t_view.acquire(t_axis_object, "t_axis", 1, sizeof(double))
        || !initial_view.acquire(initial_object, "initial_states", 2, sizeof(cdouble))
        || !lo_view.acquire(lo_freqs_object, "lo_freqs", 1, sizeof(double))
        || (has_polynomial
            && !iq_polynomial_view.acquire(
                iq_polynomial_object,
                "iq_polynomial",
                3,
                sizeof(cdouble)))
        || (has_control_modes
            && !control_modes_view.acquire(
                control_modes_object,
                "control_modes",
                1,
                sizeof(std::int64_t)))) {
        return nullptr;
    }

    Problem problem;
    problem.n = static_cast<int>(h0_view.view.shape[0]);
    const int h0_columns = static_cast<int>(h0_view.view.shape[1]);
    problem.control_count = static_cast<int>(controls_view.view.shape[0]);
    const int control_rows = static_cast<int>(controls_view.view.shape[1]);
    const int control_columns = static_cast<int>(controls_view.view.shape[2]);
    problem.sample_count = static_cast<int>(iq_view.view.shape[1]);
    const int iq_controls = static_cast<int>(iq_view.view.shape[0]);
    problem.batch_count = static_cast<int>(initial_view.view.shape[1]);
    const int initial_rows = static_cast<int>(initial_view.view.shape[0]);
    const int t_count = static_cast<int>(t_view.view.shape[0]);
    const int lo_count = static_cast<int>(lo_view.view.shape[0]);

    if (problem.n < 1 || h0_columns != problem.n
        || control_rows != problem.n || control_columns != problem.n
        || iq_controls != problem.control_count || problem.sample_count < 1
        || initial_rows != problem.n || problem.batch_count < 1
        || t_count != problem.sample_count || lo_count != problem.control_count) {
        PyErr_SetString(PyExc_ValueError, "native propagation array shapes are inconsistent");
        return nullptr;
    }
    if (has_polynomial) {
        if (!validate_polynomial_view(
                iq_polynomial_view,
                problem.control_count,
                std::max(0, problem.sample_count - 1),
                coefficient_order)) {
            return nullptr;
        }
    } else if (coefficient_order != 1 && problem.control_count > 0) {
        PyErr_SetString(
            PyExc_ValueError,
            "iq_polynomial is required when coefficient_order is not 1");
        return nullptr;
    }
    if (has_control_modes) {
        if (!validate_control_modes_view(control_modes_view, problem.control_count)) {
            return nullptr;
        }
    } else if (mode == 2 && problem.control_count > 0) {
        PyErr_SetString(PyExc_ValueError, "control_modes is required for mixed mode");
        return nullptr;
    }
    const std::size_t h0_size = static_cast<std::size_t>(problem.n) * problem.n;
    const std::size_t controls_size = static_cast<std::size_t>(problem.control_count) * problem.n * problem.n;
    const std::size_t iq_size = static_cast<std::size_t>(problem.control_count) * problem.sample_count;
    const std::size_t initial_size = static_cast<std::size_t>(problem.n) * problem.batch_count;
    for (std::size_t index = 0; index < h0_size; ++index) {
        if (!finite_value(h0_view.data<cdouble>()[index])) {
            PyErr_SetString(PyExc_ValueError, "h0 must contain only finite values");
            return nullptr;
        }
    }
    for (std::size_t index = 0; index < controls_size; ++index) {
        if (!finite_value(controls_view.data<cdouble>()[index])) {
            PyErr_SetString(PyExc_ValueError, "controls must contain only finite values");
            return nullptr;
        }
    }
    for (std::size_t index = 0; index < iq_size; ++index) {
        if (!finite_value(iq_view.data<cdouble>()[index])) {
            PyErr_SetString(PyExc_ValueError, "iq must contain only finite values");
            return nullptr;
        }
    }
    for (std::size_t index = 0; index < initial_size; ++index) {
        if (!finite_value(initial_view.data<cdouble>()[index])) {
            PyErr_SetString(PyExc_ValueError, "initial_states must contain only finite values");
            return nullptr;
        }
    }
    for (int index = 0; index < t_count; ++index) {
        if (!std::isfinite(t_view.data<double>()[index])) {
            PyErr_SetString(PyExc_ValueError, "t_axis must contain only finite values");
            return nullptr;
        }
    }
    for (int index = 0; index < lo_count; ++index) {
        if (!std::isfinite(lo_view.data<double>()[index])) {
            PyErr_SetString(PyExc_ValueError, "lo_freqs must contain only finite values");
            return nullptr;
        }
    }
    problem.h0 = h0_view.data<cdouble>();
    problem.controls = controls_view.data<cdouble>();
    problem.iq = iq_view.data<cdouble>();
    problem.t_axis = t_view.data<double>();
    problem.lo_freqs = lo_view.data<double>();
    problem.mode = mode;
    problem.atol = atol;
    problem.rtol = rtol;
    problem.max_steps = static_cast<std::int64_t>(max_steps);
    problem.sparse_expm_mode = sparse_expm;
    problem.parallel_mode = parallel;
    problem.coefficient_order = coefficient_order;
    problem.polynomial_interval_count = std::max(0, problem.sample_count - 1);
    problem.iq_polynomial = has_polynomial
        ? iq_polynomial_view.data<cdouble>()
        : nullptr;
    problem.control_modes = has_control_modes
        ? control_modes_view.data<std::int64_t>()
        : nullptr;
    problem.t0 = problem.t_axis[0];
    problem.t_last = problem.t_axis[problem.sample_count - 1];
    problem.detect_diagonal_dense();
    problem.angular_freqs.resize(static_cast<std::size_t>(problem.control_count));
    for (int control = 0; control < problem.control_count; ++control) {
        problem.angular_freqs[static_cast<std::size_t>(control)] =
            2.0 * kPi * problem.lo_freqs[control];
        problem.max_frequency = std::max(
            problem.max_frequency,
            std::abs(problem.lo_freqs[control]));
    }

    if (problem.sample_count > 1) {
        problem.source_dt = problem.t_axis[1] - problem.t_axis[0];
        if (!(problem.source_dt > 0.0)) {
            PyErr_SetString(PyExc_ValueError, "t_axis must be strictly increasing");
            return nullptr;
        }
        for (int index = 2; index < problem.sample_count; ++index) {
            const double delta = problem.t_axis[index] - problem.t_axis[index - 1];
            if (!(delta > 0.0)) {
                PyErr_SetString(PyExc_ValueError, "t_axis must be strictly increasing");
                return nullptr;
            }
        }
        problem.inv_source_dt = 1.0 / problem.source_dt;
        problem.prepare_slopes();
    } else {
        problem.source_dt = 1.0;
        problem.inv_source_dt = 1.0;
    }

    problem.prepare_frequency_cache();
    Workspace workspace;
    workspace.resize(
        static_cast<std::size_t>(problem.n) * problem.batch_count,
        problem.batch_count == 1
            ? 0
            : static_cast<std::size_t>(problem.n) * problem.n);
    workspace.resize_scales(static_cast<std::size_t>(problem.control_count));
    workspace.resize_carrier_phases(problem.unique_angular_freqs.size());

    std::vector<cdouble> state(
        static_cast<std::size_t>(problem.n) * problem.batch_count);
    const cdouble* initial_data = initial_view.data<cdouble>();
    std::copy(
        initial_data,
        initial_data + state.size(),
        state.begin());

    std::string error_message;
    std::vector<cdouble> trajectory;
    if (!run_integration(
            problem,
            state,
            workspace,
            store_trajectory != 0,
            trajectory,
            error_message)) {
        PyErr_SetString(PyExc_RuntimeError, error_message.c_str());
        return nullptr;
    }
    return make_native_result(problem, state, trajectory, store_trajectory != 0);
}

bool validate_sparse_matrix(
    const BufferView& data_view,
    const BufferView& indices_view,
    const BufferView& indptr_view,
    int n,
    const char* name,
    std::int64_t expected_offset,
    std::int64_t* nnz_out,
    bool* diagonal_out,
    bool* diagonal_single_out) {
    if (data_view.view.shape[0] != indices_view.view.shape[0]
        || indptr_view.view.shape[0] != static_cast<Py_ssize_t>(n + 1)) {
        PyErr_Format(PyExc_ValueError, "%s CSR shapes are inconsistent", name);
        return false;
    }
    const auto* indptr = indptr_view.data<std::int64_t>();
    const auto* indices = indices_view.data<std::int64_t>();
    const auto nnz = static_cast<std::int64_t>(data_view.view.shape[0]);
    if (indptr[0] != 0 || indptr[n] != expected_offset) {
        PyErr_Format(PyExc_ValueError, "%s CSR indptr does not match its data", name);
        return false;
    }
    for (int row = 0; row < n; ++row) {
        if (indptr[row] > indptr[row + 1]
            || indptr[row] < 0
            || indptr[row + 1] > nnz) {
            PyErr_Format(PyExc_ValueError, "%s CSR indptr must be monotone", name);
            return false;
        }
    }
    bool diagonal = true;
    bool diagonal_single = (nnz == static_cast<std::int64_t>(n));
    for (std::int64_t entry = 0; entry < nnz; ++entry) {
        if (indices[entry] < 0 || indices[entry] >= n
            || !finite_value(data_view.data<cdouble>()[entry])) {
            PyErr_Format(PyExc_ValueError, "%s CSR contains an invalid entry", name);
            return false;
        }
    }
    for (int row = 0; row < n; ++row) {
        if (indptr[row + 1] - indptr[row] != 1) {
            diagonal_single = false;
        }
        for (std::int64_t entry = indptr[row]; entry < indptr[row + 1]; ++entry) {
            if (indices[entry] != row) {
                diagonal = false;
                diagonal_single = false;
                break;
            }
        }
    }
    *nnz_out = nnz;
    *diagonal_out = diagonal;
    *diagonal_single_out = diagonal && diagonal_single;
    return true;
}

bool validate_control_modes_view(
    const BufferView& view,
    int control_count) {
    if (view.view.ndim != 1
        || view.view.shape[0] != static_cast<Py_ssize_t>(control_count)) {
        PyErr_SetString(
            PyExc_ValueError,
            "control_modes must have shape (control_count,)");
        return false;
    }
    const auto* modes = view.data<std::int64_t>();
    for (int control = 0; control < control_count; ++control) {
        if (modes[control] != 0 && modes[control] != 1) {
            PyErr_SetString(PyExc_ValueError, "control_modes entries must be 0 (RF) or 1 (complex envelope)");
            return false;
        }
    }
    return true;
}

OwnedSparseMatrix copy_owned_sparse(
    const cdouble* data,
    const std::int64_t* indices,
    const std::int64_t* indptr,
    int n,
    std::int64_t nnz) {
    OwnedSparseMatrix result;
    if (nnz > 0) {
        result.data.assign(data, data + nnz);
        result.indices.assign(indices, indices + nnz);
    }
    result.indptr.assign(indptr, indptr + n + 1);
    result.diagonal = true;
    result.diagonal_single = (nnz == static_cast<std::int64_t>(n));
    for (int row = 0; row < n; ++row) {
        const std::int64_t begin = result.indptr[static_cast<std::size_t>(row)];
        const std::int64_t end = result.indptr[static_cast<std::size_t>(row) + 1];
        if (end - begin != 1) {
            result.diagonal_single = false;
        }
        for (std::int64_t entry = begin; entry < end; ++entry) {
            if (result.indices[static_cast<std::size_t>(entry)] != row) {
                result.diagonal = false;
                result.diagonal_single = false;
                break;
            }
        }
    }
    return result;
}

OwnedSparseMatrix conjugate_transpose_sparse(
    const OwnedSparseMatrix& source,
    int n) {
    OwnedSparseMatrix result;
    result.indptr.assign(static_cast<std::size_t>(n) + 1, 0);
    for (const std::int64_t column : source.indices) {
        ++result.indptr[static_cast<std::size_t>(column) + 1];
    }
    for (int row = 0; row < n; ++row) {
        result.indptr[static_cast<std::size_t>(row) + 1] +=
            result.indptr[static_cast<std::size_t>(row)];
    }
    result.indices.resize(source.data.size());
    result.data.resize(source.data.size());
    std::vector<std::int64_t> cursor = result.indptr;
    for (int row = 0; row < n; ++row) {
        for (std::int64_t entry = source.indptr[static_cast<std::size_t>(row)];
             entry < source.indptr[static_cast<std::size_t>(row) + 1];
             ++entry) {
            const std::int64_t column = source.indices[static_cast<std::size_t>(entry)];
            const std::int64_t target = cursor[static_cast<std::size_t>(column)]++;
            result.indices[static_cast<std::size_t>(target)] = row;
            result.data[static_cast<std::size_t>(target)] =
                std::conj(source.data[static_cast<std::size_t>(entry)]);
        }
    }
    result.diagonal = source.diagonal;
    result.diagonal_single = source.diagonal_single;
    return result;
}

OwnedSparseMatrix sparse_matrix_product(
    const OwnedSparseMatrix& left,
    const OwnedSparseMatrix& right,
    int n) {
    OwnedSparseMatrix result;
    result.indptr.reserve(static_cast<std::size_t>(n) + 1);
    result.indptr.push_back(0);
    // The previous implementation allocated a tree map for every output
    // row.  A marker/value scratch pair preserves the same accumulation order
    // while making setup linear in the number of touched columns and reusing
    // storage across rows.
    std::vector<int> markers(static_cast<std::size_t>(n), -1);
    std::vector<cdouble> values(static_cast<std::size_t>(n), cdouble(0.0, 0.0));
    std::vector<std::int64_t> touched;
    touched.reserve(static_cast<std::size_t>(n));
    for (int row = 0; row < n; ++row) {
        touched.clear();
        for (std::int64_t left_entry = left.indptr[static_cast<std::size_t>(row)];
             left_entry < left.indptr[static_cast<std::size_t>(row) + 1];
             ++left_entry) {
            const std::int64_t pivot = left.indices[static_cast<std::size_t>(left_entry)];
            const cdouble left_value = left.data[static_cast<std::size_t>(left_entry)];
            for (std::int64_t right_entry = right.indptr[static_cast<std::size_t>(pivot)];
                 right_entry < right.indptr[static_cast<std::size_t>(pivot) + 1];
                 ++right_entry) {
                const std::int64_t column = right.indices[static_cast<std::size_t>(right_entry)];
                const std::size_t column_index = static_cast<std::size_t>(column);
                if (markers[column_index] != row) {
                    markers[column_index] = row;
                    values[column_index] = cdouble(0.0, 0.0);
                    touched.push_back(column);
                }
                values[column_index] += left_value
                    * right.data[static_cast<std::size_t>(right_entry)];
            }
        }
        std::sort(touched.begin(), touched.end());
        for (const std::int64_t column : touched) {
            const cdouble value = values[static_cast<std::size_t>(column)];
            if (value != cdouble(0.0, 0.0)) {
                result.indices.push_back(column);
                result.data.push_back(value);
            }
        }
        result.indptr.push_back(static_cast<std::int64_t>(result.data.size()));
    }
    result.diagonal = true;
    result.diagonal_single = (result.data.size() == static_cast<std::size_t>(n));
    for (int row = 0; row < n; ++row) {
        const std::int64_t begin = result.indptr[static_cast<std::size_t>(row)];
        const std::int64_t end = result.indptr[static_cast<std::size_t>(row) + 1];
        if (end - begin != 1) {
            result.diagonal_single = false;
        }
        for (std::int64_t entry = begin; entry < end; ++entry) {
            if (result.indices[static_cast<std::size_t>(entry)] != row) {
                result.diagonal = false;
                result.diagonal_single = false;
                break;
            }
        }
    }
    return result;
}

LindbladJumpPattern build_lindblad_jump_pattern(
    const OwnedSparseMatrix& collapse,
    int n,
    std::size_t max_entries) {
    LindbladJumpPattern pattern;
    pattern.row_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
    std::vector<int> nonempty_columns;
    nonempty_columns.reserve(static_cast<std::size_t>(n));
    const std::size_t total_nnz = collapse.data.size();
    for (int column = 0; column < n; ++column) {
        if (collapse.indptr[static_cast<std::size_t>(column) + 1]
            > collapse.indptr[static_cast<std::size_t>(column)]) {
            nonempty_columns.push_back(column);
        }
    }
    std::size_t pair_count = 0;
    for (int row = 0; row < n; ++row) {
        const std::size_t left_count = static_cast<std::size_t>(
            collapse.indptr[static_cast<std::size_t>(row) + 1]
            - collapse.indptr[static_cast<std::size_t>(row)]);
        if (total_nnz != 0 && left_count > max_entries / total_nnz) {
            return pattern;
        }
        const std::size_t product = left_count * total_nnz;
        if (pair_count > max_entries - product) {
            return pattern;
        }
        pair_count += product;
    }
    pattern.entries.reserve(pair_count);
    for (int row = 0; row < n; ++row) {
        for (const int column : nonempty_columns) {
            for (std::int64_t left_entry = collapse.indptr[static_cast<std::size_t>(row)];
                 left_entry < collapse.indptr[static_cast<std::size_t>(row) + 1];
                 ++left_entry) {
                const int source_row = static_cast<int>(collapse.indices[
                    static_cast<std::size_t>(left_entry)]);
                const cdouble left_value = collapse.data[
                    static_cast<std::size_t>(left_entry)];
                for (std::int64_t right_entry = collapse.indptr[
                        static_cast<std::size_t>(column)];
                     right_entry < collapse.indptr[
                         static_cast<std::size_t>(column) + 1];
                     ++right_entry) {
                    pattern.entries.push_back({
                        row,
                        column,
                        source_row,
                        static_cast<int>(collapse.indices[
                            static_cast<std::size_t>(right_entry)]),
                        left_value * std::conj(collapse.data[
                            static_cast<std::size_t>(right_entry)]),
                    });
                }
            }
        }
        pattern.row_ptr[static_cast<std::size_t>(row) + 1] =
            static_cast<std::int64_t>(pattern.entries.size());
    }
    pattern.ready = true;
    return pattern;
}

bool validate_csr_bundle(
    const cdouble* data,
    const std::int64_t* indices,
    const std::int64_t* indptr,
    const std::int64_t* offsets,
    int matrix_count,
    int n,
    std::int64_t total_nnz,
    const char* name) {
    if (matrix_count < 0 || offsets == nullptr
        || offsets[0] != 0 || offsets[matrix_count] != total_nnz) {
        PyErr_Format(PyExc_ValueError, "%s CSR offsets are inconsistent", name);
        return false;
    }
    for (int matrix = 0; matrix < matrix_count; ++matrix) {
        const std::int64_t begin_offset = offsets[matrix];
        const std::int64_t end_offset = offsets[matrix + 1];
        if (begin_offset < 0 || end_offset < begin_offset
            || end_offset > total_nnz) {
            PyErr_Format(PyExc_ValueError, "%s CSR offsets must be monotone", name);
            return false;
        }
        const std::int64_t* matrix_indptr = indptr
            + static_cast<std::size_t>(matrix) * (n + 1);
        if (matrix_indptr[0] != 0
            || matrix_indptr[n] != end_offset - begin_offset) {
            PyErr_Format(PyExc_ValueError, "%s CSR indptr does not match offsets", name);
            return false;
        }
        for (int row = 0; row < n; ++row) {
            if (matrix_indptr[row] > matrix_indptr[row + 1]
                || matrix_indptr[row] < 0
                || matrix_indptr[row + 1] > end_offset - begin_offset) {
                PyErr_Format(PyExc_ValueError, "%s CSR indptr must be monotone", name);
                return false;
            }
            std::int64_t previous_column = -1;
            for (std::int64_t entry = matrix_indptr[row];
                 entry < matrix_indptr[row + 1];
                 ++entry) {
                const std::int64_t global_entry = begin_offset + entry;
                const std::int64_t column = indices[global_entry];
                if (column < 0 || column >= n
                    || (previous_column >= 0 && column <= previous_column)
                    || !finite_value(data[global_entry])) {
                    PyErr_Format(PyExc_ValueError, "%s CSR contains invalid or unsorted entries", name);
                    return false;
                }
                previous_column = column;
            }
        }
    }
    return true;
}

bool validate_polynomial_view(
    const BufferView& view,
    int control_count,
    int interval_count,
    int coefficient_order) {
    if (view.view.ndim != 3
        || view.view.shape[0] != static_cast<Py_ssize_t>(control_count)
        || view.view.shape[1] != static_cast<Py_ssize_t>(interval_count)
        || view.view.shape[2] != static_cast<Py_ssize_t>(coefficient_order + 1)) {
        PyErr_SetString(
            PyExc_ValueError,
            "iq_polynomial must have shape (control_count, sample_count-1, coefficient_order+1)");
        return false;
    }
    const std::size_t size = static_cast<std::size_t>(control_count)
        * static_cast<std::size_t>(interval_count)
        * static_cast<std::size_t>(coefficient_order + 1);
    const cdouble* values = view.data<cdouble>();
    for (std::size_t index = 0; index < size; ++index) {
        if (!finite_value(values[index])) {
            PyErr_SetString(PyExc_ValueError, "iq_polynomial must contain only finite values");
            return false;
        }
    }
    return true;
}

PyObject* native_propagate_csr(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* h0_data_object = nullptr;
    PyObject* h0_indices_object = nullptr;
    PyObject* h0_indptr_object = nullptr;
    PyObject* controls_data_object = nullptr;
    PyObject* controls_indices_object = nullptr;
    PyObject* controls_indptr_object = nullptr;
    PyObject* controls_offsets_object = nullptr;
    PyObject* iq_object = nullptr;
    PyObject* t_axis_object = nullptr;
    PyObject* initial_object = nullptr;
    PyObject* lo_freqs_object = nullptr;
    int mode = 0;
    double atol = 1e-8;
    double rtol = 1e-6;
    long long max_steps = 2000000;
    int store_trajectory = 0;
    int sparse_expm = 1;
    PyObject* iq_polynomial_object = Py_None;
    int coefficient_order = 1;
    PyObject* control_modes_object = Py_None;
    int parallel = 1;
    static const char* keywords[] = {
        "h0_data", "h0_indices", "h0_indptr",
        "controls_data", "controls_indices", "controls_indptr", "controls_offsets",
        "iq", "t_axis", "initial_states", "lo_freqs",
        "mode", "atol", "rtol", "max_steps", "store_trajectory", "sparse_expm",
        "iq_polynomial", "coefficient_order", "control_modes", "parallel", nullptr
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOOOOOOOOOO|iddLiiOiOi",
            const_cast<char**>(keywords),
            &h0_data_object,
            &h0_indices_object,
            &h0_indptr_object,
            &controls_data_object,
            &controls_indices_object,
            &controls_indptr_object,
            &controls_offsets_object,
            &iq_object,
            &t_axis_object,
            &initial_object,
            &lo_freqs_object,
            &mode,
            &atol,
            &rtol,
            &max_steps,
            &store_trajectory,
            &sparse_expm,
            &iq_polynomial_object,
            &coefficient_order,
            &control_modes_object,
            &parallel)) {
        return nullptr;
    }
    if (mode != 0 && mode != 1 && mode != 2) {
        PyErr_SetString(PyExc_ValueError, "mode must be 0 (RF), 1 (complex envelope), or 2 (mixed)");
        return nullptr;
    }
    if (!std::isfinite(atol) || !std::isfinite(rtol)
        || !(atol > 0.0) || !(rtol > 0.0) || max_steps < 1) {
        PyErr_SetString(
            PyExc_ValueError,
            "atol and rtol must be finite and positive; max_steps must be positive");
        return nullptr;
    }
    if (sparse_expm < 0 || sparse_expm > 2) {
        PyErr_SetString(PyExc_ValueError, "sparse_expm must be 0 (off), 1 (auto), or 2 (on)");
        return nullptr;
    }
    if (coefficient_order < 0 || coefficient_order > 3) {
        PyErr_SetString(PyExc_ValueError, "coefficient_order must be between 0 and 3");
        return nullptr;
    }
    if (parallel < 0 || parallel > 2) {
        PyErr_SetString(PyExc_ValueError, "parallel must be 0 (off), 1 (auto), or 2 (on)");
        return nullptr;
    }

    BufferView h0_data_view, h0_indices_view, h0_indptr_view;
    BufferView controls_data_view, controls_indices_view, controls_indptr_view, controls_offsets_view;
    BufferView iq_view, t_view, initial_view, lo_view;
    BufferView iq_polynomial_view;
    const bool has_polynomial = iq_polynomial_object != Py_None;
    BufferView control_modes_view;
    const bool has_control_modes = control_modes_object != Py_None;
    if (!h0_data_view.acquire(h0_data_object, "h0_data", 1, sizeof(cdouble))
        || !h0_indices_view.acquire(h0_indices_object, "h0_indices", 1, sizeof(std::int64_t))
        || !h0_indptr_view.acquire(h0_indptr_object, "h0_indptr", 1, sizeof(std::int64_t))
        || !controls_data_view.acquire(controls_data_object, "controls_data", 1, sizeof(cdouble))
        || !controls_indices_view.acquire(controls_indices_object, "controls_indices", 1, sizeof(std::int64_t))
        || !controls_indptr_view.acquire(controls_indptr_object, "controls_indptr", 2, sizeof(std::int64_t))
        || !controls_offsets_view.acquire(controls_offsets_object, "controls_offsets", 1, sizeof(std::int64_t))
        || !iq_view.acquire(iq_object, "iq", 2, sizeof(cdouble))
        || !t_view.acquire(t_axis_object, "t_axis", 1, sizeof(double))
        || !initial_view.acquire(initial_object, "initial_states", 2, sizeof(cdouble))
        || !lo_view.acquire(lo_freqs_object, "lo_freqs", 1, sizeof(double))
        || (has_polynomial
            && !iq_polynomial_view.acquire(
                iq_polynomial_object,
                "iq_polynomial",
                3,
                sizeof(cdouble)))
        || (has_control_modes
            && !control_modes_view.acquire(
                control_modes_object,
                "control_modes",
                1,
                sizeof(std::int64_t)))) {
        return nullptr;
    }

    const int n = static_cast<int>(h0_indptr_view.view.shape[0]) - 1;
    const int control_count = static_cast<int>(controls_indptr_view.view.shape[0]);
    const int sample_count = static_cast<int>(iq_view.view.shape[1]);
    const int batch_count = static_cast<int>(initial_view.view.shape[1]);
    if (n < 1
        || h0_indptr_view.view.shape[0] != static_cast<Py_ssize_t>(n + 1)
        || controls_indptr_view.view.shape[1] != static_cast<Py_ssize_t>(n + 1)
        || controls_offsets_view.view.shape[0] != static_cast<Py_ssize_t>(control_count + 1)
        || iq_view.view.shape[0] != static_cast<Py_ssize_t>(control_count)
        || t_view.view.shape[0] != static_cast<Py_ssize_t>(sample_count)
        || lo_view.view.shape[0] != static_cast<Py_ssize_t>(control_count)
        || initial_view.view.shape[0] != static_cast<Py_ssize_t>(n)
        || sample_count < 1 || batch_count < 1) {
        PyErr_SetString(PyExc_ValueError, "sparse propagation array shapes are inconsistent");
        return nullptr;
    }
    if (has_polynomial) {
        if (!validate_polynomial_view(
                iq_polynomial_view,
                control_count,
                std::max(0, sample_count - 1),
                coefficient_order)) {
            return nullptr;
        }
    } else if (coefficient_order != 1 && control_count > 0) {
        PyErr_SetString(
            PyExc_ValueError,
            "iq_polynomial is required when coefficient_order is not 1");
        return nullptr;
    }
    if (has_control_modes) {
        if (!validate_control_modes_view(control_modes_view, control_count)) {
            return nullptr;
        }
    } else if (mode == 2 && control_count > 0) {
        PyErr_SetString(PyExc_ValueError, "control_modes is required for mixed mode");
        return nullptr;
    }

    std::int64_t h0_nnz = 0;
    bool h0_diagonal = false;
    bool h0_diagonal_single = false;
    if (!validate_sparse_matrix(
            h0_data_view,
            h0_indices_view,
            h0_indptr_view,
            n,
            "h0",
            static_cast<std::int64_t>(h0_data_view.view.shape[0]),
            &h0_nnz,
            &h0_diagonal,
            &h0_diagonal_single)) {
        return nullptr;
    }
    const auto* offsets = controls_offsets_view.data<std::int64_t>();
    const auto* control_indptr = controls_indptr_view.data<std::int64_t>();
    const std::int64_t controls_nnz = static_cast<std::int64_t>(controls_data_view.view.shape[0]);
    if (offsets[0] != 0 || offsets[control_count] != controls_nnz) {
        PyErr_SetString(PyExc_ValueError, "controls_offsets does not match controls_data");
        return nullptr;
    }
    for (int control = 0; control < control_count; ++control) {
        if (offsets[control] > offsets[control + 1]
            || offsets[control] < 0
            || offsets[control + 1] > controls_nnz
            || control_indptr[static_cast<std::size_t>(control) * (n + 1)] != 0
            || control_indptr[static_cast<std::size_t>(control) * (n + 1) + n]
                != offsets[control + 1] - offsets[control]) {
            PyErr_SetString(PyExc_ValueError, "controls CSR offsets/indptr are invalid");
            return nullptr;
        }
        for (int row = 0; row < n; ++row) {
            const auto begin = control_indptr[static_cast<std::size_t>(control) * (n + 1) + row];
            const auto end = control_indptr[static_cast<std::size_t>(control) * (n + 1) + row + 1];
            if (begin > end || begin < 0
                || end > offsets[control + 1] - offsets[control]) {
                PyErr_SetString(PyExc_ValueError, "controls CSR indptr must be monotone");
                return nullptr;
            }
        }
    }
    for (std::int64_t entry = 0; entry < controls_nnz; ++entry) {
        if (controls_indices_view.data<std::int64_t>()[entry] < 0
            || controls_indices_view.data<std::int64_t>()[entry] >= n
            || !finite_value(controls_data_view.data<cdouble>()[entry])) {
            PyErr_SetString(PyExc_ValueError, "controls CSR contains an invalid entry");
            return nullptr;
        }
    }
    for (std::int64_t entry = 0; entry < static_cast<std::int64_t>(iq_view.view.shape[0]) * sample_count; ++entry) {
        if (!finite_value(iq_view.data<cdouble>()[entry])) {
            PyErr_SetString(PyExc_ValueError, "iq must contain only finite values");
            return nullptr;
        }
    }
    for (std::int64_t entry = 0; entry < static_cast<std::int64_t>(initial_view.view.shape[0]) * batch_count; ++entry) {
        if (!finite_value(initial_view.data<cdouble>()[entry])) {
            PyErr_SetString(PyExc_ValueError, "initial_states must contain only finite values");
            return nullptr;
        }
    }
    for (int index = 0; index < sample_count; ++index) {
        if (!std::isfinite(t_view.data<double>()[index])) {
            PyErr_SetString(PyExc_ValueError, "t_axis must contain only finite values");
            return nullptr;
        }
    }
    for (int index = 0; index < control_count; ++index) {
        if (!std::isfinite(lo_view.data<double>()[index])) {
            PyErr_SetString(PyExc_ValueError, "lo_freqs must contain only finite values");
            return nullptr;
        }
    }

    Problem problem;
    problem.n = n;
    problem.control_count = control_count;
    problem.sample_count = sample_count;
    problem.batch_count = batch_count;
    problem.iq = iq_view.data<cdouble>();
    problem.t_axis = t_view.data<double>();
    problem.lo_freqs = lo_view.data<double>();
    problem.mode = mode;
    problem.atol = atol;
    problem.rtol = rtol;
    problem.max_steps = static_cast<std::int64_t>(max_steps);
    problem.sparse_expm_mode = sparse_expm;
    problem.coefficient_order = coefficient_order;
    problem.polynomial_interval_count = std::max(0, sample_count - 1);
    problem.iq_polynomial = has_polynomial
        ? iq_polynomial_view.data<cdouble>()
        : nullptr;
    problem.control_modes = has_control_modes
        ? control_modes_view.data<std::int64_t>()
        : nullptr;
    problem.t0 = problem.t_axis[0];
    problem.t_last = problem.t_axis[sample_count - 1];
    problem.sparse = true;
    problem.parallel_mode = parallel;
    problem.angular_freqs.resize(static_cast<std::size_t>(problem.control_count));
    problem.h0_sparse = {
        h0_data_view.data<cdouble>(),
        h0_indices_view.data<std::int64_t>(),
        h0_indptr_view.data<std::int64_t>(),
        h0_nnz,
        h0_diagonal,
        h0_diagonal_single,
    };
    problem.controls_sparse.reserve(static_cast<std::size_t>(control_count));
    for (int control = 0; control < control_count; ++control) {
        problem.angular_freqs[static_cast<std::size_t>(control)] =
            2.0 * kPi * problem.lo_freqs[control];
        bool control_diagonal = true;
        bool control_diagonal_single =
            offsets[control + 1] - offsets[control] == static_cast<std::int64_t>(n);
        const auto* control_indices_view_data = controls_indices_view.data<std::int64_t>() + offsets[control];
        for (int row = 0; row < n && control_diagonal; ++row) {
            const auto row_begin = control_indptr[static_cast<std::size_t>(control) * (n + 1) + row];
            const auto row_end = control_indptr[static_cast<std::size_t>(control) * (n + 1) + row + 1];
            if (row_end - row_begin != 1) {
                control_diagonal_single = false;
            }
            for (auto entry = row_begin; entry < row_end; ++entry) {
                if (control_indices_view_data[entry] != row) {
                    control_diagonal = false;
                    control_diagonal_single = false;
                    break;
                }
            }
        }
        problem.controls_sparse.push_back({
            controls_data_view.data<cdouble>() + offsets[control],
            controls_indices_view.data<std::int64_t>() + offsets[control],
            control_indptr + static_cast<std::size_t>(control) * (n + 1),
            offsets[control + 1] - offsets[control],
            control_diagonal,
            control_diagonal_single && control_diagonal,
        });
        problem.max_frequency = std::max(
            problem.max_frequency,
            std::abs(problem.lo_freqs[control]));
    }
    problem.detect_diagonal_sparse();
    if (sample_count > 1) {
        problem.source_dt = problem.t_axis[1] - problem.t_axis[0];
        if (!(problem.source_dt > 0.0)) {
            PyErr_SetString(PyExc_ValueError, "t_axis must be strictly increasing");
            return nullptr;
        }
        for (int index = 2; index < sample_count; ++index) {
            const double delta = problem.t_axis[index] - problem.t_axis[index - 1];
            if (!(delta > 0.0)) {
                PyErr_SetString(PyExc_ValueError, "t_axis must be strictly increasing");
                return nullptr;
            }
        }
        problem.inv_source_dt = 1.0 / problem.source_dt;
        problem.prepare_slopes();
    } else {
        problem.source_dt = 1.0;
        problem.inv_source_dt = 1.0;
    }

    std::vector<cdouble> state(
        static_cast<std::size_t>(n) * batch_count);
    std::copy(
        initial_view.data<cdouble>(),
        initial_view.data<cdouble>() + state.size(),
        state.begin());
    problem.prepare_frequency_cache();
    Workspace workspace;
    workspace.resize(state.size(), 0);
    workspace.resize_scales(static_cast<std::size_t>(problem.control_count));
    workspace.resize_carrier_phases(problem.unique_angular_freqs.size());
    std::vector<cdouble> trajectory;
    std::string error_message;
    if (!run_integration(
            problem,
            state,
            workspace,
            store_trajectory != 0,
            trajectory,
            error_message)) {
        PyErr_SetString(PyExc_RuntimeError, error_message.c_str());
        return nullptr;
    }
    return make_native_result(problem, state, trajectory, store_trajectory != 0);
}

PyObject* native_propagate_interaction_csr(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* energies_object = nullptr;
    PyObject* controls_object = nullptr;
    PyObject* indices_object = nullptr;
    PyObject* indptr_object = nullptr;
    PyObject* iq_object = nullptr;
    PyObject* t_axis_object = nullptr;
    PyObject* initial_object = nullptr;
    PyObject* lo_freqs_object = nullptr;
    int mode = 0;
    double atol = 1e-8;
    double rtol = 1e-6;
    long long max_steps = 2000000;
    int store_trajectory = 0;
    int sparse_expm = 1;
    PyObject* iq_polynomial_object = Py_None;
    int coefficient_order = 1;
    PyObject* control_modes_object = Py_None;
    int parallel = 1;
    static const char* keywords[] = {
        "diagonal_energies", "controls_data", "indices", "indptr",
        "iq", "t_axis", "initial_states", "lo_freqs",
        "mode", "atol", "rtol", "max_steps", "store_trajectory", "sparse_expm",
        "iq_polynomial", "coefficient_order", "control_modes", "parallel", nullptr
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOOOOOOO|iddLiiOiOi",
            const_cast<char**>(keywords),
            &energies_object,
            &controls_object,
            &indices_object,
            &indptr_object,
            &iq_object,
            &t_axis_object,
            &initial_object,
            &lo_freqs_object,
            &mode,
            &atol,
            &rtol,
            &max_steps,
            &store_trajectory,
            &sparse_expm,
            &iq_polynomial_object,
            &coefficient_order,
            &control_modes_object,
            &parallel)) {
        return nullptr;
    }
    if (mode != 0 && mode != 1 && mode != 2) {
        PyErr_SetString(PyExc_ValueError, "mode must be 0 (RF), 1 (complex envelope), or 2 (mixed)");
        return nullptr;
    }
    if (!std::isfinite(atol) || !std::isfinite(rtol)
        || !(atol > 0.0) || !(rtol > 0.0) || max_steps < 1) {
        PyErr_SetString(
            PyExc_ValueError,
            "atol and rtol must be finite and positive; max_steps must be positive");
        return nullptr;
    }
    if (sparse_expm < 0 || sparse_expm > 2) {
        PyErr_SetString(PyExc_ValueError, "sparse_expm must be 0 (off), 1 (auto), or 2 (on)");
        return nullptr;
    }
    if (coefficient_order < 0 || coefficient_order > 3) {
        PyErr_SetString(PyExc_ValueError, "coefficient_order must be between 0 and 3");
        return nullptr;
    }
    if (parallel < 0 || parallel > 2) {
        PyErr_SetString(PyExc_ValueError, "parallel must be 0 (off), 1 (auto), or 2 (on)");
        return nullptr;
    }

    BufferView energies_view, controls_view, indices_view, indptr_view;
    BufferView iq_view, t_view, initial_view, lo_view;
    BufferView iq_polynomial_view;
    const bool has_polynomial = iq_polynomial_object != Py_None;
    BufferView control_modes_view;
    const bool has_control_modes = control_modes_object != Py_None;
    if (!energies_view.acquire(energies_object, "diagonal_energies", 1, sizeof(cdouble))
        || !controls_view.acquire(controls_object, "controls_data", 2, sizeof(cdouble))
        || !indices_view.acquire(indices_object, "indices", 1, sizeof(std::int64_t))
        || !indptr_view.acquire(indptr_object, "indptr", 1, sizeof(std::int64_t))
        || !iq_view.acquire(iq_object, "iq", 2, sizeof(cdouble))
        || !t_view.acquire(t_axis_object, "t_axis", 1, sizeof(double))
        || !initial_view.acquire(initial_object, "initial_states", 2, sizeof(cdouble))
        || !lo_view.acquire(lo_freqs_object, "lo_freqs", 1, sizeof(double))
        || (has_polynomial
            && !iq_polynomial_view.acquire(
                iq_polynomial_object,
                "iq_polynomial",
                3,
                sizeof(cdouble)))
        || (has_control_modes
            && !control_modes_view.acquire(
                control_modes_object,
                "control_modes",
                1,
                sizeof(std::int64_t)))) {
        return nullptr;
    }

    const int n = static_cast<int>(energies_view.view.shape[0]);
    const std::int64_t nnz = static_cast<std::int64_t>(indices_view.view.shape[0]);
    const int control_count = static_cast<int>(controls_view.view.shape[0]);
    const std::int64_t control_nnz = static_cast<std::int64_t>(controls_view.view.shape[1]);
    const int sample_count = static_cast<int>(iq_view.view.shape[1]);
    const int batch_count = static_cast<int>(initial_view.view.shape[1]);
    if (n < 1
        || indptr_view.view.shape[0] != static_cast<Py_ssize_t>(n + 1)
        || control_nnz != nnz
        || iq_view.view.shape[0] != static_cast<Py_ssize_t>(control_count)
        || t_view.view.shape[0] != static_cast<Py_ssize_t>(sample_count)
        || lo_view.view.shape[0] != static_cast<Py_ssize_t>(control_count)
        || initial_view.view.shape[0] != static_cast<Py_ssize_t>(n)
        || sample_count < 1
        || batch_count < 1) {
        PyErr_SetString(PyExc_ValueError, "interaction CSR array shapes are inconsistent");
        return nullptr;
    }
    if (has_polynomial) {
        if (!validate_polynomial_view(
                iq_polynomial_view,
                control_count,
                std::max(0, sample_count - 1),
                coefficient_order)) {
            return nullptr;
        }
    } else if (coefficient_order != 1 && control_count > 0) {
        PyErr_SetString(
            PyExc_ValueError,
            "iq_polynomial is required when coefficient_order is not 1");
        return nullptr;
    }
    if (has_control_modes) {
        if (!validate_control_modes_view(control_modes_view, control_count)) {
            return nullptr;
        }
    } else if (mode == 2 && control_count > 0) {
        PyErr_SetString(PyExc_ValueError, "control_modes is required for mixed mode");
        return nullptr;
    }

    const auto* indptr = indptr_view.data<std::int64_t>();
    const auto* indices = indices_view.data<std::int64_t>();
    if (indptr[0] != 0 || indptr[n] != nnz) {
        PyErr_SetString(PyExc_ValueError, "interaction CSR indptr does not match indices");
        return nullptr;
    }
    for (int row = 0; row < n; ++row) {
        if (indptr[row] > indptr[row + 1]
            || indptr[row] < 0
            || indptr[row + 1] > nnz) {
            PyErr_SetString(PyExc_ValueError, "interaction CSR indptr must be monotone");
            return nullptr;
        }
        for (std::int64_t entry = indptr[row]; entry < indptr[row + 1]; ++entry) {
            if (indices[entry] < 0 || indices[entry] >= n) {
                PyErr_SetString(PyExc_ValueError, "interaction CSR index is out of range");
                return nullptr;
            }
            if (entry > indptr[row] && indices[entry] < indices[entry - 1]) {
                PyErr_SetString(PyExc_ValueError, "interaction CSR indices must be sorted");
                return nullptr;
            }
        }
    }
    const std::size_t energy_size = static_cast<std::size_t>(n);
    for (std::size_t index = 0; index < energy_size; ++index) {
        const cdouble energy = energies_view.data<cdouble>()[index];
        if (!finite_value(energy) || energy.imag() != 0.0) {
            PyErr_SetString(
                PyExc_ValueError,
                "diagonal_energies must be finite real values");
            return nullptr;
        }
    }
    const std::size_t controls_size =
        static_cast<std::size_t>(control_count) * static_cast<std::size_t>(nnz);
    for (std::size_t index = 0; index < controls_size; ++index) {
        if (!finite_value(controls_view.data<cdouble>()[index])) {
            PyErr_SetString(PyExc_ValueError, "interaction controls must contain only finite values");
            return nullptr;
        }
    }
    const std::size_t iq_size = static_cast<std::size_t>(control_count) * sample_count;
    for (std::size_t index = 0; index < iq_size; ++index) {
        if (!finite_value(iq_view.data<cdouble>()[index])) {
            PyErr_SetString(PyExc_ValueError, "iq must contain only finite values");
            return nullptr;
        }
    }
    const std::size_t initial_size = static_cast<std::size_t>(n) * batch_count;
    for (std::size_t index = 0; index < initial_size; ++index) {
        if (!finite_value(initial_view.data<cdouble>()[index])) {
            PyErr_SetString(PyExc_ValueError, "initial_states must contain only finite values");
            return nullptr;
        }
    }
    for (int index = 0; index < sample_count; ++index) {
        if (!std::isfinite(t_view.data<double>()[index])) {
            PyErr_SetString(PyExc_ValueError, "t_axis must contain only finite values");
            return nullptr;
        }
    }
    for (int index = 0; index < control_count; ++index) {
        if (!std::isfinite(lo_view.data<double>()[index])) {
            PyErr_SetString(PyExc_ValueError, "lo_freqs must contain only finite values");
            return nullptr;
        }
    }

    Problem problem;
    problem.n = n;
    problem.control_count = control_count;
    problem.sample_count = sample_count;
    problem.batch_count = batch_count;
    problem.iq = iq_view.data<cdouble>();
    problem.t_axis = t_view.data<double>();
    problem.lo_freqs = lo_view.data<double>();
    problem.mode = mode;
    problem.atol = atol;
    problem.rtol = rtol;
    problem.max_steps = static_cast<std::int64_t>(max_steps);
    problem.sparse_expm_mode = sparse_expm;
    problem.coefficient_order = coefficient_order;
    problem.polynomial_interval_count = std::max(0, sample_count - 1);
    problem.iq_polynomial = has_polynomial
        ? iq_polynomial_view.data<cdouble>()
        : nullptr;
    problem.control_modes = has_control_modes
        ? control_modes_view.data<std::int64_t>()
        : nullptr;
    problem.t0 = problem.t_axis[0];
    problem.t_last = problem.t_axis[sample_count - 1];
    problem.interaction_picture = true;
    problem.parallel_mode = parallel;
    problem.fused_sparse = true;
    problem.interaction_energies = energies_view.data<cdouble>();
    problem.fused_controls = controls_view.data<cdouble>();
    problem.fused_indices = indices;
    problem.fused_indptr = indptr;
    problem.fused_nnz = nnz;
    problem.interaction_dense = problem.detect_full_interaction_pattern();
    problem.prepare_fused_control_entries();
    problem.prepare_interaction_deltas();
    problem.angular_freqs.resize(static_cast<std::size_t>(control_count));
    for (int control = 0; control < control_count; ++control) {
        problem.angular_freqs[static_cast<std::size_t>(control)] =
            2.0 * kPi * problem.lo_freqs[control];
        problem.max_frequency = std::max(
            problem.max_frequency,
            std::abs(problem.lo_freqs[control]));
    }
    if (sample_count > 1) {
        problem.source_dt = problem.t_axis[1] - problem.t_axis[0];
        if (!(problem.source_dt > 0.0)) {
            PyErr_SetString(PyExc_ValueError, "t_axis must be strictly increasing");
            return nullptr;
        }
        for (int index = 2; index < sample_count; ++index) {
            const double delta = problem.t_axis[index] - problem.t_axis[index - 1];
            if (!(delta > 0.0)) {
                PyErr_SetString(PyExc_ValueError, "t_axis must be strictly increasing");
                return nullptr;
            }
        }
        problem.inv_source_dt = 1.0 / problem.source_dt;
        problem.prepare_slopes();
    } else {
        problem.source_dt = 1.0;
        problem.inv_source_dt = 1.0;
    }

    std::vector<cdouble> state(static_cast<std::size_t>(n) * batch_count);
    std::copy(
        initial_view.data<cdouble>(),
        initial_view.data<cdouble>() + state.size(),
        state.begin());
    problem.prepare_frequency_cache();
    Workspace workspace;
    workspace.resize(state.size(), 0);
    workspace.resize_scales(static_cast<std::size_t>(control_count));
    workspace.resize_carrier_phases(problem.unique_angular_freqs.size());
    workspace.resize_interaction_phases(problem.interaction_unique_deltas.size());
    if (problem.interaction_dense) {
        workspace.resize_interaction_dense(
            static_cast<std::size_t>(problem.n),
            state.size());
    }
    std::vector<cdouble> trajectory;
    std::string error_message;
    if (!run_integration(
            problem,
            state,
            workspace,
            store_trajectory != 0,
            trajectory,
            error_message)) {
        PyErr_SetString(PyExc_RuntimeError, error_message.c_str());
        return nullptr;
    }
    return make_native_result(problem, state, trajectory, store_trajectory != 0);
}

PyObject* native_propagate_banded(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* h0_banded_object = nullptr;
    PyObject* controls_banded_object = nullptr;
    PyObject* band_offsets_object = nullptr;
    PyObject* iq_object = nullptr;
    PyObject* t_axis_object = nullptr;
    PyObject* initial_object = nullptr;
    PyObject* lo_freqs_object = nullptr;
    int mode = 0;
    double atol = 1e-8;
    double rtol = 1e-6;
    long long max_steps = 2000000;
    int store_trajectory = 0;
    int sparse_expm = 1;
    PyObject* iq_polynomial_object = Py_None;
    int coefficient_order = 1;
    PyObject* control_modes_object = Py_None;
    int parallel = 1;
    static const char* keywords[] = {
        "h0_banded", "controls_banded", "band_offsets",
        "iq", "t_axis", "initial_states", "lo_freqs",
        "mode", "atol", "rtol", "max_steps", "store_trajectory", "sparse_expm",
        "iq_polynomial", "coefficient_order", "control_modes", "parallel", nullptr
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOOOOOO|iddLiiOiOi",
            const_cast<char**>(keywords),
            &h0_banded_object,
            &controls_banded_object,
            &band_offsets_object,
            &iq_object,
            &t_axis_object,
            &initial_object,
            &lo_freqs_object,
            &mode,
            &atol,
            &rtol,
            &max_steps,
            &store_trajectory,
            &sparse_expm,
            &iq_polynomial_object,
            &coefficient_order,
            &control_modes_object,
            &parallel)) {
        return nullptr;
    }
    if (mode != 0 && mode != 1 && mode != 2) {
        PyErr_SetString(PyExc_ValueError, "mode must be 0 (RF), 1 (complex envelope), or 2 (mixed)");
        return nullptr;
    }
    if (!std::isfinite(atol) || !std::isfinite(rtol)
        || !(atol > 0.0) || !(rtol > 0.0) || max_steps < 1) {
        PyErr_SetString(
            PyExc_ValueError,
            "atol and rtol must be finite and positive; max_steps must be positive");
        return nullptr;
    }
    if (sparse_expm < 0 || sparse_expm > 2) {
        PyErr_SetString(PyExc_ValueError, "sparse_expm must be 0 (off), 1 (auto), or 2 (on)");
        return nullptr;
    }
    if (coefficient_order < 0 || coefficient_order > 3) {
        PyErr_SetString(PyExc_ValueError, "coefficient_order must be between 0 and 3");
        return nullptr;
    }
    if (parallel < 0 || parallel > 2) {
        PyErr_SetString(PyExc_ValueError, "parallel must be 0 (off), 1 (auto), or 2 (on)");
        return nullptr;
    }

    BufferView h0_view, controls_view, offsets_view;
    BufferView iq_view, t_view, initial_view, lo_view;
    BufferView iq_polynomial_view;
    const bool has_polynomial = iq_polynomial_object != Py_None;
    BufferView control_modes_view;
    const bool has_control_modes = control_modes_object != Py_None;
    if (!h0_view.acquire(h0_banded_object, "h0_banded", 2, sizeof(cdouble))
        || !controls_view.acquire(controls_banded_object, "controls_banded", 3, sizeof(cdouble))
        || !offsets_view.acquire(band_offsets_object, "band_offsets", 1, sizeof(std::int64_t))
        || !iq_view.acquire(iq_object, "iq", 2, sizeof(cdouble))
        || !t_view.acquire(t_axis_object, "t_axis", 1, sizeof(double))
        || !initial_view.acquire(initial_object, "initial_states", 2, sizeof(cdouble))
        || !lo_view.acquire(lo_freqs_object, "lo_freqs", 1, sizeof(double))
        || (has_polynomial
            && !iq_polynomial_view.acquire(
                iq_polynomial_object,
                "iq_polynomial",
                3,
                sizeof(cdouble)))
        || (has_control_modes
            && !control_modes_view.acquire(
                control_modes_object,
                "control_modes",
                1,
                sizeof(std::int64_t)))) {
        return nullptr;
    }

    const int band_count = static_cast<int>(h0_view.view.shape[0]);
    const int n = static_cast<int>(h0_view.view.shape[1]);
    const int control_count = static_cast<int>(controls_view.view.shape[0]);
    const int control_bands = static_cast<int>(controls_view.view.shape[1]);
    const int control_dimension = static_cast<int>(controls_view.view.shape[2]);
    const int sample_count = static_cast<int>(iq_view.view.shape[1]);
    const int batch_count = static_cast<int>(initial_view.view.shape[1]);
    if (band_count < 1
        || n < 1
        || offsets_view.view.shape[0] != static_cast<Py_ssize_t>(band_count)
        || control_bands != band_count
        || control_dimension != n
        || iq_view.view.shape[0] != static_cast<Py_ssize_t>(control_count)
        || t_view.view.shape[0] != static_cast<Py_ssize_t>(sample_count)
        || lo_view.view.shape[0] != static_cast<Py_ssize_t>(control_count)
        || initial_view.view.shape[0] != static_cast<Py_ssize_t>(n)
        || sample_count < 1
        || batch_count < 1) {
        PyErr_SetString(PyExc_ValueError, "banded propagation array shapes are inconsistent");
        return nullptr;
    }
    if (has_polynomial) {
        if (!validate_polynomial_view(
                iq_polynomial_view,
                control_count,
                std::max(0, sample_count - 1),
                coefficient_order)) {
            return nullptr;
        }
    } else if (coefficient_order != 1 && control_count > 0) {
        PyErr_SetString(
            PyExc_ValueError,
            "iq_polynomial is required when coefficient_order is not 1");
        return nullptr;
    }
    if (has_control_modes) {
        if (!validate_control_modes_view(control_modes_view, control_count)) {
            return nullptr;
        }
    } else if (mode == 2 && control_count > 0) {
        PyErr_SetString(PyExc_ValueError, "control_modes is required for mixed mode");
        return nullptr;
    }
    const auto* band_offsets = offsets_view.data<std::int64_t>();
    for (int band = 0; band < band_count; ++band) {
        if (band_offsets[band] < -static_cast<std::int64_t>(n - 1)
            || band_offsets[band] > static_cast<std::int64_t>(n - 1)
            || (band > 0 && band_offsets[band] <= band_offsets[band - 1])) {
            PyErr_SetString(PyExc_ValueError, "band_offsets must be strictly increasing and in range");
            return nullptr;
        }
    }

    const std::size_t h0_size = static_cast<std::size_t>(band_count) * n;
    const std::size_t controls_size =
        static_cast<std::size_t>(control_count) * band_count * n;
    const std::size_t iq_size = static_cast<std::size_t>(control_count) * sample_count;
    const std::size_t initial_size = static_cast<std::size_t>(n) * batch_count;
    for (std::size_t index = 0; index < h0_size; ++index) {
        if (!finite_value(h0_view.data<cdouble>()[index])) {
            PyErr_SetString(PyExc_ValueError, "h0_banded must contain only finite values");
            return nullptr;
        }
    }
    for (std::size_t index = 0; index < controls_size; ++index) {
        if (!finite_value(controls_view.data<cdouble>()[index])) {
            PyErr_SetString(PyExc_ValueError, "controls_banded must contain only finite values");
            return nullptr;
        }
    }
    for (std::size_t index = 0; index < iq_size; ++index) {
        if (!finite_value(iq_view.data<cdouble>()[index])) {
            PyErr_SetString(PyExc_ValueError, "iq must contain only finite values");
            return nullptr;
        }
    }
    for (std::size_t index = 0; index < initial_size; ++index) {
        if (!finite_value(initial_view.data<cdouble>()[index])) {
            PyErr_SetString(PyExc_ValueError, "initial_states must contain only finite values");
            return nullptr;
        }
    }
    for (int index = 0; index < sample_count; ++index) {
        if (!std::isfinite(t_view.data<double>()[index])) {
            PyErr_SetString(PyExc_ValueError, "t_axis must contain only finite values");
            return nullptr;
        }
    }
    for (int index = 0; index < control_count; ++index) {
        if (!std::isfinite(lo_view.data<double>()[index])) {
            PyErr_SetString(PyExc_ValueError, "lo_freqs must contain only finite values");
            return nullptr;
        }
    }

    Problem problem;
    problem.n = n;
    problem.control_count = control_count;
    problem.sample_count = sample_count;
    problem.batch_count = batch_count;
    problem.iq = iq_view.data<cdouble>();
    problem.t_axis = t_view.data<double>();
    problem.lo_freqs = lo_view.data<double>();
    problem.mode = mode;
    problem.atol = atol;
    problem.rtol = rtol;
    problem.max_steps = static_cast<std::int64_t>(max_steps);
    problem.sparse_expm_mode = sparse_expm;
    problem.coefficient_order = coefficient_order;
    problem.polynomial_interval_count = std::max(0, sample_count - 1);
    problem.iq_polynomial = has_polynomial
        ? iq_polynomial_view.data<cdouble>()
        : nullptr;
    problem.control_modes = has_control_modes
        ? control_modes_view.data<std::int64_t>()
        : nullptr;
    problem.t0 = problem.t_axis[0];
    problem.t_last = problem.t_axis[sample_count - 1];
    problem.banded = true;
    problem.parallel_mode = parallel;
    problem.h0_banded = h0_view.data<cdouble>();
    problem.controls_banded = controls_view.data<cdouble>();
    problem.band_offsets = band_offsets;
    problem.band_count = band_count;
    problem.prepare_banded_entries();
    problem.detect_diagonal_banded();
    problem.angular_freqs.resize(static_cast<std::size_t>(control_count));
    for (int control = 0; control < control_count; ++control) {
        problem.angular_freqs[static_cast<std::size_t>(control)] =
            2.0 * kPi * problem.lo_freqs[control];
        problem.max_frequency = std::max(
            problem.max_frequency,
            std::abs(problem.lo_freqs[control]));
    }

    if (sample_count > 1) {
        problem.source_dt = problem.t_axis[1] - problem.t_axis[0];
        if (!(problem.source_dt > 0.0)) {
            PyErr_SetString(PyExc_ValueError, "t_axis must be strictly increasing");
            return nullptr;
        }
        for (int index = 2; index < sample_count; ++index) {
            const double delta = problem.t_axis[index] - problem.t_axis[index - 1];
            if (!(delta > 0.0)) {
                PyErr_SetString(PyExc_ValueError, "t_axis must be strictly increasing");
                return nullptr;
            }
        }
        problem.inv_source_dt = 1.0 / problem.source_dt;
        problem.prepare_slopes();
    } else {
        problem.source_dt = 1.0;
        problem.inv_source_dt = 1.0;
    }

    std::vector<cdouble> state(static_cast<std::size_t>(n) * batch_count);
    std::copy(
        initial_view.data<cdouble>(),
        initial_view.data<cdouble>() + state.size(),
        state.begin());
    problem.prepare_frequency_cache();
    Workspace workspace;
    workspace.resize(state.size(), 0);
    workspace.resize_scales(static_cast<std::size_t>(problem.control_count));
    workspace.resize_carrier_phases(problem.unique_angular_freqs.size());
    std::vector<cdouble> trajectory;
    std::string error_message;
    if (!run_integration(
            problem,
            state,
            workspace,
            store_trajectory != 0,
            trajectory,
            error_message)) {
        PyErr_SetString(PyExc_RuntimeError, error_message.c_str());
        return nullptr;
    }
    return make_native_result(problem, state, trajectory, store_trajectory != 0);
}

PyObject* native_propagate_fused_csr(PyObject*, PyObject* args, PyObject* kwargs) {
    PyObject* static_data_object = nullptr;
    PyObject* indices_object = nullptr;
    PyObject* indptr_object = nullptr;
    PyObject* controls_data_object = nullptr;
    PyObject* iq_object = nullptr;
    PyObject* t_axis_object = nullptr;
    PyObject* initial_object = nullptr;
    PyObject* lo_freqs_object = nullptr;
    int mode = 0;
    double atol = 1e-8;
    double rtol = 1e-6;
    long long max_steps = 2000000;
    int store_trajectory = 0;
    int sparse_expm = 1;
    PyObject* iq_polynomial_object = Py_None;
    int coefficient_order = 1;
    PyObject* control_modes_object = Py_None;
    int parallel = 1;
    static const char* keywords[] = {
        "static_data", "indices", "indptr", "controls_data",
        "iq", "t_axis", "initial_states", "lo_freqs",
        "mode", "atol", "rtol", "max_steps", "store_trajectory", "sparse_expm",
        "iq_polynomial", "coefficient_order", "control_modes", "parallel", nullptr
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOOOOOOO|iddLiiOiOi",
            const_cast<char**>(keywords),
            &static_data_object,
            &indices_object,
            &indptr_object,
            &controls_data_object,
            &iq_object,
            &t_axis_object,
            &initial_object,
            &lo_freqs_object,
            &mode,
            &atol,
            &rtol,
            &max_steps,
            &store_trajectory,
            &sparse_expm,
            &iq_polynomial_object,
            &coefficient_order,
            &control_modes_object,
            &parallel)) {
        return nullptr;
    }
    if (mode != 0 && mode != 1 && mode != 2) {
        PyErr_SetString(PyExc_ValueError, "mode must be 0 (RF), 1 (complex envelope), or 2 (mixed)");
        return nullptr;
    }
    if (!std::isfinite(atol) || !std::isfinite(rtol)
        || !(atol > 0.0) || !(rtol > 0.0) || max_steps < 1) {
        PyErr_SetString(
            PyExc_ValueError,
            "atol and rtol must be finite and positive; max_steps must be positive");
        return nullptr;
    }
    if (sparse_expm < 0 || sparse_expm > 2) {
        PyErr_SetString(PyExc_ValueError, "sparse_expm must be 0 (off), 1 (auto), or 2 (on)");
        return nullptr;
    }
    if (coefficient_order < 0 || coefficient_order > 3) {
        PyErr_SetString(PyExc_ValueError, "coefficient_order must be between 0 and 3");
        return nullptr;
    }
    if (parallel < 0 || parallel > 2) {
        PyErr_SetString(PyExc_ValueError, "parallel must be 0 (off), 1 (auto), or 2 (on)");
        return nullptr;
    }

    BufferView static_view, indices_view, indptr_view, controls_view;
    BufferView iq_view, t_view, initial_view, lo_view;
    BufferView iq_polynomial_view;
    const bool has_polynomial = iq_polynomial_object != Py_None;
    BufferView control_modes_view;
    const bool has_control_modes = control_modes_object != Py_None;
    if (!static_view.acquire(static_data_object, "static_data", 1, sizeof(cdouble))
        || !indices_view.acquire(indices_object, "indices", 1, sizeof(std::int64_t))
        || !indptr_view.acquire(indptr_object, "indptr", 1, sizeof(std::int64_t))
        || !controls_view.acquire(controls_data_object, "controls_data", 2, sizeof(cdouble))
        || !iq_view.acquire(iq_object, "iq", 2, sizeof(cdouble))
        || !t_view.acquire(t_axis_object, "t_axis", 1, sizeof(double))
        || !initial_view.acquire(initial_object, "initial_states", 2, sizeof(cdouble))
        || !lo_view.acquire(lo_freqs_object, "lo_freqs", 1, sizeof(double))
        || (has_polynomial
            && !iq_polynomial_view.acquire(
                iq_polynomial_object,
                "iq_polynomial",
                3,
                sizeof(cdouble)))
        || (has_control_modes
            && !control_modes_view.acquire(
                control_modes_object,
                "control_modes",
                1,
                sizeof(std::int64_t)))) {
        return nullptr;
    }

    const int n = static_cast<int>(indptr_view.view.shape[0]) - 1;
    const std::int64_t nnz = static_cast<std::int64_t>(static_view.view.shape[0]);
    const int control_count = static_cast<int>(controls_view.view.shape[0]);
    const int control_nnz = static_cast<int>(controls_view.view.shape[1]);
    const int sample_count = static_cast<int>(iq_view.view.shape[1]);
    const int batch_count = static_cast<int>(initial_view.view.shape[1]);
    if (n < 1
        || indptr_view.view.shape[0] != static_cast<Py_ssize_t>(n + 1)
        || indices_view.view.shape[0] != static_cast<Py_ssize_t>(nnz)
        || control_nnz != nnz
        || iq_view.view.shape[0] != static_cast<Py_ssize_t>(control_count)
        || t_view.view.shape[0] != static_cast<Py_ssize_t>(sample_count)
        || lo_view.view.shape[0] != static_cast<Py_ssize_t>(control_count)
        || initial_view.view.shape[0] != static_cast<Py_ssize_t>(n)
        || sample_count < 1
        || batch_count < 1) {
        PyErr_SetString(PyExc_ValueError, "fused CSR propagation array shapes are inconsistent");
        return nullptr;
    }
    if (has_polynomial) {
        if (!validate_polynomial_view(
                iq_polynomial_view,
                control_count,
                std::max(0, sample_count - 1),
                coefficient_order)) {
            return nullptr;
        }
    } else if (coefficient_order != 1 && control_count > 0) {
        PyErr_SetString(
            PyExc_ValueError,
            "iq_polynomial is required when coefficient_order is not 1");
        return nullptr;
    }
    if (has_control_modes) {
        if (!validate_control_modes_view(control_modes_view, control_count)) {
            return nullptr;
        }
    } else if (mode == 2 && control_count > 0) {
        PyErr_SetString(PyExc_ValueError, "control_modes is required for mixed mode");
        return nullptr;
    }
    const auto* indptr = indptr_view.data<std::int64_t>();
    const auto* indices = indices_view.data<std::int64_t>();
    if (indptr[0] != 0 || indptr[n] != nnz) {
        PyErr_SetString(PyExc_ValueError, "fused CSR indptr does not match static_data");
        return nullptr;
    }
    for (int row = 0; row < n; ++row) {
        if (indptr[row] > indptr[row + 1]
            || indptr[row] < 0
            || indptr[row + 1] > nnz) {
            PyErr_SetString(PyExc_ValueError, "fused CSR indptr must be monotone");
            return nullptr;
        }
    }
    for (std::int64_t entry = 0; entry < nnz; ++entry) {
        if (indices[entry] < 0 || indices[entry] >= n
            || !finite_value(static_view.data<cdouble>()[entry])) {
            PyErr_SetString(PyExc_ValueError, "fused CSR contains an invalid static entry");
            return nullptr;
        }
    }
    const std::size_t controls_size =
        static_cast<std::size_t>(control_count) * static_cast<std::size_t>(nnz);
    for (std::size_t entry = 0; entry < controls_size; ++entry) {
        if (!finite_value(controls_view.data<cdouble>()[entry])) {
            PyErr_SetString(PyExc_ValueError, "fused CSR controls contain non-finite values");
            return nullptr;
        }
    }
    const std::size_t iq_size = static_cast<std::size_t>(control_count) * sample_count;
    for (std::size_t entry = 0; entry < iq_size; ++entry) {
        if (!finite_value(iq_view.data<cdouble>()[entry])) {
            PyErr_SetString(PyExc_ValueError, "iq must contain only finite values");
            return nullptr;
        }
    }
    const std::size_t initial_size = static_cast<std::size_t>(n) * batch_count;
    for (std::size_t entry = 0; entry < initial_size; ++entry) {
        if (!finite_value(initial_view.data<cdouble>()[entry])) {
            PyErr_SetString(PyExc_ValueError, "initial_states must contain only finite values");
            return nullptr;
        }
    }
    for (int index = 0; index < sample_count; ++index) {
        if (!std::isfinite(t_view.data<double>()[index])) {
            PyErr_SetString(PyExc_ValueError, "t_axis must contain only finite values");
            return nullptr;
        }
    }
    for (int index = 0; index < control_count; ++index) {
        if (!std::isfinite(lo_view.data<double>()[index])) {
            PyErr_SetString(PyExc_ValueError, "lo_freqs must contain only finite values");
            return nullptr;
        }
    }

    Problem problem;
    problem.n = n;
    problem.control_count = control_count;
    problem.sample_count = sample_count;
    problem.batch_count = batch_count;
    problem.iq = iq_view.data<cdouble>();
    problem.t_axis = t_view.data<double>();
    problem.lo_freqs = lo_view.data<double>();
    problem.mode = mode;
    problem.atol = atol;
    problem.rtol = rtol;
    problem.max_steps = static_cast<std::int64_t>(max_steps);
    problem.sparse_expm_mode = sparse_expm;
    problem.coefficient_order = coefficient_order;
    problem.polynomial_interval_count = std::max(0, sample_count - 1);
    problem.iq_polynomial = has_polynomial
        ? iq_polynomial_view.data<cdouble>()
        : nullptr;
    problem.control_modes = has_control_modes
        ? control_modes_view.data<std::int64_t>()
        : nullptr;
    problem.t0 = problem.t_axis[0];
    problem.t_last = problem.t_axis[sample_count - 1];
    problem.fused_sparse = true;
    problem.parallel_mode = parallel;
    problem.fused_static = static_view.data<cdouble>();
    problem.fused_controls = controls_view.data<cdouble>();
    problem.fused_indices = indices;
    problem.fused_indptr = indptr;
    problem.fused_nnz = nnz;
    problem.prepare_fused_control_entries();
    problem.detect_diagonal_fused();
    problem.angular_freqs.resize(static_cast<std::size_t>(control_count));
    for (int control = 0; control < control_count; ++control) {
        problem.angular_freqs[static_cast<std::size_t>(control)] =
            2.0 * kPi * problem.lo_freqs[control];
        problem.max_frequency = std::max(
            problem.max_frequency,
            std::abs(problem.lo_freqs[control]));
    }

    if (sample_count > 1) {
        problem.source_dt = problem.t_axis[1] - problem.t_axis[0];
        if (!(problem.source_dt > 0.0)) {
            PyErr_SetString(PyExc_ValueError, "t_axis must be strictly increasing");
            return nullptr;
        }
        for (int index = 2; index < sample_count; ++index) {
            const double delta = problem.t_axis[index] - problem.t_axis[index - 1];
            if (!(delta > 0.0)) {
                PyErr_SetString(PyExc_ValueError, "t_axis must be strictly increasing");
                return nullptr;
            }
        }
        problem.inv_source_dt = 1.0 / problem.source_dt;
        problem.prepare_slopes();
    } else {
        problem.source_dt = 1.0;
        problem.inv_source_dt = 1.0;
    }

    std::vector<cdouble> state(static_cast<std::size_t>(n) * batch_count);
    std::copy(
        initial_view.data<cdouble>(),
        initial_view.data<cdouble>() + state.size(),
        state.begin());
    problem.prepare_frequency_cache();
    Workspace workspace;
    workspace.resize(state.size(), 0);
    workspace.resize_scales(static_cast<std::size_t>(problem.control_count));
    workspace.resize_carrier_phases(problem.unique_angular_freqs.size());
    std::vector<cdouble> trajectory;
    std::string error_message;
    if (!run_integration(
            problem,
            state,
            workspace,
            store_trajectory != 0,
            trajectory,
            error_message)) {
        PyErr_SetString(PyExc_RuntimeError, error_message.c_str());
        return nullptr;
    }
    return make_native_result(problem, state, trajectory, store_trajectory != 0);
}

PyObject* native_propagate_lindblad_csr(
    PyObject*,
    PyObject* args,
    PyObject* kwargs) {
    PyObject* h0_data_object = nullptr;
    PyObject* h0_indices_object = nullptr;
    PyObject* h0_indptr_object = nullptr;
    PyObject* controls_data_object = nullptr;
    PyObject* controls_indices_object = nullptr;
    PyObject* controls_indptr_object = nullptr;
    PyObject* controls_offsets_object = nullptr;
    PyObject* collapse_data_object = nullptr;
    PyObject* collapse_indices_object = nullptr;
    PyObject* collapse_indptr_object = nullptr;
    PyObject* collapse_offsets_object = nullptr;
    PyObject* iq_object = nullptr;
    PyObject* t_axis_object = nullptr;
    PyObject* initial_object = nullptr;
    PyObject* lo_freqs_object = nullptr;
    int mode = 0;
    double atol = 1e-8;
    double rtol = 1e-6;
    long long max_steps = 2000000;
    int store_trajectory = 0;
    int sparse_expm = 1;
    PyObject* iq_polynomial_object = Py_None;
    int coefficient_order = 1;
    PyObject* control_modes_object = Py_None;
    int parallel = 1;
    static const char* keywords[] = {
        "h0_data", "h0_indices", "h0_indptr",
        "controls_data", "controls_indices", "controls_indptr", "controls_offsets",
        "collapse_data", "collapse_indices", "collapse_indptr", "collapse_offsets",
        "iq", "t_axis", "initial_states", "lo_freqs",
        "mode", "atol", "rtol", "max_steps", "store_trajectory", "sparse_expm",
        "iq_polynomial", "coefficient_order", "control_modes", "parallel", nullptr
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOOOOOOOOOOOOOO|iddLiiOiOi",
            const_cast<char**>(keywords),
            &h0_data_object,
            &h0_indices_object,
            &h0_indptr_object,
            &controls_data_object,
            &controls_indices_object,
            &controls_indptr_object,
            &controls_offsets_object,
            &collapse_data_object,
            &collapse_indices_object,
            &collapse_indptr_object,
            &collapse_offsets_object,
            &iq_object,
            &t_axis_object,
            &initial_object,
            &lo_freqs_object,
            &mode,
            &atol,
            &rtol,
            &max_steps,
            &store_trajectory,
            &sparse_expm,
            &iq_polynomial_object,
            &coefficient_order,
            &control_modes_object,
            &parallel)) {
        return nullptr;
    }
    if (mode != 0 && mode != 1 && mode != 2) {
        PyErr_SetString(PyExc_ValueError, "mode must be 0 (RF), 1 (complex envelope), or 2 (mixed)");
        return nullptr;
    }
    if (!std::isfinite(atol) || !std::isfinite(rtol)
        || !(atol > 0.0) || !(rtol > 0.0) || max_steps < 1) {
        PyErr_SetString(
            PyExc_ValueError,
            "atol and rtol must be finite and positive; max_steps must be positive");
        return nullptr;
    }
    if (sparse_expm < 0 || sparse_expm > 2) {
        PyErr_SetString(PyExc_ValueError, "sparse_expm must be 0 (off), 1 (auto), or 2 (on)");
        return nullptr;
    }
    if (coefficient_order < 0 || coefficient_order > 3) {
        PyErr_SetString(PyExc_ValueError, "coefficient_order must be between 0 and 3");
        return nullptr;
    }
    if (parallel < 0 || parallel > 2) {
        PyErr_SetString(PyExc_ValueError, "parallel must be 0 (off), 1 (auto), or 2 (on)");
        return nullptr;
    }

    BufferView h0_data_view, h0_indices_view, h0_indptr_view;
    BufferView controls_data_view, controls_indices_view;
    BufferView controls_indptr_view, controls_offsets_view;
    BufferView collapse_data_view, collapse_indices_view;
    BufferView collapse_indptr_view, collapse_offsets_view;
    BufferView iq_view, t_view, initial_view, lo_view;
    BufferView iq_polynomial_view, control_modes_view;
    const bool has_polynomial = iq_polynomial_object != Py_None;
    const bool has_control_modes = control_modes_object != Py_None;
    if (!h0_data_view.acquire(h0_data_object, "h0_data", 1, sizeof(cdouble))
        || !h0_indices_view.acquire(h0_indices_object, "h0_indices", 1, sizeof(std::int64_t))
        || !h0_indptr_view.acquire(h0_indptr_object, "h0_indptr", 1, sizeof(std::int64_t))
        || !controls_data_view.acquire(controls_data_object, "controls_data", 1, sizeof(cdouble))
        || !controls_indices_view.acquire(controls_indices_object, "controls_indices", 1, sizeof(std::int64_t))
        || !controls_indptr_view.acquire(controls_indptr_object, "controls_indptr", 2, sizeof(std::int64_t))
        || !controls_offsets_view.acquire(controls_offsets_object, "controls_offsets", 1, sizeof(std::int64_t))
        || !collapse_data_view.acquire(collapse_data_object, "collapse_data", 1, sizeof(cdouble))
        || !collapse_indices_view.acquire(collapse_indices_object, "collapse_indices", 1, sizeof(std::int64_t))
        || !collapse_indptr_view.acquire(collapse_indptr_object, "collapse_indptr", 2, sizeof(std::int64_t))
        || !collapse_offsets_view.acquire(collapse_offsets_object, "collapse_offsets", 1, sizeof(std::int64_t))
        || !iq_view.acquire(iq_object, "iq", 2, sizeof(cdouble))
        || !t_view.acquire(t_axis_object, "t_axis", 1, sizeof(double))
        || !initial_view.acquire(initial_object, "initial_states", 2, sizeof(cdouble))
        || !lo_view.acquire(lo_freqs_object, "lo_freqs", 1, sizeof(double))
        || (has_polynomial
            && !iq_polynomial_view.acquire(
                iq_polynomial_object,
                "iq_polynomial",
                3,
                sizeof(cdouble)))
        || (has_control_modes
            && !control_modes_view.acquire(
                control_modes_object,
                "control_modes",
                1,
                sizeof(std::int64_t)))) {
        return nullptr;
    }

    const int n = static_cast<int>(h0_indptr_view.view.shape[0]) - 1;
    const int control_count = static_cast<int>(controls_indptr_view.view.shape[0]);
    const int collapse_count = static_cast<int>(collapse_indptr_view.view.shape[0]);
    const int sample_count = static_cast<int>(iq_view.view.shape[1]);
    const int batch_count = static_cast<int>(initial_view.view.shape[1]);
    const std::int64_t h0_nnz = static_cast<std::int64_t>(h0_data_view.view.shape[0]);
    const std::int64_t controls_nnz = static_cast<std::int64_t>(controls_data_view.view.shape[0]);
    const std::int64_t collapse_nnz = static_cast<std::int64_t>(collapse_data_view.view.shape[0]);
    if (n < 1 || h0_indptr_view.view.shape[0] != static_cast<Py_ssize_t>(n + 1)
        || h0_indices_view.view.shape[0] != static_cast<Py_ssize_t>(h0_nnz)
        || controls_indptr_view.view.shape[1] != static_cast<Py_ssize_t>(n + 1)
        || controls_offsets_view.view.shape[0] != static_cast<Py_ssize_t>(control_count + 1)
        || collapse_count < 1
        || collapse_indptr_view.view.shape[1] != static_cast<Py_ssize_t>(n + 1)
        || collapse_offsets_view.view.shape[0] != static_cast<Py_ssize_t>(collapse_count + 1)
        || iq_view.view.shape[0] != static_cast<Py_ssize_t>(control_count)
        || t_view.view.shape[0] != static_cast<Py_ssize_t>(sample_count)
        || lo_view.view.shape[0] != static_cast<Py_ssize_t>(control_count)
        || initial_view.view.shape[0]
            != static_cast<Py_ssize_t>(static_cast<std::size_t>(n) * n)
        || sample_count < 1 || batch_count < 1) {
        PyErr_SetString(PyExc_ValueError, "Lindblad CSR array shapes are inconsistent");
        return nullptr;
    }
    std::int64_t validated_h0_nnz = 0;
    bool validated_h0_diagonal = false;
    bool validated_h0_diagonal_single = false;
    if (!validate_sparse_matrix(
            h0_data_view,
            h0_indices_view,
            h0_indptr_view,
            n,
            "h0",
            h0_nnz,
            &validated_h0_nnz,
            &validated_h0_diagonal,
            &validated_h0_diagonal_single)) {
        return nullptr;
    }
    if (!validate_csr_bundle(
            controls_data_view.data<cdouble>(),
            controls_indices_view.data<std::int64_t>(),
            controls_indptr_view.data<std::int64_t>(),
            controls_offsets_view.data<std::int64_t>(),
            control_count,
            n,
            controls_nnz,
            "controls")) {
        return nullptr;
    }
    if (!validate_csr_bundle(
            collapse_data_view.data<cdouble>(),
            collapse_indices_view.data<std::int64_t>(),
            collapse_indptr_view.data<std::int64_t>(),
            collapse_offsets_view.data<std::int64_t>(),
            collapse_count,
            n,
            collapse_nnz,
            "collapse")) {
        return nullptr;
    }
    if (has_polynomial) {
        if (!validate_polynomial_view(
                iq_polynomial_view,
                control_count,
                std::max(0, sample_count - 1),
                coefficient_order)) {
            return nullptr;
        }
    } else if (coefficient_order != 1 && control_count > 0) {
        PyErr_SetString(
            PyExc_ValueError,
            "iq_polynomial is required when coefficient_order is not 1");
        return nullptr;
    }
    if (has_control_modes) {
        if (!validate_control_modes_view(control_modes_view, control_count)) {
            return nullptr;
        }
    } else if (mode == 2 && control_count > 0) {
        PyErr_SetString(PyExc_ValueError, "control_modes is required for mixed mode");
        return nullptr;
    }

    const std::size_t iq_size = static_cast<std::size_t>(control_count) * sample_count;
    const std::size_t initial_size = static_cast<std::size_t>(n) * n * batch_count;
    for (std::size_t index = 0; index < iq_size; ++index) {
        if (!finite_value(iq_view.data<cdouble>()[index])) {
            PyErr_SetString(PyExc_ValueError, "iq must contain only finite values");
            return nullptr;
        }
    }
    for (std::size_t index = 0; index < initial_size; ++index) {
        if (!finite_value(initial_view.data<cdouble>()[index])) {
            PyErr_SetString(PyExc_ValueError, "initial_states must contain only finite values");
            return nullptr;
        }
    }
    for (int index = 0; index < sample_count; ++index) {
        if (!std::isfinite(t_view.data<double>()[index])) {
            PyErr_SetString(PyExc_ValueError, "t_axis must contain only finite values");
            return nullptr;
        }
    }
    for (int index = 0; index < control_count; ++index) {
        if (!std::isfinite(lo_view.data<double>()[index])) {
            PyErr_SetString(PyExc_ValueError, "lo_freqs must contain only finite values");
            return nullptr;
        }
    }

    Problem problem;
    problem.n = n;
    problem.control_count = control_count;
    problem.sample_count = sample_count;
    problem.batch_count = batch_count;
    problem.iq = iq_view.data<cdouble>();
    problem.t_axis = t_view.data<double>();
    problem.lo_freqs = lo_view.data<double>();
    problem.mode = mode;
    problem.atol = atol;
    problem.rtol = rtol;
    problem.max_steps = static_cast<std::int64_t>(max_steps);
    problem.sparse_expm_mode = sparse_expm;
    problem.parallel_mode = parallel;
    problem.coefficient_order = coefficient_order;
    problem.polynomial_interval_count = std::max(0, sample_count - 1);
    problem.iq_polynomial = has_polynomial
        ? iq_polynomial_view.data<cdouble>()
        : nullptr;
    problem.control_modes = has_control_modes
        ? control_modes_view.data<std::int64_t>()
        : nullptr;
    problem.t0 = problem.t_axis[0];
    problem.t_last = problem.t_axis[sample_count - 1];
    problem.lindblad = true;
    problem.lindblad_collapse_count = collapse_count;
    problem.sparse = true;
    problem.h0_sparse = {
        h0_data_view.data<cdouble>(),
        h0_indices_view.data<std::int64_t>(),
        h0_indptr_view.data<std::int64_t>(),
        h0_nnz,
        validated_h0_diagonal,
        validated_h0_diagonal_single,
    };
    const auto* control_offsets = controls_offsets_view.data<std::int64_t>();
    const auto* control_indptr = controls_indptr_view.data<std::int64_t>();
    const auto* controls_data_base = controls_data_view.data<cdouble>();
    const auto* controls_indices_base = controls_indices_view.data<std::int64_t>();
    problem.controls_sparse.reserve(static_cast<std::size_t>(control_count));
    for (int control = 0; control < control_count; ++control) {
        const std::int64_t offset = control_offsets[control];
        const std::int64_t count = control_offsets[control + 1] - offset;
        problem.controls_sparse.push_back({
            controls_data_base == nullptr ? nullptr : controls_data_base + offset,
            controls_indices_base == nullptr ? nullptr : controls_indices_base + offset,
            control_indptr + static_cast<std::size_t>(control) * (n + 1),
            count,
            false,
            false,
        });
        problem.angular_freqs.push_back(2.0 * kPi * problem.lo_freqs[control]);
        problem.max_frequency = std::max(
            problem.max_frequency,
            std::abs(problem.lo_freqs[control]));
    }
    const auto* collapse_offsets = collapse_offsets_view.data<std::int64_t>();
    const auto* collapse_indptr = collapse_indptr_view.data<std::int64_t>();
    const auto* collapse_data = collapse_data_view.data<cdouble>();
    const auto* collapse_indices = collapse_indices_view.data<std::int64_t>();
    problem.lindblad_collapse.reserve(static_cast<std::size_t>(collapse_count));
    problem.lindblad_products.reserve(static_cast<std::size_t>(collapse_count));
    for (int collapse_index = 0; collapse_index < collapse_count; ++collapse_index) {
        const std::int64_t offset = collapse_offsets[collapse_index];
        const std::int64_t count = collapse_offsets[collapse_index + 1] - offset;
        OwnedSparseMatrix collapse = copy_owned_sparse(
            collapse_data == nullptr ? nullptr : collapse_data + offset,
            collapse_indices == nullptr ? nullptr : collapse_indices + offset,
            collapse_indptr + static_cast<std::size_t>(collapse_index) * (n + 1),
            n,
            count);
        OwnedSparseMatrix dagger = conjugate_transpose_sparse(collapse, n);
        OwnedSparseMatrix product = sparse_matrix_product(dagger, collapse, n);
        problem.lindblad_collapse.push_back(std::move(collapse));
        problem.lindblad_products.push_back(std::move(product));
    }
    problem.lindblad_jump_patterns.reserve(static_cast<std::size_t>(collapse_count));
    constexpr std::size_t kJumpPatternBudgetBytes = 64u * 1024u * 1024u;
    std::size_t jump_pattern_budget = kJumpPatternBudgetBytes;
    for (const OwnedSparseMatrix& collapse : problem.lindblad_collapse) {
        const std::size_t entry_budget = jump_pattern_budget
            / std::max<std::size_t>(sizeof(LindbladJumpEntry), 1u);
        LindbladJumpPattern pattern = build_lindblad_jump_pattern(
            collapse, n, std::min<std::size_t>(entry_budget, 1u * 1024u * 1024u));
        if (pattern.ready) {
            const std::size_t bytes = pattern.entries.size()
                * sizeof(LindbladJumpEntry)
                + pattern.row_ptr.size() * sizeof(std::int64_t);
            if (bytes > jump_pattern_budget) {
                pattern = LindbladJumpPattern();
            } else {
                jump_pattern_budget -= bytes;
            }
        }
        problem.lindblad_jump_patterns.push_back(std::move(pattern));
    }
    problem.detect_diagonal_sparse();
    problem.detect_lindblad_diagonal();
    if (sample_count > 1) {
        problem.source_dt = problem.t_axis[1] - problem.t_axis[0];
        if (!(problem.source_dt > 0.0)) {
            PyErr_SetString(PyExc_ValueError, "t_axis must be strictly increasing");
            return nullptr;
        }
        for (int index = 2; index < sample_count; ++index) {
            const double delta = problem.t_axis[index] - problem.t_axis[index - 1];
            if (!(delta > 0.0)) {
                PyErr_SetString(PyExc_ValueError, "t_axis must be strictly increasing");
                return nullptr;
            }
        }
        problem.inv_source_dt = 1.0 / problem.source_dt;
        problem.prepare_slopes();
    } else {
        problem.source_dt = 1.0;
        problem.inv_source_dt = 1.0;
    }
    problem.prepare_frequency_cache();
    std::vector<cdouble> state(initial_size);
    std::copy(
        initial_view.data<cdouble>(),
        initial_view.data<cdouble>() + state.size(),
        state.begin());
    Workspace workspace;
    workspace.resize(state.size(), 0);
    workspace.resize_scales(static_cast<std::size_t>(problem.control_count));
    workspace.resize_carrier_phases(problem.unique_angular_freqs.size());
    std::vector<cdouble> trajectory;
    std::string error_message;
    if (!run_integration(
            problem,
            state,
            workspace,
            store_trajectory != 0,
            trajectory,
            error_message)) {
        PyErr_SetString(PyExc_RuntimeError, error_message.c_str());
        return nullptr;
    }
    return make_native_result(problem, state, trajectory, store_trajectory != 0);
}

PyMethodDef module_methods[] = {
    {
        "propagate",
        reinterpret_cast<PyCFunction>(native_propagate),
        METH_VARARGS | METH_KEYWORDS,
        "Propagate one or more coherent ket states with a dense time-dependent Hamiltonian.",
    },
    {
        "propagate_csr",
        reinterpret_cast<PyCFunction>(native_propagate_csr),
        METH_VARARGS | METH_KEYWORDS,
        "Propagate coherent ket states using CSR Hamiltonian and drive operators.",
    },
    {
        "propagate_banded",
        reinterpret_cast<PyCFunction>(native_propagate_banded),
        METH_VARARGS | METH_KEYWORDS,
        "Propagate coherent ket states using exact banded Hamiltonian operators.",
    },
    {
        "propagate_fused_csr",
        reinterpret_cast<PyCFunction>(native_propagate_fused_csr),
        METH_VARARGS | METH_KEYWORDS,
        "Propagate coherent ket states using a fused union CSR pattern.",
    },
    {
        "propagate_interaction_csr",
        reinterpret_cast<PyCFunction>(native_propagate_interaction_csr),
        METH_VARARGS | METH_KEYWORDS,
        "Propagate coherent ket states in an exact diagonal interaction picture.",
    },
    {
        "propagate_lindblad_csr",
        reinterpret_cast<PyCFunction>(native_propagate_lindblad_csr),
        METH_VARARGS | METH_KEYWORDS,
        "Propagate density matrices with a matrix-free exact Lindblad CSR kernel.",
    },
    {nullptr, nullptr, 0, nullptr},
};

PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT,
    "_dynamics",
    "Optional native dynamics kernel for pysuqu.",
    -1,
    module_methods,
};

}  // namespace

PyMODINIT_FUNC PyInit__dynamics() {
    return PyModule_Create(&module_definition);
}
