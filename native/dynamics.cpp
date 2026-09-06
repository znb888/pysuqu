#define PY_SSIZE_T_CLEAN

#include <Python.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <limits>
#include <string>
#include <vector>

namespace {

using cdouble = std::complex<double>;
constexpr double kPi = 3.141592653589793238462643383279502884;

static_assert(sizeof(cdouble) == sizeof(double) * 2, "complex128 ABI mismatch");

bool finite_value(cdouble value) {
    return std::isfinite(value.real()) && std::isfinite(value.imag());
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
};

struct Problem {
    const cdouble* h0{nullptr};
    const cdouble* controls{nullptr};
    const cdouble* iq{nullptr};
    const double* t_axis{nullptr};
    const double* lo_freqs{nullptr};
    int n{0};
    int control_count{0};
    int sample_count{0};
    int batch_count{0};
    int mode{0};
    double atol{1e-8};
    double rtol{1e-6};
    std::int64_t max_steps{2000000};
    std::int64_t attempted_steps{0};
    std::int64_t rhs_evaluations{0};
    double max_error{0.0};
    double max_trial_error{0.0};
    double source_dt{0.0};
    double inv_source_dt{0.0};
    double t0{0.0};
    double t_last{0.0};
    double max_frequency{0.0};
    std::vector<double> angular_freqs;
    bool sparse{false};
    SparseMatrixView h0_sparse;
    std::vector<SparseMatrixView> controls_sparse;

    cdouble coefficient(int control, double t) const {
        if (t < t0 || t > t_last) {
            return cdouble(0.0, 0.0);
        }
        if (sample_count == 1) {
            const cdouble value = iq[control * sample_count];
            if (mode == 1) {
                return value;
            }
            const double angular_frequency = angular_freqs.empty()
                ? 2.0 * kPi * lo_freqs[control]
                : angular_freqs[static_cast<std::size_t>(control)];
            if (angular_frequency == 0.0) {
                return cdouble(value.real(), 0.0);
            }
            const double phase = angular_frequency * t;
            return value.real() * std::cos(phase) - value.imag() * std::sin(phase);
        }

        double position = (t - t0) * inv_source_dt;
        if (position < 0.0) {
            position = 0.0;
        }
        const int left = std::min(sample_count - 1, std::max(0, static_cast<int>(std::floor(position))));
        const int right = std::min(sample_count - 1, left + 1);
        const double fraction = (right == left) ? 0.0 : position - static_cast<double>(left);
        const cdouble left_value = iq[control * sample_count + left];
        const cdouble right_value = iq[control * sample_count + right];
        const cdouble value = left_value + fraction * (right_value - left_value);
        if (mode == 1) {
            return value;
        }
        const double angular_frequency = angular_freqs.empty()
            ? 2.0 * kPi * lo_freqs[control]
            : angular_freqs[static_cast<std::size_t>(control)];
        if (angular_frequency == 0.0) {
            return cdouble(value.real(), 0.0);
        }
        const double phase = angular_frequency * t;
        return value.real() * std::cos(phase) - value.imag() * std::sin(phase);
    }

    void build_hamiltonian(double t, Workspace& workspace) const {
        std::copy(h0, h0 + static_cast<std::size_t>(n) * n, workspace.hmat.begin());
        for (int control = 0; control < control_count; ++control) {
            const cdouble scale = coefficient(control, t);
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

    void rhs(
        double t,
        const std::vector<cdouble>& state,
        std::vector<cdouble>& derivative,
        Workspace& workspace) {
        ++rhs_evaluations;
        if (sparse) {
            std::fill(derivative.begin(), derivative.end(), cdouble(0.0, 0.0));
            add_sparse_matrix(h0_sparse, cdouble(1.0, 0.0), state, derivative);
            for (int control = 0; control < control_count; ++control) {
                add_sparse_matrix(
                    controls_sparse[static_cast<std::size_t>(control)],
                    coefficient(control, t),
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
            for (int control = 0; control < control_count; ++control) {
                workspace.scales[static_cast<std::size_t>(control)] = coefficient(control, t);
            }
            for (int row = 0; row < n; ++row) {
                cdouble value(0.0, 0.0);
                for (int column = 0; column < n; ++column) {
                    const std::size_t matrix_index = static_cast<std::size_t>(row) * n + column;
                    cdouble matrix_value = h0[matrix_index];
                    for (int control = 0; control < control_count; ++control) {
                        matrix_value += workspace.scales[static_cast<std::size_t>(control)]
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
    if (problem.mode == 0 && problem.max_frequency > 0.0) {
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

        for (std::size_t index = 0; index < state_size; ++index) {
            trial[index] = state[index] + step * (
                35.0 / 384.0 * k1[index]
                + 500.0 / 1113.0 * k3[index]
                + 125.0 / 192.0 * k4[index]
                - 2187.0 / 6784.0 * k5[index]
                + 11.0 / 84.0 * k6[index]);
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

bool run_integration(
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

    bool success = true;
    // The whole numerical loop runs without the Python GIL.  No Python object
    // or callback is touched until all intervals have been integrated.
    Py_BEGIN_ALLOW_THREADS
    try {
        for (int sample = 0; sample + 1 < problem.sample_count; ++sample) {
            if (!integrate_interval(
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
    value = PyUnicode_FromString(problem.sparse ? "cpp_dopri5_csr" : "cpp_dopri5");
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
    static const char* keywords[] = {
        "h0", "controls", "iq", "t_axis", "initial_states", "lo_freqs",
        "mode", "atol", "rtol", "max_steps", "store_trajectory", nullptr
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOOOOO|iddLi",
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
            &store_trajectory)) {
        return nullptr;
    }
    if (mode != 0 && mode != 1) {
        PyErr_SetString(PyExc_ValueError, "mode must be 0 (RF) or 1 (complex envelope)");
        return nullptr;
    }
    if (!(atol > 0.0) || !(rtol > 0.0) || max_steps < 1) {
        PyErr_SetString(PyExc_ValueError, "atol, rtol, and max_steps must be positive");
        return nullptr;
    }

    BufferView h0_view, controls_view, iq_view, t_view, initial_view, lo_view;
    if (!h0_view.acquire(h0_object, "h0", 2, sizeof(cdouble))
        || !controls_view.acquire(controls_object, "controls", 3, sizeof(cdouble))
        || !iq_view.acquire(iq_object, "iq", 2, sizeof(cdouble))
        || !t_view.acquire(t_axis_object, "t_axis", 1, sizeof(double))
        || !initial_view.acquire(initial_object, "initial_states", 2, sizeof(cdouble))
        || !lo_view.acquire(lo_freqs_object, "lo_freqs", 1, sizeof(double))) {
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
    problem.t0 = problem.t_axis[0];
    problem.t_last = problem.t_axis[problem.sample_count - 1];
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
            if (!(delta > 0.0)
                || std::abs(delta - problem.source_dt)
                    > 1e-9 * std::max(1.0, std::abs(problem.source_dt))) {
                PyErr_SetString(PyExc_ValueError, "native propagation requires a regular t_axis");
                return nullptr;
            }
        }
        problem.inv_source_dt = 1.0 / problem.source_dt;
    } else {
        problem.source_dt = 1.0;
        problem.inv_source_dt = 1.0;
    }

    Workspace workspace;
    workspace.resize(
        static_cast<std::size_t>(problem.n) * problem.batch_count,
        problem.batch_count == 1
            ? 0
            : static_cast<std::size_t>(problem.n) * problem.n);
    workspace.resize_scales(static_cast<std::size_t>(problem.control_count));

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
    static const char* keywords[] = {
        "h0_data", "h0_indices", "h0_indptr",
        "controls_data", "controls_indices", "controls_indptr", "controls_offsets",
        "iq", "t_axis", "initial_states", "lo_freqs",
        "mode", "atol", "rtol", "max_steps", "store_trajectory", nullptr
    };
    if (!PyArg_ParseTupleAndKeywords(
            args,
            kwargs,
            "OOOOOOOOOOO|iddLi",
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
            &store_trajectory)) {
        return nullptr;
    }
    if (mode != 0 && mode != 1) {
        PyErr_SetString(PyExc_ValueError, "mode must be 0 (RF) or 1 (complex envelope)");
        return nullptr;
    }
    if (!(atol > 0.0) || !(rtol > 0.0) || max_steps < 1) {
        PyErr_SetString(PyExc_ValueError, "atol, rtol, and max_steps must be positive");
        return nullptr;
    }

    BufferView h0_data_view, h0_indices_view, h0_indptr_view;
    BufferView controls_data_view, controls_indices_view, controls_indptr_view, controls_offsets_view;
    BufferView iq_view, t_view, initial_view, lo_view;
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
        || !lo_view.acquire(lo_freqs_object, "lo_freqs", 1, sizeof(double))) {
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
    problem.t0 = problem.t_axis[0];
    problem.t_last = problem.t_axis[sample_count - 1];
    problem.sparse = true;
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
    if (sample_count > 1) {
        problem.source_dt = problem.t_axis[1] - problem.t_axis[0];
        if (!(problem.source_dt > 0.0)) {
            PyErr_SetString(PyExc_ValueError, "t_axis must be strictly increasing");
            return nullptr;
        }
        for (int index = 2; index < sample_count; ++index) {
            const double delta = problem.t_axis[index] - problem.t_axis[index - 1];
            if (!(delta > 0.0)
                || std::abs(delta - problem.source_dt)
                    > 1e-9 * std::max(1.0, std::abs(problem.source_dt))) {
                PyErr_SetString(PyExc_ValueError, "native propagation requires a regular t_axis");
                return nullptr;
            }
        }
        problem.inv_source_dt = 1.0 / problem.source_dt;
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
    Workspace workspace;
    workspace.resize(state.size(), 0);
    workspace.resize_scales(static_cast<std::size_t>(problem.control_count));
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
