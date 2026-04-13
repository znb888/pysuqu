import sys
import types

import numpy as np


def install_qutip_stub():
    if 'qutip' in sys.modules:
        return

    qutip = types.ModuleType('qutip')

    class Qobj:
        def __init__(self, data=None, dims=None):
            array = np.array(data, dtype=complex) if data is not None else np.zeros((1, 1), dtype=complex)
            if array.ndim == 0:
                array = array.reshape((1, 1))
            self._data = array
            self.dims = dims or self._infer_dims(array)
            self._refresh_type_flags()

        @staticmethod
        def _infer_dims(array):
            if array.ndim != 2:
                raise ValueError('Qobj stub expects 2D data.')

            rows, cols = array.shape
            if cols == 1 and rows != 1:
                return [[rows], [1]]
            if rows == 1 and cols != 1:
                return [[1], [cols]]
            return [[rows], [cols]]

        def _refresh_type_flags(self):
            left_dims, right_dims = self.dims
            self.isket = all(dim == 1 for dim in right_dims) and not all(dim == 1 for dim in left_dims)
            self.isbra = all(dim == 1 for dim in left_dims) and not all(dim == 1 for dim in right_dims)
            self.isoper = not self.isket and not self.isbra
            if self.isket:
                self.type = 'ket'
            elif self.isbra:
                self.type = 'bra'
            else:
                self.type = 'oper'

        @property
        def shape(self):
            return self._data.shape

        def __getitem__(self, key):
            return self._data[key]

        def full(self):
            return np.array(self._data, copy=True)

        def dag(self):
            return Qobj(self._data.conjugate().T, dims=[list(self.dims[1]), list(self.dims[0])])

        def unit(self):
            norm = np.linalg.norm(self._data)
            if norm == 0:
                return Qobj(self._data, dims=[list(self.dims[0]), list(self.dims[1])])
            return Qobj(self._data / norm, dims=[list(self.dims[0]), list(self.dims[1])])

        def overlap(self, other):
            return np.vdot(self._data.reshape(-1), other.full().reshape(-1))

        def eigenstates(self):
            values, vectors = np.linalg.eigh(self._data)
            ket_dims = [list(self.dims[0]), [1] * len(self.dims[0])]
            states = [Qobj(vectors[:, idx].reshape((-1, 1)), dims=ket_dims) for idx in range(vectors.shape[1])]
            return values.real, states

        def __add__(self, other):
            if other == 0:
                return self
            if isinstance(other, Qobj):
                return Qobj(self._data + other._data, dims=[list(self.dims[0]), list(self.dims[1])])
            return Qobj(self._data + other, dims=[list(self.dims[0]), list(self.dims[1])])

        def __radd__(self, other):
            return self.__add__(other)

        def __sub__(self, other):
            if isinstance(other, Qobj):
                return Qobj(self._data - other._data, dims=[list(self.dims[0]), list(self.dims[1])])
            return Qobj(self._data - other, dims=[list(self.dims[0]), list(self.dims[1])])

        def __rsub__(self, other):
            if other == 0:
                return -self
            if isinstance(other, Qobj):
                return Qobj(other._data - self._data, dims=[list(self.dims[0]), list(self.dims[1])])
            return Qobj(other - self._data, dims=[list(self.dims[0]), list(self.dims[1])])

        def __mul__(self, other):
            if isinstance(other, Qobj):
                dims = [list(self.dims[0]), list(other.dims[1])]
                return Qobj(self._data @ other._data, dims=dims)
            return Qobj(self._data * other, dims=[list(self.dims[0]), list(self.dims[1])])

        def __rmul__(self, other):
            if isinstance(other, Qobj):
                dims = [list(other.dims[0]), list(self.dims[1])]
                return Qobj(other._data @ self._data, dims=dims)
            return Qobj(other * self._data, dims=[list(self.dims[0]), list(self.dims[1])])

        def __truediv__(self, other):
            return Qobj(self._data / other, dims=[list(self.dims[0]), list(self.dims[1])])

        def __pow__(self, power):
            return Qobj(
                np.linalg.matrix_power(self._data, power),
                dims=[list(self.dims[0]), list(self.dims[1])],
            )

        def __neg__(self):
            return Qobj(-self._data, dims=[list(self.dims[0]), list(self.dims[1])])

        def __abs__(self):
            if self._data.size == 1:
                return abs(self._data.item())
            return float(np.linalg.norm(self._data))

        def __repr__(self):
            return f'Qobj(shape={self._data.shape}, dims={self.dims}, type={self.type!r})'

    class Result:
        def __init__(self, states=None, times=None, expect=None):
            self.states = [] if states is None else states
            self.times = [] if times is None else times
            self.expect = [] if expect is None else expect

    class Bloch:
        def __init__(self):
            self.view = None
            self.point_marker = []
            self.point_size = []
            self.points = []
            self.shown = False
            qutip._last_bloch = self

        @staticmethod
        def _normalize_points(points):
            if isinstance(points, np.ndarray):
                return Bloch._normalize_points(points.tolist())
            if isinstance(points, (list, tuple)):
                return [Bloch._normalize_points(value) for value in points]

            value = complex(points)
            if abs(value.imag) < 1e-12:
                return float(value.real)
            return value

        def add_points(self, points, meth=None):
            self.points.append({
                'points': self._normalize_points(points),
                'meth': meth,
            })
            return self

        def show(self):
            self.shown = True
            return None

    qutip.Qobj = Qobj
    qutip.Result = Result
    qutip.Bloch = Bloch

    def _normalize_tensor_args(args):
        if len(args) == 1 and isinstance(args[0], (list, tuple)):
            return list(args[0])
        return list(args)

    def tensor(*args):
        operators = _normalize_tensor_args(args)
        if not operators:
            return Qobj()

        data = operators[0].full()
        left_dims = list(operators[0].dims[0])
        right_dims = list(operators[0].dims[1])
        for operator in operators[1:]:
            data = np.kron(data, operator.full())
            left_dims.extend(operator.dims[0])
            right_dims.extend(operator.dims[1])
        return Qobj(data, dims=[left_dims, right_dims])

    def basis(dim, index):
        data = np.zeros((dim, 1), dtype=complex)
        data[index, 0] = 1.0
        return Qobj(data, dims=[[dim], [1]])

    def qeye(dim):
        return Qobj(np.eye(dim, dtype=complex), dims=[[dim], [dim]])

    def qdiags(diagonals, offsets):
        if isinstance(offsets, (list, tuple, np.ndarray)):
            if len(offsets) != 1:
                raise NotImplementedError('Qobj stub only supports a single diagonal.')
            offset = int(offsets[0])
        else:
            offset = int(offsets)

        diagonal = np.asarray(diagonals, dtype=complex)
        size = len(diagonal) + abs(offset)
        data = np.zeros((size, size), dtype=complex)
        row_indices, col_indices = np.diag_indices_from(data)
        if offset >= 0:
            row_indices = row_indices[: len(diagonal)]
            col_indices = col_indices[offset : offset + len(diagonal)]
        else:
            row_indices = row_indices[-offset : -offset + len(diagonal)]
            col_indices = col_indices[: len(diagonal)]
        data[row_indices, col_indices] = diagonal
        return Qobj(data, dims=[[size], [size]])

    def destroy(dim):
        data = np.zeros((dim, dim), dtype=complex)
        for level in range(1, dim):
            data[level - 1, level] = np.sqrt(level)
        return Qobj(data, dims=[[dim], [dim]])

    def sigmax():
        return Qobj(np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex), dims=[[2], [2]])

    def sigmay():
        return Qobj(np.array([[0.0, -1j], [1j, 0.0]], dtype=complex), dims=[[2], [2]])

    def sigmaz():
        return Qobj(np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex), dims=[[2], [2]])

    def expect(operator, state):
        if state.isket:
            return (state.dag() * operator * state)._data.item()
        if state.isbra:
            return (state * operator * state.dag())._data.item()
        return np.trace(operator.full() @ state.full())

    def ket2dm(obj):
        if obj.isket:
            return obj * obj.dag()
        if obj.isbra:
            return obj.dag() * obj
        return obj

    def _matrix_sqrt(data):
        eigenvalues, eigenvectors = np.linalg.eigh(data)
        clipped = np.clip(eigenvalues.real, a_min=0.0, a_max=None)
        return eigenvectors @ np.diag(np.sqrt(clipped)) @ eigenvectors.conjugate().T

    def fidelity(left, right):
        left_dm = ket2dm(left).full()
        right_dm = ket2dm(right).full()
        if np.allclose(left_dm, right_dm):
            return 1.0
        left_sqrt = _matrix_sqrt(left_dm)
        overlap = left_sqrt @ right_dm @ left_sqrt
        eigenvalues = np.linalg.eigvalsh(overlap)
        return float(np.sum(np.sqrt(np.clip(eigenvalues.real, a_min=0.0, a_max=None))))

    def mesolve(H, rho0, tlist, c_ops=None, e_ops=None, options=None, args=None):
        times = list(np.asarray(tlist, dtype=float))
        result = Result(
            states=[rho0 for _ in times],
            times=times,
            expect=[] if e_ops in (None, []) else [0 for _ in e_ops],
        )
        qutip._last_mesolve_call = {
            'H': H,
            'rho0': rho0,
            'tlist': times,
            'c_ops': list(c_ops or []),
            'e_ops': list(e_ops or []),
            'options': dict(options or {}),
            'args': dict(args or {}),
        }
        return result

    qutip.tensor = tensor
    qutip.basis = basis
    qutip.qeye = qeye
    qutip.qdiags = qdiags
    qutip.destroy = destroy
    qutip.sigmax = sigmax
    qutip.sigmay = sigmay
    qutip.sigmaz = sigmaz
    qutip.expect = expect
    qutip.ket2dm = ket2dm
    qutip.fidelity = fidelity
    qutip.mesolve = mesolve
    qutip._last_mesolve_call = None
    qutip._last_bloch = None
    sys.modules['qutip'] = qutip


def install_plotly_stub():
    if 'plotly' in sys.modules:
        return

    plotly = types.ModuleType('plotly')
    graph_objects = types.ModuleType('plotly.graph_objects')
    subplots = types.ModuleType('plotly.subplots')

    class Figure:
        def __init__(self):
            self.traces = []
            self.vlines = []
            self.layout_updates = []
            self.yaxis_updates = []

        def add_trace(self, trace, secondary_y=None):
            self.traces.append((trace, secondary_y))
            return self

        def add_vline(self, *args, **kwargs):
            self.vlines.append((args, kwargs))
            return self

        def update_layout(self, *args, **kwargs):
            self.layout_updates.append((args, kwargs))
            return self

        def update_yaxes(self, *args, **kwargs):
            self.yaxis_updates.append((args, kwargs))
            return self

        def show(self):
            return None

    class Scatter:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def make_subplots(*args, **kwargs):
        return Figure()

    graph_objects.Figure = Figure
    graph_objects.Scatter = Scatter
    plotly.graph_objects = graph_objects
    subplots.make_subplots = make_subplots
    sys.modules['plotly'] = plotly
    sys.modules['plotly.graph_objects'] = graph_objects
    sys.modules['plotly.subplots'] = subplots


def install_test_stubs():
    install_qutip_stub()
    install_plotly_stub()
