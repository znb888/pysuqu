'''
Lib for qutip utility functions.

 Author: Zhou Naibin
 USTC
 Since 2025-04-15
'''

from functools import lru_cache

import numpy as np
import qutip as qt
from qutip import Qobj, basis, tensor

def truncate_precision(qobject: Qobj, threshold: float = 1e-15) -> Qobj:
    """
    Truncates components of a quantum object with absolute values below a threshold to zero.
    
    If the input is a quantum state (Ket or Bra), it is automatically re-normalized 
    to ensure the norm remains 1 after truncation. Operators are not normalized to 
    preserve physical quantities (e.g., energy).

    Args:
        qobject (Qobj): Input quantum object (Qutip object).
        threshold (float): Cutoff threshold. Component magnitudes below this 
                           value will be set to 0. Default is 1e-15.

    Returns:
        Qobj: Processed quantum object.
    """
    data = qobject.full()

    data[np.abs(data) < threshold] = 0
    new_qobj = Qobj(data, dims=qobject.dims)

    if new_qobj.isket or new_qobj.isbra:
        return new_qobj.unit()
    else:
        return new_qobj


@lru_cache(maxsize=256)
def _build_truncation_projector_pair(
    old_dims_key: tuple[int, ...],
    new_dims_key: tuple[int, ...],
) -> tuple[Qobj, Qobj]:
    """Build the reusable tensor projector pair for one exact Hilbert-space truncation."""
    projectors = []
    for old_d, new_d in zip(old_dims_key, new_dims_key):
        proj_mat = np.eye(old_d)[:new_d, :]
        projectors.append(Qobj(proj_mat))

    projector = tensor(projectors)
    return projector, projector.dag()

def truncate_hilbert_space(qobj: Qobj, new_dims_list: list) -> Qobj:
    """
    Truncates the Hilbert space of a composite quantum system to specified lower dimensions
    using projection operators. Works for both product states and coupled/entangled systems.

    Args:
        qobj (Qobj): The input quantum object (Operator, Ket, or Bra).
        new_dims_list (list): A list of integers specifying the new dimension for each subsystem 
                              (e.g., [5, 4, 5]).

    Returns:
        Qobj: The truncated quantum object with updated dimensions.
    """
    old_dims_key = tuple(int(dim) for dim in qobj.dims[0])
    new_dims_key = tuple(int(dim) for dim in new_dims_list)

    if len(old_dims_key) != len(new_dims_key):
        raise ValueError(
            f"Dimension mismatch: input has {len(old_dims_key)} subsystems, "
            f"but target dims provide {len(new_dims_key)}."
        )

    if old_dims_key == new_dims_key:
        return qobj

    for old_d, new_d in zip(old_dims_key, new_dims_key):
        if new_d > old_d:
            raise ValueError(f"New dimension {new_d} cannot be larger than old dimension {old_d}.")

    projector, projector_dag = _build_truncation_projector_pair(old_dims_key, new_dims_key)

    if qobj.isoper:
        new_qobj = projector * qobj * projector_dag
    elif qobj.isket:
        new_qobj = projector * qobj
    elif qobj.isbra:
        new_qobj = qobj * projector_dag
    else:
        raise TypeError("Unsupported Qobj type. Must be Operator, Ket, or Bra.")

    return new_qobj

def cal_product_state(state: list[int], Nlevel: list[int]) -> Qobj:
    """
    Calculates the product state of multiple qubits.

    Args:
        state_list (list): A list of Qobj state index for each qubit.

    Returns:
        Qobj: The product state tensor of all input states.
    """
    product_state = basis(Nlevel[0],state[0])
    
    for i in range(1, len(state)):
        product_state = tensor(product_state, basis(Nlevel[i], state[i]))
    
    return product_state

def cal_product_state_list(state_list: list[list[int]], Nlevel: list[int]) -> list[Qobj]:
    """
    Calculates the product state of multiple qubits for each state list in the input.

    Args:
        state_list_list (list): A list of lists, where each inner list contains Qobj state index for each qubit.

    Returns:
        list: A list of Qobj product states for each input state list.
    """
    product_state_list = []
    for state in state_list:
        product_state_list.append(cal_product_state(state, Nlevel))
    return product_state_list

def gate_fidelity_bystate(final_state: Qobj, target_state: Qobj) -> float:
    """
    Calculates the gate fidelity between the final state and the target state.

    Args:
        final_state (Qobj): The final state after applying the gate.
        target_state (Qobj): The target state to compare against.

    Returns:
        float: The gate fidelity between the final state and the target state.
    """
    if final_state.isket and target_state.isket:
        fidelity = np.abs(final_state.overlap(target_state))**2
    else:
        fidelity = qt.fidelity(final_state, target_state)**2
        
    return float(fidelity)

def gate_fidelity_bymat(U_actual: Qobj, U_ideal: Qobj) -> float:
    """
    Calculates the Average Gate Fidelity between an actual process and an ideal unitary.
    
    This metric corresponds to the value measured by Randomized Benchmarking (RB).
    
    Formula:
        F_avg = (d * F_pro + 1) / (d + 1)
        where F_pro is the process fidelity (overlap of Choi matrices) and d is dimension.

    Args:
        U_actual (Qobj): The implemented gate. Can be:
                         1. A Unitary operator (dims [[N,N],[N,N]], type='oper')
                         2. A Superoperator/Channel (dims [[[N,N],[N,N]]], type='super')
        U_ideal (Qobj):  The target ideal gate. Must be a Unitary operator (type='oper').

    Returns:
        float: The Average Gate Fidelity (0 to 1).
    """
    if U_ideal.type != 'oper':
        raise ValueError("Target U_ideal must be a unitary operator (type='oper').")

    fid_avg = qt.average_gate_fidelity(U_actual, target=U_ideal)
    
    return float(fid_avg)


