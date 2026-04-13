'''
Lib for multiqubit simulation.

Author: Naibin Zhou
USTC
Since 2023-12-05
'''
# import
import numpy as np
import matplotlib.pyplot as plt
from copy import copy
# local lib
from ..funclib import *
from ..funclib.qutiplib import cal_product_state_list
from .analysis import (
    analyze_multi_qubit_coupler_sensitivity,
    calculate_multi_qubit_coupler_self_sensitivity,
    calculate_multi_qubit_sensitivity_analytical,
    calculate_multi_qubit_sensitivity_numerical,
    get_multi_qubit_frequency_at_coupler_flux,
    plot_multi_qubit_sensitivity_curve,
)
from .base import ParameterizedQubit, e, hbar, pi
from .single import GroundedTransmon, RLINE

RNAN = 1e20


def _validate_readout_couple_mode(couple_mode: str) -> None:
    if couple_mode != 'capac':
        raise ValueError(f'Unsupported couple_mode: {couple_mode}')


def _raise_multiqubit_placeholder(owner: str, method_name: str) -> None:
    raise NotImplementedError(f'{owner}.{method_name}() is not implemented yet.')


class GroundedTransmonList(GroundedTransmon):
    
    """
        A list of grounded transmons with different parameters. 
    """
    
    def __init__(
        self,
        *args, **kwargs
    ):
        _raise_multiqubit_placeholder('GroundedTransmonList', '__init__')
    

class QCRFGRModel(ParameterizedQubit):
    '''
    QCR-FGR Model
    topo: Resonator-FloatingQubit-GroundedTransmon
    
    Inherits cal_coupler_sensitivity() from ParameterizedQubit.
    '''
    def __init__(
            self, 
            capacitance_list: list[float],
            junc_resis_list: list[float],
            qrcouple: list[float],
            flux_list: list[float] = [0,0,0],
            trunc_ener_level: list[float] = [10,8,10],
            *args, **kwargs
        ):
        Cq1g, Cq2g, Cqq, Cc, Cqc = capacitance_list
        self._capac = np.array([
            [Cq1g,  Cqq,  Cqc,],
            [Cqq,  Cq2g,  0,  ],
            [Cqc, 0,    Cc,   ],
        ])
        self._resis = np.ones_like(self._capac)*RNAN
        self._resis[0,1]=self._resis[1,0]=junc_resis_list[0]
        self._resis[2,2]=junc_resis_list[1]
        self._flux = np.zeros_like(self._capac)
        self._flux[0,1]=self._flux[1,0]=flux_list[0]
        self._flux[2,2]=flux_list[1]
        self._qrcouple_term = qrcouple
        super().__init__(
            capacitances=self._capac, 
            junctions_resistance=self._resis,
            fluxes=self._flux, 
            trunc_ener_level=trunc_ener_level, 
            structure_index=[2,1], 
            *args, **kwargs)
        self._refresh_basic_metrics()
        self.print_basic_info()

    def _refresh_basic_metrics(self):
        stateNumList = [[1,0],[0,1],[2,0]]
        standard_state = cal_product_state_list(stateNumList, self._Nlevel)
        state_index = [self.find_state(state) for state in standard_state]

        self.qubit_f01 = self.get_energylevel(state_index[0])/2/pi
        self.qubit_anharm = self.get_energylevel(state_index[2])/2/pi-2*self.qubit_f01
        self.coupler_f01 = self.get_energylevel(state_index[1])/2/pi
        self.rq_g = self.get_readout_couple(readout_freq=6.5e9, couple_mode='capac', is_print=False)
        self.qc_g = self.get_coupler_couple(is_print=False)

    def change_hamiltonian(self, new_hamiltonian):
        updated = super().change_hamiltonian(new_hamiltonian)
        self._refresh_basic_metrics()
        return updated
        
    def print_basic_info(self, is_print: bool = True):
        if is_print:
            print(f'Qubit frequency: {self.qubit_f01:.3f} GHz')
            print(f'Coupler frequency: {self.coupler_f01:.3f} GHz')
            print(f'Qubit anharmonicity: {self.qubit_anharm*1e3:.3f} MHz')

            print(f'Readout coupling strenth: {self.rq_g/1e6/2/pi:.3f} MHz (res_freq=6.5GHz, Capac couple)')
            print(f"Qubit-Coupler direct coupling: {self.qc_g*1e3:.3f}MHz")

    def get_coupler_couple(self, mode: str = 'overlap', is_print: bool = True) -> float:
        """
        Get the coupling strenth between qubit and coupler.
        """
        if mode == 'direct':
            Cqsum = e**2/2/self.Ec[0,0]/1e9/hbar
            Ccsum = e**2/2/self.Ec[1,1]/1e9/hbar
            eta = self._capac[1,1]/(self._capac[0,0]+self._capac[1,1])
            self.qc_g = self._capac[0][2]*eta*np.sqrt(self.qubit_f01*self.coupler_f01/Cqsum/Ccsum)/2
        elif mode == 'overlap':
            statelist = [[1,0],[0,1]]
            standard_state = cal_product_state_list(statelist, self._Nlevel)
            hamil = self.get_hamiltonian()
            self.qc_g = abs(standard_state[1].dag()*hamil*standard_state[0])/2/pi
        
        if is_print:
            print(f"Qubit-Coupler direct coupling: {self.qc_g*1e3:.3f}MHz")
        return self.qc_g

    def get_readout_couple(
        self, 
        readout_freq:float,
        couple_mode: str = 'capac',
        is_print: bool = True,
    ):
        _validate_readout_couple_mode(couple_mode)
        omega_qubit = self.get_energylevel(1)*1e9
        omega_res = readout_freq*2*pi
        Cq = e**2/2/self.Ec[0,0]/1e9/hbar
        Cr = 1/8/readout_freq/RLINE
        Cqr1, Cqr2 = self._qrcouple_term
        Cq1 = self.Maxwellmat['capac'][0,0]-self._capac[0,1]-Cqr1
        Cq2 = self.Maxwellmat['capac'][1,1]-self._capac[1,0]-Cqr2
        C_eff = abs(Cqr1*Cq1-Cqr2*Cq2)/(Cq1+Cq2)
        g = C_eff*np.sqrt(omega_res*omega_qubit/(Cr*Cq))/2
        if is_print:
            print(f'Capacitance coupling strenth: {g/1e6/2/pi:.3f}MHz')
        return g



class FGF1V1Coupling(ParameterizedQubit):
    '''
    Topo: F-G-F
    
    Inherits cal_coupler_sensitivity() from ParameterizedQubit.
    '''
    
    def __init__(
        self,
        capacitance_list: list[float],
        junc_resis_list: list[float],
        qrcouple: list[float],
        flux_list: list[float] = [0,0,0],
        trunc_ener_level: list[float] = [10,8,10],
        is_print: bool = True,
        *args, **kwargs
    ):
        """
            capacitance_list: set of all capacitances neccesary, [C11, C12, C1q, Cc, C2q, C21, C22, Cqq, Cq1c, Cq2c]
            junc_resis_list: the junc_resistance list of F-G-F
            flux_list: the flux list of F-G-F
        """
        C11g, C12g, C1q, Cc, C2q, C21g, C22g, Cqq, Cq1c, Cq2c = capacitance_list
        self._capac = np.array([
            [C11g,  C1q,  Cq1c,  Cqq,    0],
            [C1q,  C12g,  0,     0,      0],
            [Cq1c, 0,    Cc,    Cq2c,   0],
            [Cqq,  0,    Cq2c,  C21g,  C2q],
            [0,    0,    0,     C2q,  C22g]
        ])
        self._resis = np.ones_like(self._capac)*RNAN
        self._resis[0,1]=self._resis[1,0]=junc_resis_list[0]
        self._resis[2,2]=junc_resis_list[1]
        self._resis[3,4]=self._resis[4,3]=junc_resis_list[2]
        self._flux = np.zeros_like(self._capac)
        self._flux[0,1]=self._flux[1,0]=flux_list[0]
        self._flux[2,2]=flux_list[1]
        self._flux[3,4]=self._flux[4,3]=flux_list[2]
        self._Nlevel = trunc_ener_level
        self._qrcouple_term = qrcouple
        super().__init__(
            capacitances=self._capac,
            junctions_resistance=self._resis,
            fluxes=self._flux,
            trunc_ener_level=self._Nlevel,
            structure_index=[2,1,2],
            *args, **kwargs
        )
        self._refresh_basic_metrics()
        self.print_basic_info(is_print=is_print)

    def _refresh_basic_metrics(self):
        stateNumList = [[0,0,0],[0,0,1],[1,0,0],[0,1,0],[1,0,1],[0,0,2],[2,0,0]]
        standard_state = cal_product_state_list(stateNumList, self._Nlevel)
        state_index = [self.find_state(state) for state in standard_state]

        self.qubit1_f01 = self.get_energylevel(state_index[1])/2/pi
        self.qubit2_f01 = self.get_energylevel(state_index[2])/2/pi
        self.qubit_f01 = (self.qubit1_f01+self.qubit2_f01)/2
        self.coupler_f01 = self.get_energylevel(state_index[3])/2/pi
        self.qubit1_anharm = self.get_energylevel(state_index[5])/2/pi-2*self.qubit1_f01
        self.qubit2_anharm = self.get_energylevel(state_index[6])/2/pi-2*self.qubit2_f01
        self.qubit_anharm = (self.qubit1_anharm+self.qubit2_anharm)/2

        self.qr_g = self.get_readout_couple(readout_freq=6.5e9, couple_mode='capac', is_print=False)
        self.qq_g = self.get_qq_dcouple(is_print=False)
        self.qc_g = self.get_qc_couple(is_print=False)
        self.qq_geff = self.get_qq_ecouple(is_print=False)

    def change_hamiltonian(self, new_hamiltonian):
        updated = super().change_hamiltonian(new_hamiltonian)
        self._refresh_basic_metrics()
        return updated
    
    def print_basic_info(self, is_print: bool = True):
        if is_print:
            print(f'Qubit 1 frequency: {self.qubit1_f01:.3f} GHz')
            print(f'Qubit 2 frequency: {self.qubit2_f01:.3f} GHz')
            print(f'Coupler frequency: {self.coupler_f01:.3f} GHz')
            print(f'Qubit anharmonicity: {self.qubit_anharm*1e3:.3f} MHz')
            print(f'Readout coupling strenth: {self.qr_g/1e6/2/pi:.3f} MHz (Read_freq=6.5GHz, Capac couple)')
            print(f"Qubit-Qubit direct coupling: {self.qq_g*1e3:.3f}MHz")
            print(f"Qubit-Coupler direct coupling: {self.qc_g*1e3:.3f}MHz")
            print(f"Qubit-Qubit effective coupling: {self.qq_geff*1e3:.3f} MHz")

        return True

    @property    
    def capac_keywords(self):
        print("C11, C12, C1q, Cc, C2q, C21, C22, Cqq, Cq1c, Cq2c")
        return ["C11", "C12", "C1q", "Cc", "C2q", "C21", "C22", "Cqq", "Cq1c", "Cq2c"]
    
    def topology(self):
        print('F-G-F')
        return 0
    
    def QubitDephasingbyCouplerThermal(self, coupler_flux):
        _raise_multiqubit_placeholder('FGF1V1Coupling', 'QubitDephasingbyCouplerThermal')
    
    def get_readout_couple(
        self, 
        readout_freq:float,
        couple_mode: str = 'capac',
        is_print: bool = True,
    ) -> float:
        _validate_readout_couple_mode(couple_mode)
        omega_qubit = self.qubit1_f01*1e9*2*pi
        omega_res = readout_freq*2*pi
        Cq = e**2/2/self.Ec[0,0]/1e9/hbar
        Cr = 1/8/readout_freq/RLINE
        Cqr1, Cqr2 = self._qrcouple_term
        Cq1 = self.Maxwellmat['capac'][0,0]-self._capac[0,1]-Cqr1
        Cq2 = self.Maxwellmat['capac'][1,1]-self._capac[1,0]-Cqr2
        C_eff = abs(Cqr1*Cq1-Cqr2*Cq2)/(Cq1+Cq2)
        self.qr_g = C_eff*np.sqrt(omega_res*omega_qubit/(Cr*Cq))/2
        if is_print:
            print(f'Capacitance coupling strenth: {self.qr_g/1e6/2/pi:.3f}MHz')
        return self.qr_g
    
    def get_qc_couple(self, mode: str = 'overlap', is_print: bool = True) -> float:
        '''
        Get Qubit - Coupler direct coupling strength.
        
        mode: str
            'direct': calculating by couple capacitance.
            'overlap': calculating by state overlap.
        '''
        if mode=='direct':
            Cqsum = e**2/2/self.Ec[0,0]/1e9/hbar
            Ccsum = e**2/2/self.Ec[1,1]/1e9/hbar
            Cqc = e**2/2/self.Ec[0,1]/1e9/hbar
            eta = self._capac[1,1]/(self._capac[0,0]+self._capac[1,1])
            self.qc_g = np.sqrt(self.qubit1_f01*self.coupler_f01*Cqsum*Ccsum)/2/Cqc
        elif mode=='overlap':
            statelist = [[0,0,1],[0,1,0],[1,0,0]]
            standard_state = cal_product_state_list(statelist, self._Nlevel)
            hamil = self.get_hamiltonian()
            self.q1c_g = abs(standard_state[1].dag()*hamil*standard_state[0])/2/pi
            self.q2c_g = abs(standard_state[1].dag()*hamil*standard_state[2])/2/pi
            self.qc_g = (self.q1c_g+self.q2c_g)/2
        
        if is_print:
            print(f"Qubit-Coupler direct coupling: {self.qc_g*1e3:.3f}MHz")
        return self.qc_g
    
    def get_qq_dcouple(self, mode: str = 'overlap', is_print: bool = True) -> float:
        '''
        Get Qubit - Qubit direct coupling strength.
        
        mode: str
            'capac': calculating by couple capacitance.
            'overlap': calculating by state overlap.
        '''
        if mode=='capac':
            Cq1sum = e**2/2/self.Ec[0,0]/1e9/hbar
            # Ccsum = e**2/2/self.Ec[1,1]/1e9/hbar
            Cq2sum = e**2/2/self.Ec[2,2]/1e9/hbar
            Cqq = e**2/2/self.Ec[0,2]/1e9/hbar
            # Cqc1 = e**2/2/self.Ec[0,1]/1e9/hbar
            # Cqc2 = e**2/2/self.Ec[1,2]/1e9/hbar
            eta1 = self._capac[1,1]/(self._capac[0,0]+self._capac[1,1])
            eta2 = self._capac[3,3]/(self._capac[4,4]+self._capac[3,3])
            # eta = eta1*eta2
            # eta = Cqc1*Cqc2/Cqq/Ccsum+1
            self.qq_g = np.sqrt(self.qubit1_f01*self.qubit2_f01*Cq1sum*Cq2sum)/2/Cqq
        elif mode=='overlap':
            statelist = [[0,0,1],[1,0,0]]
            standard_state = cal_product_state_list(statelist, self._Nlevel)
            hamil = self.get_hamiltonian()
            self.qq_g = abs(standard_state[1].dag()*hamil*standard_state[0])/2/pi
        else:
            raise ValueError(f"mode {mode} is not supported for type {type}")
        
        if is_print:
            print(f"Qubit-Qubit direct coupling: {self.qq_g*1e3:.3f}MHz")
        return self.qq_g
    
    def get_qq_ecouple(self, method: str = 'ES', is_print: bool = True) -> float:
        '''
        Get Qubit - Qubit effective coupling strength.
        
        method: str
            'ED': calculating by energy difference.
            'SW': calculating by SW formula.
            'ES': calculating by ES formula.
        '''
        if method=='ED':
            """
            ED requires the two qubit is just the same frequency, so it only works when we design the same frequency qubits. 
            """
            freq_q1 = self.qubit1_f01
            freq_q2 = self.qubit2_f01
            self.qq_geff = (freq_q1-freq_q2)/2
            
        elif method=='SW':
            '''
            SW requires anharmonicity is far lower than the frequency_diff between qubits and coupler, and the g_12 is a second order small 
            quantity of g_1c and g_2c. So it doesn't work when coupler_detune is large. 
            '''
            freq_q1 = self.qubit1_f01
            freq_q2 = self.qubit1_f01
            freq_c = self.coupler_f01
            
            delta1 = freq_q1-freq_c
            delta2 = freq_q2-freq_c
            sum1 = freq_q1+freq_c
            sum2 = freq_q2+freq_c
            Cc = self._capac[2,2]
            C12 = self._capac[0,3]
            C1c = self._capac[0,2]
            C2c = self._capac[2,3]
            C1 = self._capac[0,0]
            C2 = self._capac[1,1]
            eta = C1c*C2c/C12/Cc
            self.qq_geff = (freq_c*eta*(1/(delta1)+1/(delta2)-1/(sum1)-1/(sum2))/4+eta+1)*C12*np.sqrt(freq_q1*freq_q2/C1/C2)/2
        
        elif method=='ES':
            freq_q1 = self.qubit1_f01
            freq_q2 = self.qubit1_f01
            freq_c = self.coupler_f01
            g1 = g2 = self.qc_g
            g12 = self.qq_g

            delta1 = self.qubit1_f01 - self.coupler_f01
            delta2 = self.qubit2_f01 - self.coupler_f01
            sum1 = self.qubit1_f01 + self.coupler_f01
            sum2 = self.qubit2_f01 + self.coupler_f01

            self.qq_geff = g1*g2*(1/delta1+1/delta2-1/sum1-1/sum2)/2+g12

        else:
            raise ValueError(f"method {method} is not supported.")
        
        if is_print:
            print(f"Qubit-Qubit effective coupling: {self.qq_geff*1e3:.3f} MHz")
        
        return self.qq_geff



class FGF2V7Coupling(ParameterizedQubit):
    '''
            G   G
            |   |
    Topo: G-F-G-F-G
            |   |
            G   G
    '''
    
    def __init__(
        self,
        capacitance_list: list[float],
        junc_resis_list: list[float],
        flux_list: list[float] = [0,0],
        trunc_ener_level: list[float] = [8,5],
        is_symmetric: bool = True,
        *args, **kwargs
    ):
        """
            capacitance_list: set of all capacitances neccesary, [C11, C12, C1q, Cc, C2q, C21, C22, Cqq, Cq1c, Cq2c]
            junc_resis_list: the junc_resistance list of F-G-F
            flux_list: the flux list of F-G-F
        """
        if is_symmetric:
            C11g, C12g, C1q, Cc, C2q, C21g, C22g, Cqq, Cq1c, Cq2c = capacitance_list
            self._capac = np.array([
                [C11g,  C1q,  Cq1c,  Cqq,   0,      Cq1c,   0,      0,      0,      0,      0],
                [C1q,  C12g,  0,     0,     0,      0,      Cq2c,   Cq2c,   0,      0,      0],
                [Cq1c, 0,    Cc,    Cq2c,   0,      0,      0,      0,      0,      0,      0],
                [Cqq,  0,    Cq2c,  C21g,   C2q,    0,      0,      0,      Cq2c,   0,      0],
                [0,    0,    0,     C2q,    C22g,   0,      0,      0,      0,      Cq1c,   Cq1c],
                [Cq1c, 0,    0,     0,      0,      Cc,     0,      0,      0,      0,      0],
                [0,    Cq2c, 0,     0,      0,      0,      Cc,     0,      0,      0,      0],
                [0,    Cq2c, 0,     0,      0,      0,      0,      Cc,     0,      0,      0],
                [0,    0,    0,     Cq2c,   0,      0,      0,      0,      Cc,     0,      0],
                [0,    0,    0,     0,      Cq1c,   0,      0,      0,      0,      Cc,     0],
                [0,    0,    0,     0,      Cq1c,   0,      0,      0,      0,      0,      Cc],
            ])
            self._resis = np.ones_like(self._capac)
            self._resis[0,1]=self._resis[1,0]=junc_resis_list[0]
            self._resis[3,4]=self._resis[4,3]=junc_resis_list[0]
            for ii in [2,5,6,7,8,9,10]:
                self._resis[ii,ii]=junc_resis_list[1]
            self._flux = np.zeros_like(self._capac)
            self._flux[0,1]=self._flux[1,0]=flux_list[0]
            self._flux[3,4]=self._flux[4,3]=flux_list[0]
            for ii in [2,5,6,7,8,9,10]:
                self._flux[ii,ii]=flux_list[1]
            self._Nlevel = [trunc_ener_level[0],trunc_ener_level[1],trunc_ener_level[0]]+[trunc_ener_level[1]]*6
        else:
            self._capac = np.array(capacitance_list)
            self._resis = np.array(junc_resis_list)
            self._flux = np.array(flux_list)
            self._Nlevel = [trunc_ener_level[0],trunc_ener_level[1],trunc_ener_level[0]]+[trunc_ener_level[1]]*6
        super().__init__(
            capacitances=self._capac,
            junctions_resistance=self._resis,
            fluxes=self._flux,
            trunc_ener_level=self._Nlevel,
            structure_index=[2,1,2,1,1,1,1,1,1],
            *args, **kwargs
        )
        self._refresh_basic_metrics()
        self.print_basic_info()

    def _refresh_basic_metrics(self):
        self.qubit1_f01 = self.get_energylevel(1)/2/pi
        self.qubit2_f01 = self.get_energylevel(2)/2/pi
        self.qubit_f01 = (self.qubit1_f01+self.qubit2_f01)/2
        el_initial = [self.get_energylevel(ii)/2/pi for ii in range(1,7)]

        anhar_pre = -self.Ec[0,0]/2/pi
        target = 2*self.qubit1_f01+anhar_pre
        indice_200 = np.where(np.abs(el_initial - target) < 80e-3)[0][0]
        target = self.qubit2_f01+self.qubit1_f01
        indice_101 = np.where(np.abs(el_initial - target) < 1e-3)[0][0]

        indice_010 = 13-2*indice_200-indice_101

        self.qubit1_anharm = el_initial[indice_200]-2*self.qubit1_f01
        self.qubit2_anharm = el_initial[indice_200+1]-2*self.qubit2_f01
        self.qubit_anharm = (self.qubit1_anharm+self.qubit2_anharm)/2

        self.coupler_f01 = el_initial[indice_010]

    def change_hamiltonian(self, new_hamiltonian):
        updated = super().change_hamiltonian(new_hamiltonian)
        self._refresh_basic_metrics()
        return updated
    
    def print_basic_info(self, is_print: bool = True):
        if is_print:
            print(f'Qubit 1 frequency: {self.qubit1_f01:.3f}GHz')
            print(f'Qubit 2 frequency: {self.qubit2_f01:.3f}GHz')
            print(f'Coupler frequency: {self.coupler_f01:.3f}GHz')
            print(f'Qubit anharmonicity: {self.qubit_anharm*1e3:.3f}MHz')
        
        return True

class FGFGG1V1V3Coupling(ParameterizedQubit):
    
    def __init__(
        self,
        *args,
        **kwargs,
    ):
        _raise_multiqubit_placeholder('FGFGG1V1V3Coupling', '__init__')


def _qcrfgr_cal_coupler_sensitivity(
    self,
    coupler_flux_point: float,
    method: str = 'numerical',
    flux_step: float = 1e-4,
    qubit_idx: int = 0,
    is_print: bool = True,
    is_plot: bool = False,
) -> float:
    return analyze_multi_qubit_coupler_sensitivity(
        self,
        coupler_flux_point=coupler_flux_point,
        method=method,
        flux_step=flux_step,
        qubit_idx=qubit_idx,
        qubit_fluxes=None,
        is_print=is_print,
        is_plot=is_plot,
    )


def _qcrfgr_cal_sensitivity_numerical(
    self,
    coupler_flux_point: float,
    flux_step: float,
    qubit_idx: int,
) -> float:
    return calculate_multi_qubit_sensitivity_numerical(
        self,
        coupler_flux_point,
        flux_step,
        qubit_idx,
    )


def _qcrfgr_cal_sensitivity_analytical(
    self,
    coupler_flux_point: float,
    qubit_idx: int,
) -> float:
    return calculate_multi_qubit_sensitivity_analytical(
        self,
        coupler_flux_point,
        qubit_idx,
    )


def _qcrfgr_cal_coupler_self_sensitivity(self, coupler_flux: float) -> float:
    return calculate_multi_qubit_coupler_self_sensitivity(self, coupler_flux)


def _qcrfgr_get_qubit_frequency_at_coupler_flux(
    self,
    coupler_flux: float,
    qubit_idx: int,
    flux_offset: float = 0.0,
) -> float:
    return get_multi_qubit_frequency_at_coupler_flux(
        self,
        coupler_flux,
        qubit_idx=qubit_idx,
        flux_offset=flux_offset,
    )


def _qcrfgr_plot_sensitivity_curve(
    self,
    coupler_flux_point: float,
    flux_step: float,
    qubit_idx: int,
    sensitivity: float,
):
    plot_multi_qubit_sensitivity_curve(
        self,
        coupler_flux_point,
        flux_step,
        qubit_idx,
        sensitivity,
    )


def _fgf1_cal_coupler_sensitivity(
    self,
    coupler_flux_point: float,
    method: str = 'numerical',
    flux_step: float = 1e-4,
    qubit_idx: int = None,
    qubit_fluxes=None,
    is_print: bool = True,
    is_plot: bool = False,
) -> float:
    return analyze_multi_qubit_coupler_sensitivity(
        self,
        coupler_flux_point=coupler_flux_point,
        method=method,
        flux_step=flux_step,
        qubit_idx=qubit_idx,
        qubit_fluxes=qubit_fluxes,
        is_print=is_print,
        is_plot=is_plot,
    )


def _fgf1_cal_sensitivity_numerical(
    self,
    coupler_flux_point: float,
    flux_step: float,
    qubit_idx: int,
    qubit_fluxes=None,
) -> float:
    return calculate_multi_qubit_sensitivity_numerical(
        self,
        coupler_flux_point,
        flux_step,
        qubit_idx,
        qubit_fluxes=qubit_fluxes,
    )


def _fgf1_cal_sensitivity_analytical(
    self,
    coupler_flux_point: float,
    qubit_idx: int,
) -> float:
    return calculate_multi_qubit_sensitivity_analytical(
        self,
        coupler_flux_point,
        qubit_idx,
    )


def _fgf1_cal_coupler_self_sensitivity(self, coupler_flux: float) -> float:
    return calculate_multi_qubit_coupler_self_sensitivity(self, coupler_flux)


def _fgf1_get_qubit_frequency_at_coupler_flux(
    self,
    coupler_flux: float,
    qubit_idx: int = None,
    qubit_fluxes=None,
    flux_offset: float = 0.0,
) -> float:
    return get_multi_qubit_frequency_at_coupler_flux(
        self,
        coupler_flux,
        qubit_idx=qubit_idx,
        qubit_fluxes=qubit_fluxes,
        flux_offset=flux_offset,
    )


def _fgf1_plot_sensitivity_curve(
    self,
    coupler_flux_point: float,
    flux_step: float,
    qubit_idx: int,
    sensitivity: float,
):
    plot_multi_qubit_sensitivity_curve(
        self,
        coupler_flux_point,
        flux_step,
        qubit_idx,
        sensitivity,
    )


QCRFGRModel.cal_coupler_sensitivity = _qcrfgr_cal_coupler_sensitivity
QCRFGRModel._cal_sensitivity_numerical = _qcrfgr_cal_sensitivity_numerical
QCRFGRModel._cal_sensitivity_analytical = _qcrfgr_cal_sensitivity_analytical
QCRFGRModel._cal_coupler_self_sensitivity = _qcrfgr_cal_coupler_self_sensitivity
QCRFGRModel._get_qubit_frequency_at_coupler_flux = _qcrfgr_get_qubit_frequency_at_coupler_flux
QCRFGRModel._plot_sensitivity_curve = _qcrfgr_plot_sensitivity_curve

FGF1V1Coupling.cal_coupler_sensitivity = _fgf1_cal_coupler_sensitivity
FGF1V1Coupling._cal_sensitivity_numerical = _fgf1_cal_sensitivity_numerical
FGF1V1Coupling._cal_sensitivity_analytical = _fgf1_cal_sensitivity_analytical
FGF1V1Coupling._cal_coupler_self_sensitivity = _fgf1_cal_coupler_self_sensitivity
FGF1V1Coupling._get_qubit_frequency_at_coupler_flux = _fgf1_get_qubit_frequency_at_coupler_flux
FGF1V1Coupling._plot_sensitivity_curve = _fgf1_plot_sensitivity_curve

__all__ = [
    'FGF1V1Coupling',
    'FGF2V7Coupling',
    'FGFGG1V1V3Coupling',
    'GroundedTransmonList',
    'QCRFGRModel',
    'RNAN',
]
