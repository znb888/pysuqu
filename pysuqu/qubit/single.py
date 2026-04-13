'''
Author: Naibin Zhou
USTC
Since 2023-12-05
'''

# import
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from typing import Optional
# local lib
from .analysis import analyze_single_qubit_spectrum
from .base import ParameterizedQubit, Phi0, e, h, hbar, pi

RNAN = 1e20
RLINE = 50

class SingleQubitBase(ParameterizedQubit):
    
    def __init__(
        self,
        capacitance: list[list],
        junction_resistance: list[list],
        inductance: list[list],
        flux: list[list],
        trunc_ener_level: list[int] = [20], 
        junc_ratio: list[list] = [[1]],
        structure: list[int] = [1],
        qr_couple: list[float] = [0],
        qr_couplemode: str = 'capac',
        qubit_class: str = 'NaQ',
        *args, **kwargs
    ):
        # Handle both 1x1 (GroundedTransmon) and 2x2 (FloatingTransmon) capacitance matrices
        if structure[0] == 1:
            self._capac = capacitance[0][0] if isinstance(capacitance[0][0], (int, float)) else np.array(capacitance[0][0])
        else:
            self._capac = np.array(capacitance) if not isinstance(capacitance, np.ndarray) else capacitance
        # Don't overwrite _flux if already set by subclass (e.g., FloatingTransmon)
        if not hasattr(self, '_flux'):
            self._flux = flux
        self._Nlevel = trunc_ener_level
        self.qubit_class = qubit_class
        self._qrcouple_term = qr_couple
        self._qr_couplemode = qr_couplemode
        super().__init__(
            capacitances=capacitance,
            junctions_resistance=junction_resistance,
            inductances=inductance,
            fluxes=flux,
            trunc_ener_level=trunc_ener_level, 
            junc_ratio=junc_ratio,
            structure_index=structure,
            *args, **kwargs
        )
        if hasattr(self, '_energylevels'):
            self._refresh_basic_state()
        self.print_basic_info()

    def _refresh_basic_metrics(self):
        metrics = analyze_single_qubit_spectrum(self)
        self.f01 = metrics.f01
        self.anharmonicity = metrics.anharmonicity
        return metrics

    def _calculate_readout_couple(
        self,
        readout_freq: float = 6.5e9,
        couple_mode: str = 'capac',
    ):
        omega_qubit = self.f01*1e9*2*pi
        omega_res = readout_freq*2*pi

        if couple_mode == 'capac':
            Cq = e**2/2/self.Ec[0,0]/1e9/hbar
            Cr = 1/8/readout_freq/RLINE
            if self.qubit_class == 'FloatingTransmon':
                Cqr1, Cqr2 = self._qrcouple_term
                Cq1 = self.Maxwellmat['capac'][0,0] - self._capac[0][1] - Cqr1
                Cq2 = self.Maxwellmat['capac'][1,1] - self._capac[1][0] - Cqr2
                C_eff = abs(Cqr1*Cq1-Cqr2*Cq2)/(Cq1+Cq2)
            elif self.qubit_class == 'GroundedTransmon':
                C_eff = self._qrcouple_term[0]
            else:
                print(f'Qubit class {self.qubit_class} not supported!')
                return False

            return C_eff*np.sqrt(omega_res*omega_qubit/(Cr*Cq))/2

        print(f'Coupling mode {couple_mode} not supported!')
        return False

    def _refresh_basic_state(self):
        metrics = self._refresh_basic_metrics()
        if hasattr(self, '_qrcouple_term') and hasattr(self, '_qr_couplemode'):
            self.qr_g = self._calculate_readout_couple(
                readout_freq=6.5e9,
                couple_mode=self._qr_couplemode,
            )
        return metrics

    def change_hamiltonian(self, new_hamiltonian):
        updated = super().change_hamiltonian(new_hamiltonian)
        self._refresh_basic_state()
        return updated

    @staticmethod
    def _raise_placeholder(method_name: str):
        raise NotImplementedError(f"SingleQubitBase.{method_name}() is not implemented yet.")
    
    def print_basic_info(self, is_print: bool = True):
        if is_print:
            print(f'Qubit frequency: {self.f01:.3f} GHz')
            print(f'Qubit anharmonicity: {self.anharmonicity:.3f} GHz')
            print(f'Qubit readout coupling strenth: {self.qr_g/1e6/2/pi:.3f} MHz.(Read_freq=6.5 GHz, Capac couple, {self._qrcouple_term})')
    
    def Capac_drive(
        self, 
        coupl_capac: float, 
        drive_voltage: float
    ):
        '''
        Input unit: SI, Output unit: same with print content
        
        drive_voltage: the voltage amplitude of drive signal. 
        coupl_capac: the coupling capacitance between qubit and drive line. 
        '''
        C_strength = coupl_capac*drive_voltage*(8*self.Ej[0,0]*self.Ec[0,0]**3)**(1/4)/e
        T1_c = self._capac/((self.f01*1e9*2*pi)**2*coupl_capac**2*RLINE)
        
        print(f'Qubit freq: {self.f01} GHz')
        print(f'Capac_coupling drive_strength: {C_strength*1e3} MHz')
        print(f'T1 suppressed by drive_capac_coupling: {T1_c*1e6} us')
        return [self.f01, C_strength*1e3, T1_c*1e6]
    
    def Induc_drive(
        self, 
        coupl_induc: float, 
        drive_voltage:float
    ):
        '''
        Input unit: SI, Output unit: same with print content
        
        drive_voltage: the voltage amplitude of drive signal. 
        coupl_induc: the coupling inductance between qubit and drive line. 
        '''
        current = drive_voltage/RLINE
        I_strength = 2*pi*coupl_induc*current*(2*self.Ec[0,0]*self.Ej[0,0]**3)**(1/4)/Phi0
        T1_i = RLINE/((self.f01*1e9*2*pi)**4*coupl_induc**2*self._capac)
        
        print(f'Qubit freq: {self.f01} GHz')
        print(f'Induc_coupling drive_strength: {I_strength*1e3} MHz')
        print(f'T1 suppressed by drive_induc_coupling: {T1_i*1e6} us')
        return [self.f01, I_strength*1e3, T1_i*1e6]
    
    def drive_strength(
        self, 
        voltage_amp: float, 
        coupl_term: float, 
        mode: str = 'ind'
    ):
        '''
        Input unit: SI, Output unit: MHz
        
        voltage_amp: the voltage amplitude of drive signal. 
        coupl_term: coupling_capac for mode 'cap', coupling_induc for mode 'ind'. 
        '''
        if mode == 'ind':
            current = voltage_amp/RLINE
            I_strength = 2*pi*coupl_term*current*(2*self.Ec[0,0]*self.Ej[0,0]**3)**(1/4)/Phi0*1e3
            print(f'Capac_coupling drive_strength: {I_strength} MHz')
            return I_strength
        
        elif mode == 'cap':
            C_strength = coupl_term*voltage_amp*(8*self.Ej[0,0]*self.Ec[0,0]**3)**(1/4)/e*1e3
            print(f'Induc_coupling drive_strength: {C_strength} MHz')
            return C_strength
        
        elif mode == 'ind+cap':
            current = voltage_amp/RLINE
            I_strength = 2*pi*coupl_term[0]*current*(2*self.Ec[0,0]*self.Ej[0,0]**3)**(1/4)/Phi0*1e3
            C_strength = coupl_term[1]*voltage_amp*(8*self.Ej[0,0]*self.Ec[0,0]**3)**(1/4)/e*1e3
            print(f'Induc_coupling drive_strength: {I_strength} MHz\nCapac_coupling drive_strength: {C_strength} MHz')
            return [I_strength, C_strength]
            
        elif mode == 'cap+ind':
            current = voltage_amp/RLINE
            I_strength = 2*pi*coupl_term[1]*current*(2*self.Ec[0,0]*self.Ej[0,0]**3)**(1/4)/Phi0*1e3
            C_strength = coupl_term[0]*voltage_amp*(8*self.Ej[0,0]*self.Ec[0,0]**3)**(1/4)/e*1e3
            print(f'Capac_coupling drive_strength: {C_strength} MHz\nInduc_coupling drive_strength: {I_strength} MHz')
            return [C_strength, I_strength]
    
    def drive_loss(
        self, 
        capa_drive: float, 
        indu_drive:float
    ):
        freq = self.f01*1e9*2*pi
        T1_c = self._capac/(freq**2*capa_drive**2*RLINE)
        T1_i = RLINE/(freq**4*indu_drive**2*self._capac)
        
        T1_drive = 1/(1/T1_i+1/T1_c)
        
        print(f'Qubit freq: {freq/1e9/2/pi} GHz\nT1 suppressed by drive_induc_coupling: {T1_i*1e6} us\nT1 suppressed by drive_capac_coupling: {T1_c*1e6} us\nT1 induced by drive_line: {T1_drive*1e6} us')
        
        return [freq/1e9/2/pi, T1_i, T1_c, T1_drive]
    
    def EnvsReadoutphoton(self):
        '''
        The relationship between qubit frequency and photon number in readout cavity. 
        Can be used to calculate the kappa of readout cavity. 
        '''
        self._raise_placeholder("EnvsReadoutphoton")
    
    def get_Readout_parameter(
        self, 
        rq_coupleterm: float,
        readout_freq: float,
        rf_coupleterm: float = None,
        kappa_read: float = None,
        kappa_purcell: float = None,
        relposition: float = None,
        purcell_freq: float = None,
        coupling_mode: dict = {'rq':'capac','rf':'induc'}
    ):
        '''
        Input Unit: SI, Output unit: same with print content
        
        rq_coupleterm: the coupling capacitance/inductance between qubit and readout cavity, F/H. 
        readout_freq: the resonant frequency of readout cavity. 
        rf_coupleterm: the coupling capacitance/inductance between readout and purcell filter, F/H.
        kappa_read: the decay rate or 3dB width of reaout cavity.
        kappa_purcell:  the decay rate or 3dB width of purcell filter.
        relposition: the relative position of readout, (0,1). 
        purcell_freq: the resonant frequency of purcell filter. 
        capac_pr: the coupling capacitance between readout and purcell filter. 
        '''
        metrics = self._refresh_basic_metrics()
        freq = metrics.f01*1e9
        Cr = 1/8/readout_freq/RLINE
        Lr = 2*RLINE/pi**2/readout_freq
        if coupling_mode['rq'] == 'capac':
            Cq = e**2/2/self.Ec[0,0]/1e9/hbar
            Cqq = (Cq*Cr+Cq*rq_coupleterm+Cr*rq_coupleterm)/(Cr+rq_coupleterm)
            Crr = (Cq*Cr+Cq*rq_coupleterm+Cr*rq_coupleterm)/(Cq+rq_coupleterm)
            Ccc = (Cq*Cr+Cq*rq_coupleterm+Cr*rq_coupleterm)/(rq_coupleterm)
            g = np.sqrt(freq*readout_freq)*np.sqrt(Cqq*Crr)/2/Ccc
        elif coupling_mode['rq'] == 'induc':
            raise NotImplementedError(
                "SingleQubitBase.get_Readout_parameter() does not implement "
                "coupling_mode['rq'] == 'induc' yet."
            )
        else:
            raise ValueError("please choose coupling_mode in ['capac','induc']")
        print(f'Frequencies of Qubit and Readout: {freq/1e9:.4f} GHz, {readout_freq/1e9:.4f} GHz')
        print(f'The coupling strength between Qubit and Readout: {g/1e6:.4f} MHz')
        Delta = freq-readout_freq
        chi = -g**2/Delta/(1+Delta/metrics.anharmonicity/1e9)
        print(f'The dispersive shift of qubit readout: {chi/1e6:.4f} MHz')
        
        if kappa_purcell is not None:
            if kappa_read is None:
                if purcell_freq is None:
                    purcell_freq = readout_freq
                if coupling_mode['rf'] == 'capac':
                    Cp = 1/4/purcell_freq/RLINE
                    coupl_pr = rf_coupleterm*np.sqrt(readout_freq*purcell_freq)/np.sqrt(Cr*Cp)/2*abs(np.cos(pi*relposition))
                elif coupling_mode['rf'] == 'induc':
                    Lp = RLINE/pi**2/purcell_freq
                    coupl_pr = rf_coupleterm*np.sqrt(readout_freq*purcell_freq)/np.sqrt(Lp*Lr)/2*np.sin(pi*relposition)
                kappa_read_eff = 4*coupl_pr**2*kappa_purcell/2/(kappa_purcell**2/4+4*(readout_freq-purcell_freq)**2)
                T1_p = (Delta/g)**2*readout_freq*(Delta*2/kappa_purcell)**2/freq/kappa_read_eff
                print(f'The coupling strength between Readout and Purcell filter: {coupl_pr/1e6:.4f} MHz')
                print(f'The kappa_eff of readout is {kappa_read_eff/1e6:.4f} MHz')
                print(f'T1 upper bound by purcell effect: {T1_p*1e6:.4f} us')
                return [freq, readout_freq, g, chi, coupl_pr, T1_p]
            else:
                T1_p = (Delta/g)**2*readout_freq*(Delta*2/kappa_purcell)**2/freq/kappa_read
                print(f'T1 suppressed by readout cavity: {T1_p*1e6:.4f} us')
                return [freq, readout_freq, g, chi, T1_p]
        else:
            if kappa_read is None:
                return [freq, readout_freq, g, chi]
            else:
                T1_p = (Delta/g)**2*readout_freq/freq/kappa_read
                print(f'T1 suppressed by readout cavity: {T1_p*1e6:.4f} us')
                return [freq, readout_freq, g, chi, T1_p]
        
    
    def EnvsCapa(self):
        self._raise_placeholder('EnvsCapa')
    
    def EnvsInduc(self):
        self._raise_placeholder('EnvsInduc')
    
    def EnvsJuncResis(self):
        self._raise_placeholder('EnvsJuncResis')

    def get_readout_couple(
        self,
        readout_freq: float = 6.5e9,
        couple_mode: str = 'capac',
        is_print: bool = True,
    ):
        self._refresh_basic_metrics()
        self.qr_g = self._calculate_readout_couple(
            readout_freq=readout_freq,
            couple_mode=couple_mode,
        )
        if is_print and self.qr_g is not False:
            print(f'Capacitance coupling strenth: {self.qr_g/1e6/2/pi:.3f}MHz')
        return self.qr_g


class GroundedTransmon(SingleQubitBase):
    
    '''
    Single grounded transmon. 
    '''
    
    def __init__(
        self,
        capacitance: float,
        junction_resistance: float,
        inductance: float = 1e20,
        flux: float = 0,
        trunc_ener_level: int = 20,
        junc_ratio: float = 1,
        qr_couple: list[float] = [5e-15],
        qr_couplemode: str = 'capac',
        *args, **kwargs
    ):
        self._capac = [[capacitance]]
        self._resis = [[junction_resistance]]
        self._Nlevel = trunc_ener_level
        self._flux = flux
        super().__init__(
            capacitance=self._capac,
            junction_resistance=self._resis,
            inductance=[[inductance]],
            flux=[[flux]],
            flux_value=flux,
            trunc_ener_level=[trunc_ener_level], 
            junc_ratio=[[junc_ratio]],
            qr_couple = qr_couple,
            qr_couplemode = qr_couplemode,
            qubit_class = self.__class__.__name__,
            *args, **kwargs
        )
    
    def fit_by_frequency_and_anharmonicity(
        self,
        test_freq: float,
        test_anh: float,
        guess: list[float] = [85e-15, 10000],
    ):
        origin_capac = np.array(self.get_element_matrices('capac'), copy=True)
        origin_resis = np.array(self.get_element_matrices('resis'), copy=True)

        def cost_func(para):
            capac, resis = para
            self.change_para(capac=[[capac]], resis=[[resis]])
            freq = self.f01
            anh = self.anharmonicity
            return abs(freq-test_freq)+abs(anh-test_anh)
        
        result = sp.optimize.minimize(cost_func, guess, method='Nelder-Mead')
        print(f'The most fit para[Capacitance, Resistance]: {result.x}')

        self.change_para(capac=origin_capac, resis=origin_resis)
        return result.x



class FloatingTransmon(SingleQubitBase):
    
    '''
    Single Floating Transmon. 
    '''
    
    def __init__(
        self,
        basic_element: list[float],
        flux: float = 0,
        trunc_ener_level: float = 20,
        junc_ratio: float = 1,
        qr_couple: list[float] = [10e-15,0],
        qr_couplemode: str = 'capac',
        *args, **kwargs
    ):
        '''
        basic_element: a list of basic elements for FloatingQubit, [C1,C2,C_sigma,R]
        '''
        C1, C2, C_sigma, Resis = basic_element
        self._Nlevel = [trunc_ener_level]
        self._capac = np.array([[C1,C_sigma],[C_sigma, C2]])
        self._flux = np.array([[0,flux],[flux,0]])
        self._induc = np.ones_like(self._capac)*RNAN
        self._resis = np.array([[RNAN, Resis],[Resis, RNAN]])
        self._junc_ratio = np.array([[1,junc_ratio],[junc_ratio,1]])
        self._Nlevel = trunc_ener_level
        super().__init__(
            capacitance=self._capac,
            junction_resistance=self._resis,
            inductance=self._induc,
            flux=self._flux,
            flux_value=flux,
            trunc_ener_level=[trunc_ener_level], 
            junc_ratio=self._junc_ratio,
            structure=[2],
            qr_couple = qr_couple,
            qr_couplemode = qr_couplemode,
            qubit_class = self.__class__.__name__,
            *args, **kwargs
        )
    
    def fit_by_frequency_and_anharmonicity(
        self,
        test_freq: float,
        test_anh: float,
        guess: list[float] = [150e-15, 150e-15, 5e-15, 10000],
    ):
        origin_para = [self._capac, self._resis]

        def cost_func(para):
            capac1, capac2, capacq, resis0 = para
            capac = [[capac1, capacq],[capacq, capac2]]
            resis = [[RNAN, resis0],[resis0, RNAN]]
            self.change_para(capac=capac, resis=resis)
            freq = self.f01
            anh = self.anharmonicity
            return abs(freq-test_freq)+abs(anh-test_anh)
        
        result = sp.optimize.minimize(cost_func, guess, method='Nelder-Mead')
        print(f'The most fit para[Capacitance, Resistance]: {result.x}')

        self.change_para(capac=origin_para[0], resis=origin_para[1])
        return result.x

__all__ = [
    'FloatingTransmon',
    'GroundedTransmon',
    'Phi0',
    'RNAN',
    'RLINE',
    'SingleQubitBase',
    'pi',
]



