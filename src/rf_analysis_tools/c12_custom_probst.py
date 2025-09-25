import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from resonator_tools import circuit # use latest resonator_tools package from https://github.com/sebastianprobst/resonator_tools
from resonator_tools.utilities import Watt2dBm, dBm2Watt
from scipy.constants import hbar
import warnings


def my_plotall(self, title='', path='', port='reflection'):
    """
    Plots the measurement and the fit results for this trace

    Overwrites the initial plotall() function from the Probst resonator tools
    with some nicer layout of the figure and returns the figure handle.
    Prevents overlapping of axes labels.

    Parameters
    ----------
    title : str, optional
        Title of the figure. The default is ''.
    path : str, optional
        Path to measurement data. Will be added as sucaption.
        The default is ''.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object of the plot.

    """
    real = self.z_data_raw.real
    imag = self.z_data_raw.imag
    real2 = self.z_data_sim.real
    imag2 = self.z_data_sim.imag

    if port == 'reflection':
        Stext = 'S_{11}'
    else:
        Stext = 'S_{21}'

    # plt.figure(dpi=300)
    fig, axs = plt.subplots(2, 2)
    ax = axs[0, 0]
    ax = plt.subplot(221)
    ax.plot(real, imag, 'o', mfc='none', label='rawdata', markersize=5, color='darkgray')
    ax.plot(real2, imag2, label='fit', color='black')
    ax.set_xlabel('$\Re (%s)$' % Stext)
    ax.set_ylabel('$\Im (%s)$' % Stext)
    ax.legend(frameon=False, fontsize=8)
    ax = axs[0, 1]
    ax.plot(self.f_data * 1e-9, np.absolute(self.z_data_raw), 'o', mfc='none', label='rawdata', markersize=5,
            color='darkgray')
    ax.plot(self.f_data * 1e-9, np.absolute(self.z_data_sim), label='fit', color='black')
    ax.set_xlabel(r'$f$ / GHz')
    ax.set_ylabel('$|%s|$' % Stext)
    ax.ticklabel_format(useOffset=False)
    # plt.ticklabel_format(useMathText=True)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1, 0]
    ax.plot(self.f_data * 1e-9, np.unwrap(np.angle(self.z_data_raw)), 'o', mfc='none', label='rawdata', markersize=5,
            color='darkgray')
    ax.plot(self.f_data * 1e-9, np.unwrap(np.angle(self.z_data_sim)), label='fit', color='black')
    ax.set_xlabel('$f$ / GHz')
    ax.set_ylabel('arg$(%s) / rad$' % Stext)
    ax.ticklabel_format(useOffset=False)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3f'))
    ax.legend(frameon=False, fontsize=8)
    ax = axs[1, 1]
    ax.axis('tight')
    ax.axis('off')
    fit = self.fitresults;
    float_formatter = r"${:.2e}$".format;
    np.set_printoptions(formatter={'float_kind': float_formatter})
    if port == 'notch':  # others not implemented
        cell_con = [[fit['chi_square'], ''],
                    [fit['Ql'], fit['Ql_err']],
                    [fit['absQc'], fit['absQc_err']],
                    [fit['Qc_dia_corr'], ''],
                    [fit['Qi_dia_corr'], fit['Qi_dia_corr_err']],
                    [fit['fr'], fit['fr_err']],
                    [fit['phi0'], fit['phi0_err']],
                    [fit['theta0'], '']]
        for ix, x in enumerate(cell_con):
            for iy, y in enumerate(x):
                if y != '':
                    cell_con[ix][iy] = float_formatter(y)
        rowlabel = [r'$\chi ^2$', r'$Q_l$', r'$|Q_c|$', r'$Re\{Q_c^{-1}\}^{-1}$', r'$Q_i$', r'$f_r$', r'$\phi_0$',
                    r'$\theta _0$']
        collabel = ['Value', 'Error']
        ax.table(cellText=cell_con, rowLabels=rowlabel, colLabels=collabel, loc='center')  # ,
        # # rowColours=colors,

    plt.suptitle(title, fontsize='small')
    plt.tight_layout()

    ax = axs[0, 0]
    ax.text(1.05, 1.05, path,
            verticalalignment='bottom', horizontalalignment='center',
            transform=ax.transAxes, fontsize=6)
    plt.show()
    return fig

circuit.notch_port.plotall = my_plotall
circuit.reflection_port.plotall = my_plotall
circuit.transmission_port.plotall = my_plotall


def my_plotrawdata(self):
    """
    Plots only the measured raw data of this trace.

    Overwrites the initial Probst resonator-tools plot function but there have
    been no major changes yet. It only returns a figure handle in addtion atm.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object of the plot.

    """
    real = self.z_data_raw.real
    imag = self.z_data_raw.imag
    fig, axs = plt.subplots(2, 2)
    plt.subplot(221)
    plt.plot(real, imag, label='rawdata')
    plt.xlabel('Re(S21)')
    plt.ylabel('Im(S21)')
    plt.legend()
    plt.subplot(222)
    plt.plot(self.f_data * 1e-9, np.absolute(self.z_data_raw), label='rawdata')
    plt.xlabel('f (GHz)')
    plt.ylabel('|S21|')
    plt.legend()
    plt.subplot(223)
    plt.plot(self.f_data * 1e-9, np.angle(self.z_data_raw), label='rawdata')
    plt.xlabel('f (GHz)')
    plt.ylabel('arg(|S21|)')
    plt.legend()
    plt.show()
    return fig
circuit.notch_port.plotrawdata = my_plotrawdata


def my_autofit(self,electric_delay=None,fcrop=None,Ql_guess=None, fr_guess=None, guess_delay=True, pass_delay_guess=0, lwr_tau_bound=-np.inf):
    '''
    automatic calibration and fitting
    electric_delay: set the electric delay manually
    fcrop = (f1,f2) : crop the frequency range used for fitting
    '''
    if fcrop is None:
        self._fid = np.ones(self.f_data.size,dtype=bool)
    else:
        f1, f2 = fcrop
        self._fid = np.logical_and(self.f_data>=f1,self.f_data<=f2)
    delay, amp_norm, alpha, fr, Ql, A2, frcal =\
            self.do_calibration(self.f_data[self._fid],self.z_data_raw[self._fid],ignoreslope=True,guessdelay=guess_delay,fixed_delay=electric_delay,Ql_guess=Ql_guess, fr_guess=fr_guess, pass_delay_guess=pass_delay_guess, lwr_tau_bound=lwr_tau_bound)
    self.z_data = self.do_normalization(self.f_data,self.z_data_raw,delay,amp_norm,alpha,A2,frcal)
    self.fitresults = self.circlefit(self.f_data[self._fid],self.z_data[self._fid],fr,Ql,refine_results=False,calc_errors=True)
    self.z_data_sim = A2*(self.f_data-frcal)+self._S21_notch(self.f_data,fr=self.fitresults["fr"],Ql=self.fitresults["Ql"],Qc=self.fitresults["absQc"],phi=self.fitresults["phi0"],a=amp_norm,alpha=alpha,delay=delay)
    self.z_data_sim_norm = self._S21_notch(self.f_data,fr=self.fitresults["fr"],Ql=self.fitresults["Ql"],Qc=self.fitresults["absQc"],phi=self.fitresults["phi0"],a=1.0,alpha=0.,delay=0.)
    self._delay = delay
circuit.notch_port.autofit = my_autofit

def my_do_calibration(self,f_data,z_data,ignoreslope=True,guessdelay=True,fixed_delay=None, Ql_guess=None, fr_guess=None, pass_delay_guess=0, lwr_tau_bound=-np.inf):
    '''
    performs an automated calibration and tries to determine the prefactors a, alpha, delay
    fr, Ql, and a possible slope are extra information, which can be used as start parameters for subsequent fits
    see also "do_normalization"
    the calibration procedure works for transmission line resonators as well
    '''
    delay, params = self.get_delay(f_data,z_data,ignoreslope=ignoreslope,guess=guessdelay,delay=fixed_delay, pass_delay_guess=pass_delay_guess, lwr_tau_bound=lwr_tau_bound)
    z_data = (z_data-params[1]*(f_data-params[4]))*np.exp(2.*1j*np.pi*delay*f_data)
    xc, yc, r0 = self._fit_circle(z_data)
    zc = complex(xc,yc)
    if Ql_guess is None: Ql_guess=np.absolute(params[5])
    if fr_guess is None: fr_guess=params[4]
    fitparams = self._phase_fit(f_data,self._center(z_data,zc),0.,Ql_guess,fr_guess)
    theta, Ql, fr = fitparams
    beta = self._periodic_boundary(theta+np.pi,np.pi)
    offrespoint = complex((xc+r0*np.cos(beta)),(yc+r0*np.sin(beta)))
    alpha = np.angle(offrespoint)
    a = np.absolute(offrespoint)
    return delay, a, alpha, fr, Ql, params[1], params[4]
circuit.notch_port.do_calibration = my_do_calibration