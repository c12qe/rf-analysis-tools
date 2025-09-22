"""
This module contains models and fitters for resonators that are operated in transmission.

Fitting resonators in this configuration is more complicated than in the other configurations because the
off-resonance data goes to 0 instead of 1, while the on-resonance data goes to a value that depends on the losses.
Because there is no fixed reference point, more information must be provided in order to successfully fit the data. The
existing models are thus less-developed than those for the other configurations. The current limitations are
- the existing fitters all use hardcoded background models;
- the existing models assume that both ports have equal coupling losses;
- the Kerr nonlinear models are not yet implemented;
- the example notebooks have not been created yet.

If you need to fit resonators in this configuration, ask!
"""
from __future__ import absolute_import, division, print_function

import numpy as np

from . import background, base, guess, linear


class AbstractSymmetricTransmission(base.ResonatorModel):
    """
    This class models a resonator operated in transmission. It assumes that two ports have equal coupling losses (or,
    equivalently, equal coupling quality factors).
    """

    # This is the peak value of the transmission on resonance when the internal loss is zero.
    reference_point = 0.5 + 0j

    # ToDo: verify
    io_coupling_coefficient = 1


# Linear models and fitters

class LinearSymmetricTransmission(AbstractSymmetricTransmission):
    """
    This class models a linear resonator operated in transmission where the two ports have equal coupling losses (or,
    equivalently, equal coupling quality factors).

    The model parameters are the resonance frequency, the internal loss (defined as the inverse of the internal quality
    factor), and the coupling loss (defined as the sum of the inverses of the equal coupling quality factors). The
    total / loaded / resonator quality factor is
      Q = 1 / (internal_loss + coupling_loss).
    """

    def __init__(self, *args, **kwargs):
        """
        :param args: arguments passed directly to lmfit.model.Model.__init__().
        :param kwds: keywords passed directly to lmfit.model.Model.__init__().
        """

        def symmetric_transmission(frequency, resonance_frequency, coupling_loss, internal_loss):
            detuning = frequency / resonance_frequency - 1
            return 1 / (1 + (internal_loss + 2j * detuning) / coupling_loss)

        super(LinearSymmetricTransmission, self).__init__(func=symmetric_transmission, *args, **kwargs)

    # ToDo: implement and test guess.guess_smooth
    def guess(self, data, frequency=None, coupling_loss=None):
        """
        Return a lmfit.Parameters object containing reasonable initial values generated from the given data.

        :param data: an array of complex transmission data.
        :param frequency: an array of real frequencies at which the data was measured.
        :param coupling_loss: if not None, the coupling loss is set to the given value and is not varied in the fit.
        :return: lmfit.Parameters
        """
        params = self.make_params()
        smoothed_magnitude = guess.smooth(np.abs(data))
        peak_index = np.argmax(smoothed_magnitude)
        resonance_frequency_guess = frequency[peak_index]  # guess that the resonance is the highest point
        params['resonance_frequency'].set(value=resonance_frequency_guess, min=frequency.min(), max=frequency.max())
        power_minus_half_max = smoothed_magnitude ** 2 - smoothed_magnitude[peak_index] ** 2 / 2
        f1 = np.interp(0, power_minus_half_max[:peak_index], frequency[:peak_index])
        f2 = np.interp(0, -power_minus_half_max[peak_index:], frequency[peak_index:])
        linewidth = f2 - f1
        internal_plus_coupling = linewidth / resonance_frequency_guess
        internal_over_coupling = (1 / np.abs(data[peak_index]) - 1)
        if coupling_loss is None:
            params['coupling_loss'].set(value=internal_plus_coupling / (1 + internal_over_coupling),
                                        min=1e-12, max=1)
            params['internal_loss'].set(value=(internal_plus_coupling * internal_over_coupling /
                                               (1 + internal_over_coupling)),
                                        min=1e-12, max=1)
        else:
            params['coupling_loss'].set(value=coupling_loss, vary=False)
            params['internal_loss'].set(value=internal_plus_coupling - coupling_loss, min=1e-12, max=1)
        return params


class CCxSTFitterKnownMagnitude(linear.LinearResonatorFitter):
    """
    This class fits a composite model that is the product of the ComplexConstant background model and the
    SymmetricTransmission model.

    It should be used when the magnitude of the background response is known and the cable delay has been calibrated so
    that the background phase is constant across the band, but it will fit for a constant phase offset.
    """

    def __init__(self, frequency, data, background_magnitude, errors=None, **fit_kwds):
        """
        Fit the given data.

        :param frequency: an array of real frequencies at which the data was measured.
        :param data: an array of complex transmission data.
        :param background_magnitude: the value of the transmission in the absence of the resonator, in the same units
          as the data meaning NOT in dB.
        :param errors: an array of complex numbers that are the standard errors of the mean of the data points; the
          errors for the real and imaginary parts may be different; if no errors are provided then all points will be
          weighted equally.
        :param fit_kwds: keyword arguments passed directly to lmfit.model.Model.fit().
        """
        self.background_magnitude = background_magnitude
        super(CCxSTFitterKnownMagnitude, self).__init__(frequency=frequency, data=data,
                                                        foreground_model=LinearSymmetricTransmission(),
                                                        background_model=background.MagnitudePhase(),
                                                        errors=errors, **fit_kwds)

    def guess(self, frequency, data):
        phase_guess = np.angle(data[np.argmax(np.abs(data))])
        params = self.background_model.make_params(magnitude=self.background_magnitude, phase=phase_guess)
        params['magnitude'].vary = False
        background_values = self.background_model.eval(params=params, frequency=frequency)
        params.update(self.foreground_model.guess(data=data / background_values, frequency=frequency))
        return params


class CCxSTFitterKnownCoupling(linear.LinearResonatorFitter):
    """
    This class fits a composite model that is the product of the ComplexConstant background model and the
    SymmetricTransmission model.

    It should be used when the the coupling loss (i.e. the inverse coupling quality factor) is known, presumably from
    another measurement or a simulation, and when the cable delay has been calibrated so that the background phase is
    constant across the band.
    """

    def __init__(self, frequency, data, coupling_loss, errors=None, **fit_kwds):
        """
        Fit the given data to a composite model that is the product of the ComplexConstant background model and the
        SymmetricTransmission model.

        :param frequency: an array of real frequencies at which the data was measured.
        :param data: an array of complex transmission data (NOT in dB).
        :param coupling_loss: the fixed value of the coupling loss, or inverse coupling quality factor.
        :param errors: an array of complex numbers that are the standard errors of the mean of the data points; the
          errors for the real and imaginary parts may be different; if no errors are provided then all points will be
          weighted equally.
        :param fit_kwds: keyword arguments passed directly to lmfit.model.Model.fit().
        """
        self.known_coupling_loss = coupling_loss
        super(CCxSTFitterKnownCoupling, self).__init__(frequency=frequency, data=data,
                                                       foreground_model=LinearSymmetricTransmission(),
                                                       background_model=background.MagnitudePhase(),
                                                       errors=errors, **fit_kwds)

    def guess(self, frequency, data):
        params = self.background_model.guess(data=self.data, frequency=self.frequency)
        params.update(self.foreground_model.guess(data=(data /
                                                        self.background_model.eval(params=params, frequency=frequency)),
                                                  frequency=frequency, coupling_loss=self.known_coupling_loss))
        return params


class LinearSymmetricTransmissionLeakage(AbstractSymmetricTransmission):
    """
    This class models a linear resonator operated in transmission where the two ports have equal coupling losses (or,
    equivalently, equal coupling quality factors), with additional leakage directly from the input port to the output
    port.

    The model parameters are the resonance frequency, the internal loss (defined as the inverse of the internal quality
    factor), and the coupling loss (defined as the sum of the inverses of the equal coupling quality factors), along
    with an additional complex number for the leakage transmission.

    The total / loaded / resonator quality factor is
      Q = 1 / (internal_loss + coupling_loss).
    """

    def __init__(self, *args, **kwargs):
        """
        :param args: arguments passed directly to lmfit.model.Model.__init__().
        :param kwds: keywords passed directly to lmfit.model.Model.__init__().
        """

        def symmetric_transmission_leakage(frequency, resonance_frequency, coupling_loss, internal_loss,
                                           off_resonance_real, off_resonance_imag):
            detuning = frequency / resonance_frequency - 1
            return 1 / (1 + (
                        internal_loss + 2j * detuning) / coupling_loss) + off_resonance_real + 1j * off_resonance_imag

        super(LinearSymmetricTransmissionLeakage, self).__init__(func=symmetric_transmission_leakage, *args, **kwargs)

    def guess(self, data, frequency=None, coupling_loss=None):
        params = self.make_params()
        smooth_data = guess.smooth(data)
        resonance_frequency = np.median(frequency[guess.largest(guess.distances(smooth_data), fraction=0.1)])
        params['resonance_frequency'].set(value=resonance_frequency, min=frequency.min(), max=frequency.max())
        resonance_index = np.argmin(np.abs(frequency - resonance_frequency))
        linewidth = abs(frequency[np.argmin(smooth_data.imag)] - frequency[np.argmax(smooth_data.imag)])
        internal_plus_coupling = linewidth / resonance_frequency
        internal_over_coupling = (1 / np.abs(data[resonance_index]) - 1)
        if coupling_loss is None:
            params['coupling_loss'].set(value=internal_plus_coupling / (1 + internal_over_coupling),
                                        min=1e-12, max=1)
            params['internal_loss'].set(value=(internal_plus_coupling * internal_over_coupling /
                                               (1 + internal_over_coupling)),
                                        min=1e-12, max=1)
        else:
            params['coupling_loss'].set(value=coupling_loss, vary=False)
            params['internal_loss'].set(value=internal_plus_coupling - coupling_loss, min=1e-12, max=1)
        off_resonance = np.median(data[guess.smallest(guess.distances(smooth_data), fraction=0.1)])
        params['off_resonance_real'].set(value=off_resonance.real)
        params['off_resonance_imag'].set(value=off_resonance.imag)
        return params


class MPxLSTLFitterKnownMagnitude(linear.LinearResonatorFitter):
    """
    This class fits a composite model that is the product of the MagnitudePhase background model and the
    LinearSymmetricTransmissionLeakage model.

    It should be used when the magnitude of the background response is known and the cable delay has been calibrated so
    that the background phase is constant across the band, but it will fit for a constant phase offset.
    """

    def __init__(self, frequency, data, background_magnitude, errors=None, **fit_kwds):
        """Fit the given data.

        :param numpy.ndarray[float] frequency: an array of real frequencies at which the data was measured.
        :param numpy.ndarray[complex] data: an array of complex transmission data.
        :param complex background_magnitude: the value of the transmission in the absence of the resonator, in the same
                                             units as the data meaning NOT in dB.
        :param errors: an array of complex numbers that are the standard errors of the mean of the data points; the
                       errors for the real and imaginary parts may be different; if no errors are provided then all
                       points will be weighted equally.
        :type errors: numpy.ndarray[complex] or None
        :param fit_kwds: keyword arguments passed directly to lmfit.model.Model.fit().
        """
        self.background_magnitude = background_magnitude
        super(MPxLSTLFitterKnownMagnitude, self).__init__(frequency=frequency, data=data,
                                                          foreground_model=LinearSymmetricTransmissionLeakage(),
                                                          background_model=background.MagnitudePhase(),
                                                          errors=errors, **fit_kwds)

    def guess(self, frequency, data):
        phase_guess = np.angle(data[np.argmax(np.abs(data))])
        params = self.background_model.make_params(magnitude=self.background_magnitude, phase=phase_guess)
        params['magnitude'].vary = False
        background_values = self.background_model.eval(params=params, frequency=frequency)
        params.update(self.foreground_model.guess(data=data / background_values, frequency=frequency))
        return params

