__all__ = ['ErrorGenerator']

import logging
from typing import List
import numpy as np
import scipy.stats as stats
from ocelot import Element, Quadrupole, Sextupole, Octupole, SBend

logger = logging.getLogger(__name__)


class ErrorGenerator:
    """
    Utility class to generate perturbations (errors) of selected elements
    """

    def __init__(self,
                 elements: List[Element] = None,
                 abse: float = None,
                 fse: float = None,
                 dx: float = None,
                 dy: float = None,
                 dtilt: float = None,
                 cutoff_sigma: int = 2):
        self.elements = elements
        self.abse = abse
        self.fse = fse
        self.dx = dx
        self.dy = dy
        self.dtilt = dtilt
        self.cutoff = cutoff_sigma

    ref_params = {SBend: 'fse', Quadrupole: 'k1', Sextupole: 'k2', Octupole: 'k3'}

    @property
    def summary(self):
        return f'EG: abs:{self.abse:.3e} rel:{self.fse:.3e} x:{self.dx:.3e} y:{self.dy:.3e} dtilt:{self.dtilt:.3e}'

    def __repr__(self):
        return self.summary

    def add_errors(self,
                   elements: List[Element] = None,
                   abse: float = None,
                   fse: float = None,
                   dx: float = None,
                   dy: float = None,
                   dtilt: float = None,
                   ref_str: List[float] = None,
                   param_dict=None):
        """
        Adds normally distributed error to specified elements and parameters. By default, saves previous values
        of strength to reference parameters if they don't already exist.
        Can be used standalone, but intended for repeated calls without arguments to keep generating new seeds
        :param elements: Elements to apply errors to
        :param abse: Absolute error on strength
        :param fse: Relative error on strength
        :param dx: x
        :param dy: y
        :param dtilt: tilt
        :param ref_str: List of same length as elements with reference strengths
        :param param_dict:
        :return: 2D array of generated errors in parameter order
        """
        elements = elements or self.elements
        abse = abse or self.abse or 0.0
        fse = fse or self.fse or 0.0
        dx = dx or self.dx or 0.0
        dy = dy or self.dy or 0.0
        dtilt = dtilt or self.dtilt or 0.0

        for v in [abse, fse, dx, dy, dtilt]:
            assert v >= 0.0

        if ref_str:
            assert len(ref_str) == len(elements)

        # Mapping of type and strength attributes - dipole k1 needs more careful treatment
        param_dict = param_dict or ErrorGenerator.ref_params

        N = len(elements)
        errors = np.zeros((N, 5))
        errors[:, 0] = stats.truncnorm.rvs(-self.cutoff, self.cutoff, loc=0.0, scale=fse, size=N)
        errors[:, 1] = stats.truncnorm.rvs(-self.cutoff, self.cutoff, loc=0.0, scale=abse, size=N)
        for j, (mu, sigma) in enumerate(zip([0.0, 0.0, 0.0], (dx, dy, dtilt))):
            errors[:, j+2] = stats.truncnorm.rvs(-self.cutoff, self.cutoff, loc=mu, scale=sigma, size=N)

        for i, el in enumerate(elements):
            # Per element type errors
            param = param_dict[el.__class__]
            if not hasattr(el, param):
                logger.warning(f'Element {el.id}|{el.__class__.__name__} has no error attribute {param}')
                continue
            param_ref = param + '_ref'
            if ref_str:
                ref_k = ref_str[i]
            else:
                if hasattr(el, param_ref):
                    ref_k = getattr(el, param_ref)
                else:
                    ref_k = getattr(el, param)
                    setattr(el, param_ref, ref_k)

            v = errors[i, 0]  # fse
            v2 = errors[i, 1]  # abse
            #if v > 0.0 and v2 > 0.0:
            setattr(el, param, ref_k + ref_k * v + v2)

            # These are shared for all elements, and are assumed to have 0.0 mean0
            if dx != 0.0 or dy != 0.0 or dtilt != 0.0:
                for j, param in enumerate(['dx', 'dy', 'dtilt']):
                    v = errors[i, j+2]
                    setattr(el, param, v)

        return errors

    def reset(self, elements):
        """
        Reset elements to reference values for any elements which have them
        :param elements:
        :return:
        """
        elements = elements or self.elements
        for el in elements:
            p_str = ErrorGenerator.ref_params[el.__class__]
            for param in ['dx', 'dy', 'dtilt', p_str]:
                param_ref = param + '_ref'
                if hasattr(el, param_ref):
                    setattr(el, param, getattr(el, param_ref))
