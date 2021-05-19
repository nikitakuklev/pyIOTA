__all__ = ['Writer']

import datetime
import logging
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace
from typing import Dict

import numpy as np
from ocelot import Sextupole, Hcor, Vcor, Element, Edge, Marker, Octupole, Matrix, \
    Quadrupole, SBend, Drift, Solenoid, Cavity, Monitor, Multipole
from ..lattice.elements import LatticeContainer, NLLens, HKPoly, ILMatrix, Recirculator
from .. import __version__ as pyiotaver

logger = logging.getLogger(__name__)


class Writer:
    def __init__(self, options: Dict):
        self.options = options or {}
        self.iota = None
        self.verbose = False

    def set_options(self, opt: Dict) -> None:
        """
        Replace options dict with new one
        """
        self.options = opt

    # def _check_critical_options(self):
    #     """
    #     Verify options consistency
    #     """
    #     assert self.iota
    #     assert self.options['sr'] in [False, True]
    #     assert self.options['isr'] in [False, True]
    #     for k in ['dip_kicks', 'quad_kicks', 'sext_kicks', 'oct_kicks']:
    #         assert isinstance(self.options[k], int) and 1 <= self.options[k] < 128
    #     if 'aperture_scale' not in self.options:
    #         self.options['aperture_scale'] = 1

    def import_from_madx(self, path):
        try:
            sys.path.insert(1, Path.home().as_posix())
            import pymadx
        except ImportError:
            raise Exception('This class requires pymadx library!')
        self.iota = pymadx.Data.Tfs(path)

    def generate_lattice(self):
        pass

    def generate_header(self, box: LatticeContainer) -> str:
        """
        Generate lattice file header
        :return: Header string
        """
        sl = []
        sl.append(f'!This file is auto-generated by pyIOTA (github.com/nikitakuklev/pyIOTA)\n')
        sl.append(f'!Send bug reports to nkuklev@uchicago.edu\n')
        sl.append(f"!Module: {Writer.__module__ + '.' + Writer.__qualname__} v{pyiotaver}\n")
        sl.append(f'!Source file: {box.source_file}\n')
        sl.append(f'!Lattice title: {box.name}\n')
        sl.append(f'!Origin: {box.source}\n\n')
        sl.append(f'!Time: {datetime.datetime.now().isoformat()}\n')
        return ''.join(sl)

    # def write_lattice(self, path) -> None:
    #     """
    #     Writes current lattice to the specified path. DEPRECATED.
    #     :param path: Full file path
    #     """
    #     self._check_critical_options()
    #     iota = self.iota
    #     opt = SimpleNamespace(**self.options)
    #     sl = []
    #
    #     dipoles = iota.GetElementsOfType('SBEND')
    #     quads = iota.GetElementsOfType('QUADRUPOLE')
    #     sextupoles = iota.GetElementsOfType('SEXTUPOLE')
    #     octupoles = iota.GetElementsOfType('OCTUPOLE')
    #     drifts = iota.GetElementsOfType('DRIFT')
    #     markers = iota.GetElementsOfType('MARKER')
    #     solenoids = iota.GetElementsOfType('SOLENOID')
    #     vkickers = iota.GetElementsOfType('VKICKER')
    #     cavities = iota.GetElementsOfType('RFCAVITY')
    #     monitors = iota.GetElementsOfType('MONITOR')
    #     nllenses = iota.GetElementsOfType('NLLENS')
    #     header = iota.header
    #
    #     # Preamble
    #     sl.append(self.generate_header())
    #     sl.append('\n')
    #     sl.append(f'% {opt.sr} sto flag_synch\n')
    #     sl.append(f'% {opt.isr} sto flag_isr\n')
    #     sl.append(f'% {opt.dip_kicks} sto dip_kicks\n')
    #     sl.append(f'% {opt.quad_kicks} sto quad_kicks\n')
    #     sl.append(f'% {opt.sext_kicks} sto sext_kicks\n')
    #     sl.append(f'% {opt.oct_kicks} sto oct_kicks\n')
    #     sl.append('\n')
    #
    #     # Set charge
    #     sl.append(f'CHRG: CHARGE, TOTAL={header["NPART"] * 1.602176634e-19:e}\n')
    #
    #     sl.append('!MAIN BENDS\n')
    #     for el in dipoles:
    #         idx = iota.IndexFromName(el['UNIQUENAME'])
    #         de1 = iota[idx - 1]
    #         de2 = iota[idx + 1]
    #
    #         sl.append(f"{el['UNIQUENAME']}: CSBEND, l={el['L']:.10f}, angle={el['ANGLE']}, e1=0, e2=0, &\n")
    #         sl.append(
    #             f"               h1={de1['H1']:+.10f}, h2={de2['H1']:+.10f}, hgap={de1['HGAP']}, fint={de1['FINT']}, &\n")
    #         sl.append(f"               EDGE_ORDER=1, EDGE1_EFFECTS=3, EDGE2_EFFECTS=3, &\n")
    #         # f.write('               edge_order=2, edge1_effects=2, edge2_effects=2, fse_correction=1, &\n') # Forum post says should work ok? but doesnt
    #         sl.append(f'                N_KICKS="dip_kicks", ISR="flag_isr", SYNCH_RAD="flag_synch"\n')
    #     sl.append('\n')
    #
    #     sl.append('!MAIN QUADS\n')
    #     for el in [q for q in quads if q['UNIQUENAME'].startswith('Q')]:
    #         if el['UNIQUENAME'] in ['QC1R', 'QC2R', 'QC3R', 'QC1L', 'QC2L', 'QC3L']:
    #             sl.append(
    #                 f'{el["UNIQUENAME"]:<6}: KQUAD, l={el["L"]:.10f}, k1={el["K1L"] / el["L"]:+.10e}, N_KICKS="quad_kicks", ISR="flag_isr", SYNCH_RAD="flag_synch", tilt=0.0\n')
    #         else:
    #             sl.append(
    #                 f'{el["UNIQUENAME"]:<6}: KQUAD, l={el["L"]:.10f}, k1={el["K1L"] / el["L"]:+.10e}, N_KICKS="quad_kicks", ISR="flag_isr", SYNCH_RAD="flag_synch"\n')
    #     sl.append('\n')
    #
    #     sl.append('!COMBINED HV CORRECTORS+SKEW QUADS\n')
    #     for el in [q for q in quads if not q['UNIQUENAME'].startswith('Q')]:
    #         if el['UNIQUENAME'] in ['QC1R', 'QC2R', 'QC3R']:
    #             sl.append(
    #                 f'{el["UNIQUENAME"]:<6}: KQUAD, l={el["L"]:.10f}, k1={el["K1L"] / el["L"]:+.10e}, N_KICKS="quad_kicks", ISR="flag_isr", SYNCH_RAD="flag_synch"\n')
    #         else:
    #             sl.append(
    #                 f'{el["UNIQUENAME"]:<6}: KQUAD, l={el["L"]:.10f}, k1={el["K1L"] / el["L"]:+.10e}, N_KICKS="quad_kicks", ISR="flag_isr", SYNCH_RAD="flag_synch"\n')
    #     sl.append('\n')
    #
    #     sl.append('!SEXTUPOLES\n')
    #     for el in sextupoles:
    #         sl.append(
    #             f'{el["UNIQUENAME"]:<10}: KSEXT, l={el["L"]}, k2={el["K2L"] / el["L"]:.10e}, N_KICKS="sext_kicks", ISR="flag_isr", SYNCH_RAD="flag_synch"\n')
    #     sl.append('\n')
    #
    #     sl.append('! Nonlinear quasi-integrable insert - strengths set with param file later\n')
    #     for el in octupoles:
    #         # f.write('{:<10}: KOCT,l={},k3={},N_KICKS="oct_kicks"\n'.format(el['UNIQUENAME'], el['L'], el['K3L']/el['L']))
    #         sl.append(
    #             f'{el["UNIQUENAME"]:<10}: KOCT, l={el["L"]}, k3={0:.10e}, N_KICKS="oct_kicks", ISR="flag_isr", SYNCH_RAD="flag_synch"\n')
    #     sl.append('\n')
    #
    #     sl.append('!SOLENOIDS\n')
    #     for el in solenoids:
    #         sl.append(f'{el["UNIQUENAME"]:<10}: EDRIFT, l={el["L"]}\n')
    #     sl.append('\n')
    #
    #     sl.append('!NL MAGNET\n')
    #     for el in nllenses:
    #         sl.append('{:<10}: EDRIFT, l={}\n'.format(el['UNIQUENAME'], el['L']))
    #     sl.append('\n')
    #
    #     sl.append('!VKICKERS\n')
    #     for el in vkickers:
    #         # f.write('{:<10}: EVKICK,l={},kick={}\n'.format(el['UNIQUENAME'], el['L'], 0))
    #         sl.append('{:<10}: EDRIFT, l={}\n'.format(el['UNIQUENAME'], el['L']))
    #     sl.append('\n')
    #
    #     sl.append('! Voltages set in task file\n')
    #     for el in cavities:
    #         # f.write('{:<10}: rfca,l={},volt={},change_t=1\n'.format(el['UNIQUENAME'], el['L'], el['VOLT']*1e6))
    #         sl.append('{:<10}: RFCA, l={}, volt=0.0, change_t=1\n'.format(el['UNIQUENAME'], el['L']))
    #     sl.append('\n')
    #
    #     sl.append('!MONITORS\n')
    #     for el in monitors:
    #         sl.append('{:<10}: MONI, l={}\n'.format(el['UNIQUENAME'], el['L']))
    #     sl.append('\n')
    #
    #     sl.append('!DRIFTS\n')
    #     for el in drifts:
    #         sl.append('{:<10}: EDRIFT, l={}\n'.format(el['UNIQUENAME'], el['L']))
    #     sl.append('\n')
    #
    #     names = []
    #     for el in markers:
    #         # name = pick_next_name(el['UNIQUENAME'], names)
    #         name = el['UNIQUENAME']
    #         sl.append('{:<10}: MARKER, FITPOINT=1\n'.format(name))
    #         # print(el['INDEX'],el['UNIQUENAME'],name)
    #         # iota.RenameElement(el['INDEX'], name)
    #     sl.append('\n')
    #
    #     # Finishing touches
    #     # f.write('IBSELE: IBSCATTER, VERBOSE=0\n')
    #     sl.append('!WATCHPOINTS\n')
    #     sl.append('!Now inserted from task file, to control filenames/splitting\n')
    #     # f.write('!W0: WATCH, MODE="coordinate", INTERVAL=1, DISABLE=0, FILENAME="%s_w0.track"\n')
    #     # f.write('!W1: WATCH, MODE="parameter", INTERVAL=1, DISABLE=0, FILENAME="%s_w1.track"\n')
    #     sl.append('\n')
    #     sl.append('!APERTURES\n')
    #     sl.append('!Default escape criterion very large (10m), and there are timing issues with CLEAN element\n')
    #     sl.append('!It needs change_t, which doesnt work for 4D tracking (when RF has not been setup)\n')
    #     sl.append('!For now, globally limit transverse aperture, since our beampipe is <=2in\n')
    #     sl.append('!Also add worst restriction point in ring (at IOR)\n')
    #     sl.append('!Will have to cut bucket escapes in post-processing\n')
    #     sl.append('!C0: CLEAN, DELTALIMIT=0.1\n')
    #     # f.write('MA1: MAXAMP, X_MAX=0.025, Y_MAX=0.025, ELLIPTICAL=1 \n')
    #     # f.write('MA1: MAXAMP, X_MAX=0.030, Y_MAX=0.030, ELLIPTICAL=1 \n')
    #     # f.write('APER: RCOL, X_MAX=0.007, Y_MAX=0.007 \n')
    #     # f.write('MA1: MAXAMP, X_MAX=0.050, Y_MAX=0.050, ELLIPTICAL=1 \n')
    #     sl.append('MA1: MAXAMP, X_MAX=0.0381, Y_MAX=0.0381, ELLIPTICAL=1 \n')  # 1.5x aperture
    #     # f.write('APER: ECOL, X_MAX=0.008, Y_MAX=0.008 \n')
    #     # sl.append('APER: ECOL, X_MAX=0.005925, Y_MAX=0.00789 \n')  # 1.5x actual NL aperture
    #     if opt.aperture_scale != 1:
    #         print(f'Lattice aperture scaled by {opt.aperture_scale}')
    #         sl.append(
    #             f'APER: ECOL, X_MAX={3.9446881e-3 * opt.aperture_scale}, Y_MAX={5.25958413e-3 * opt.aperture_scale} \n')
    #     else:
    #         sl.append('APER: ECOL, X_MAX=3.9446881e-3, Y_MAX=5.25958413e-3 \n')  # 1x actual NL aperture
    #     # f.write('W1: WATCH, MODE="coordinate", INTERVAL=1, FILENAME="%s_w1.track"')
    #     sl.append('\n')
    #
    #     sl.append('RC: RECIRC \n')
    #     sl.append('MAL: MALIGN \n')
    #     sl.append('\n')
    #
    #     sl.append('!Full line has markers in pairs, due to bug? feature? where you cant access fitpoint \n')
    #     sl.append('!properties if only single marker instance is present, need 2 or more \n')
    #
    #     seq = list(iota.sequence)
    #     markerdict = markers.data
    #     # print(seq,markerdict)
    #     dipedge_names = [el['UNIQUENAME'] for el in iota.GetElementsOfType('DIPEDGE')]
    #     seq2 = []
    #     for s in seq:
    #         if s not in dipedge_names:
    #             seq2.append(s)
    #             if s in markerdict:
    #                 seq2.append(s)
    #                 # print(s)
    #     wrapstr = 'iota: LINE=(CHRG, MAL, RC, MA1, APER, {})\n'.format(', '.join(seq2))
    #     sl.append(textwrap.fill(wrapstr, break_long_words=False, break_on_hyphens=False).replace('\n', ' &\n'))
    #     sl.append('\n\n')
    #     sl.append('!Short version has no monitors or markers, from elegant forum might make parallel version faster\n')
    #     seq = list(iota.sequence)
    #     dipedge_names = [el['UNIQUENAME'] for el in iota.GetElementsOfType('DIPEDGE')] + \
    #                     [el['UNIQUENAME'] for el in monitors] + \
    #                     [el['UNIQUENAME'] for el in markers]
    #     seq2 = []
    #     for s in seq:
    #         if s not in dipedge_names:
    #             seq2.append(s)
    #     wrapstr = 'iota_short: LINE=(CHRG, MAL, RC, MA1, APER, {})\n'.format(', '.join(seq2))
    #     sl.append(textwrap.fill(wrapstr, break_long_words=False, break_on_hyphens=False).replace('\n', ' &\n'))
    #     sl.append('\n')
    #
    #     with open(path, 'w', newline='\n') as f:
    #         f.write(''.join(sl))

    def _check_critical_options_ng(self):
        """
        Verify options consistency
        """
        assert self.options['sr'] in [False, True]
        assert self.options['isr'] in [False, True]
        for k in ['dip_kicks', 'quad_kicks', 'sext_kicks', 'oct_kicks']:
            assert isinstance(self.options[k], int) and 1 <= self.options[k] <= 128

        if 'global_aperture' not in self.options:
            self.options['global_aperture'] = False
        else:
            assert isinstance(self.options['global_aperture'], (int, float))

        if 'limiting_aperture' not in self.options:
            self.options['limiting_aperture'] = False
        else:
            assert isinstance(self.options['limiting_aperture'], (int, float))

        if 'add_mal' not in self.options:
            self.options['add_mal'] = False
        else:
            assert isinstance(self.options['add_mal'], bool)
        # if 'aperture_scale' not in self.options:
        #     self.options['aperture_scale'] = 1

    def _validate_element(self, el: Element, elements: list):
        if len(el.id) >= 99:
            idx = el.id[:99]
            logger.warning(f'Name ({el.id}) is ({len(el.id)}) chars, truncating to 99')
            idx = el.id[:99]
            raise Exception  # temp
        else:
            idx = el.id

        if idx in elements[0]:
            if self.verbose:
                print(f'Found repeat definition: ({idx})')
            if not isinstance(el, Drift):
                raise AttributeError(f'Repeat non-drift ({idx}) ({el.__class__.__name__})')
            else:
                e_orig = elements[1][elements[0].index(idx)]
                if el.l != e_orig.l:
                    raise AttributeError(f'Repeat drift ({idx}) has different lengths {el.l}|{e_orig.l}')
                return True
        else:
            elements[0].append(idx)
            elements[1].append(el)
            return False

    def _check_element_names(self, box: LatticeContainer):
        names = [el.id for el in box.lattice.sequence]
        n_uniques = len(set(names))
        if any(len(idx) >= 99 for idx in names):
            raise Exception  # temp
            names = [idx[:99] for idx in names]
            n_uniques2 = len(set(names))
            if n_uniques != n_uniques2:
                raise Exception(f'Concatenation to 99 characters created ID conflicts - aborting')

    def write_lattice_ng(self,
                         fpath: Path,
                         box: LatticeContainer,
                         debug: bool = True,
                         save: bool = False,
                         # add_limiting_aperture: bool = True,
                         # add_misalignment_el: bool = True
                         ):
        """
        Writes current lattice to the specified path.
        :param fpath: Full file path
        """
        if isinstance(fpath, str):
            raise Exception('Paths should be using Pathlib')
        self._check_critical_options_ng()
        self._check_element_names(box)
        # iota = self.iota
        lat = box.lattice
        opt = SimpleNamespace(**self.options)
        if debug:
            logger.info(opt)
        sl = []
        elements = [[], []]

        # Get a view that removes elements whenever they are looked at
        view = box.get_onetimeview()

        dipoles = view.get_elements(SBend)
        edges = view.get_elements(Edge)
        quads = view.get_elements(Quadrupole)
        sextupoles = view.get_elements(Sextupole)
        octupoles = view.get_elements(Octupole)
        drifts = view.get_elements(Drift)
        solenoids = view.get_elements(Solenoid)
        # vkickers = iota.GetElementsOfType('VKICKER')
        cavities = view.get_elements(Cavity)
        monitors = view.get_elements(Monitor)
        markers = view.get_elements(Marker)
        nllenses = view.get_elements(NLLens)
        matrices = view.get_elements(Matrix)
        ilmatrices = view.get_elements(ILMatrix)
        recirculators = view.get_elements(Recirculator)
        multipoles = view.get_elements(Multipole)

        # The edges in sequences are ocelot-based, and so can be ignored
        if len(dipoles) != len(edges) // 2:
            raise Exception(f'Element mismatch - have ({len(dipoles)}) dipoles and ({len(edges)}) edges')

        # Custom elegant elements
        hkpoly = view.get_elements(HKPoly)

        # Preamble
        sl.append(self.generate_header(box))
        sl.append('\n')

        sl.append(f'% {opt.sr} sto flag_synch\n')
        sl.append(f'% {opt.isr} sto flag_isr\n')
        sl.append(f'% {opt.dip_kicks} sto dip_kicks\n')
        sl.append(f'% {opt.quad_kicks} sto quad_kicks\n')
        sl.append(f'% {opt.sext_kicks} sto sext_kicks\n')
        sl.append(f'% {opt.oct_kicks} sto oct_kicks\n')
        sl.append('\n')

        # Set charge
        # sl.append(f'CHRG: CHARGE, TOTAL={header["NPART"] * 1.602176634e-19:e}\n')
        # sl.append(f'CHRG: CHARGE, TOTAL={lat.:e}\n')

        def ma(el):
            if el.dx != 0.0 or el.dy != 0.0 or (el.tilt + el.dtilt) != 0.0:
                return f' dx={el.dx:+.10e}, dy={el.dy:+.10e}, tilt={el.tilt + el.dtilt:+.10e},'
            else:
                return ''

        if dipoles:
            sl.append('!DIPOLES\n')
            for el in dipoles:
                if self._validate_element(el, elements): continue
                sl.append(f"{el.id}: CSBEND, l={el.l:.10f}, angle={el.angle}, e1=0, e2=0, &\n")
                sl.append(
                    f" h1={el.h_pole1:+.10f}, h2={el.h_pole1:+.10f}, hgap={el.gap / 2:+.10f}, fint={el.fint}, &\n")
                ef = getattr(el, 'elegant_edge_effects', 3)
                edge_order = getattr(el, 'elegant_edge_order', 1)
                if ef == 2 or ef == 4:
                    fcorr = getattr(el, 'elegant_fse_correction', 1)
                    sl.append(f" FSE_CORRECTION = {fcorr}, &\n")
                sl.append(f" EDGE_ORDER={edge_order}, EDGE1_EFFECTS={ef}, EDGE2_EFFECTS={ef}, {ma(el)} &\n")
                sl.append(f' N_KICKS="dip_kicks", ISR="flag_isr", SYNCH_RAD="flag_synch"\n')
            sl.append('\n')

        main_quads = [q for q in quads if q.id.startswith('Q')]
        if main_quads:
            sl.append('!QUADS\n')
            for el in main_quads:
                if self._validate_element(el, elements): continue
                sl.append(f'{el.id:<6}: KQUAD, l={el.l:.10f}, k1={el.k1:+.10e},{ma(el)}'
                          f' N_KICKS="quad_kicks", ISR="flag_isr", SYNCH_RAD="flag_synch"\n')
            sl.append('\n')

        other_quads = [q for q in quads if q not in main_quads]
        if other_quads:
            sl.append('!COMBINED HV CORRECTORS+SKEW QUADS\n')
            for el in other_quads:
                if self._validate_element(el, elements): continue
                add_hcor = 0
                add_vcor = 0
                for c in box.correctors:
                    if c.ref_el is el:
                        if isinstance(c, Hcor):
                            add_hcor = 1
                        elif isinstance(c, Vcor):
                            add_vcor = 1
                        else:
                            continue
                sl.append(
                    f'{el.id:<6}: KQUAD, l={el.l:.10f}, k1={el.k1:+.10e},{ma(el)}'
                    f' HSTEERING={add_hcor}, VSTEERING={add_vcor},'
                    f' N_KICKS="quad_kicks", ISR="flag_isr", SYNCH_RAD="flag_synch"\n')
            sl.append('\n')

        if sextupoles:
            sl.append('!SEXTUPOLES\n')
            for el in sextupoles:
                if self._validate_element(el, elements): continue
                if el.l == 0.0:
                    raise Exception("Thin integrator element found - this will have no effect, use MULT or HKPOLY")
                sl.append(f'{el.id:<10}: KSEXT, l={el.l}, k2={el.k2:+.10e},{ma(el)}'
                          f' N_KICKS="sext_kicks", ISR="flag_isr", SYNCH_RAD="flag_synch"\n')
            sl.append('\n')

        if octupoles:
            sl.append('!OCTUPOLES + quasi-integrable insert\n')
            for el in octupoles:
                if self._validate_element(el, elements): continue
                if el.l == 0.0:
                    raise Exception("Thin integrator element found - this will have no effect, use MULT or HKPOLY")
                sl.append(
                    f'{el.id:<10}: KOCT, l={el.l}, k3={el.k3:.10e},{ma(el)}'
                    f' N_KICKS="oct_kicks", ISR="flag_isr", SYNCH_RAD="flag_synch"\n')
            sl.append('\n')

        if multipoles:
            sl.append('!MULTIPOLES\n')
            for el in multipoles:
                kn = np.array(el.kn)
                if np.sum(kn != 0.0) > 1:
                    raise Exception('Elegant MULT only supports a single multipole order - use HKPOLY')
                order = np.argmax(kn > 0.0)
                assert order >= 2
                if self._validate_element(el, elements): continue
                if getattr(el, 'extra_args', None):
                    n_kicks = el.extra_args.get('N_KICKS', 1)
                else:
                    n_kicks = 1
                sl.append(
                    f'{el.id:<10}: MULT, l={el.l}, knl={el.kn[order]:.10e}, order={order}, {ma(el)}'
                    f' N_KICKS={n_kicks}, SYNCH_RAD="flag_synch"\n')
            sl.append('\n')

        if solenoids:
            sl.append('!SOLENOIDS\n')
            for el in solenoids:
                if self._validate_element(el, elements): continue
                sl.append(f'{el.id:<10}: EDRIFT, l={el.l}\n')
            sl.append('\n')

        if nllenses:
            sl.append('!Danilov-Nagaitsev nonlinear magnet\n')
            for el in nllenses:
                if self._validate_element(el, elements): continue
                sl.append(f'{el.id:<10}: EDRIFT, l={el.l}\n')
            sl.append('\n')

        # sl.append('!VKICKERS\n')
        # for el in vkickers:
        #     # f.write('{:<10}: EVKICK,l={},kick={}\n'.format(el['UNIQUENAME'], el['L'], 0))
        #     sl.append('{:<10}: EDRIFT, l={}\n'.format(el['UNIQUENAME'], el['L']))
        # sl.append('\n')

        if cavities:
            sl.append('! Voltages set in task file\n')
            for el in cavities:
                if self._validate_element(el, elements): continue
                sl.append(f'{el.id:<10}: RFCA, l={el.l}, volt=0.0, change_t=1\n')
            sl.append('\n')

        if monitors:
            sl.append('!MONITORS\n')
            for el in monitors:
                if self._validate_element(el, elements): continue
                if getattr(el,'elegant_co_fitpoint',False):
                    sl.append(f'{el.id:<10}: MONI, l={el.l}, CO_FITPOINT=1\n')
                else:
                    sl.append(f'{el.id:<10}: MONI, l={el.l}\n')
            sl.append('\n')

        if markers:
            sl.append('!MARKERS/WPS\n')
            for el in markers:
                if self._validate_element(el, elements): continue
                if getattr(el, 'elegant_watchpoint', False):
                    # Special watchpoints
                    props_str = ''
                    for (k, v) in el.elegant_watchpoint_props.items():
                        props_str += f'{k} = {v}, '
                    sl.append(f'{el.id:<10}: WATCH, {props_str}\n')
                else:
                    sl.append(f'{el.id:<10}: MARKER\n')
            sl.append('\n')

        if drifts:
            sl.append('!DRIFTS\n')
            for el in drifts:
                if self._validate_element(el, elements): continue
                sl.append(f'{el.id:<10}: EDRIFT, l={el.l:.10e}\n')
            sl.append('\n')

        if matrices:
            sl.append('!MATRICES\n')
            for el in matrices:
                if self._validate_element(el, elements): continue
                if el.r.shape != (6, 6):
                    raise Exception(f'First order matrix shape is {el.r.shape}, not 6x6????')
                matrix_terms = []
                for i, j in np.ndindex(el.r.shape):
                    if el.r[i, j] != 0:
                        matrix_terms.append(f'R{i + 1}{j + 1}={el.r[i, j]}')
                sl.append(f'{el.id:<10}: EMATRIX, l={el.l}, C5={el.l}, ' + ', '.join(matrix_terms) + '\n')
            sl.append('\n')

        if hkpoly:
            sl.append('!HKPOLY (polynomial hamiltonian)\n')
            for el in hkpoly:
                if self._validate_element(el, elements): continue
                arg_string = [f'{k.upper()}={v}' for (k, v) in el.hkpoly_args.items()]
                sl.append(f"{el.id:<10}: HKPOLY, l={el.l}, {', '.join(arg_string)}\n")
            sl.append('\n')

        if ilmatrices:
            sl.append('!ILMATRICES \n')
            for el in ilmatrices:
                if self._validate_element(el, elements): continue
                arg_string = [f'{k.upper()}={v}' for (k, v) in el.extra_args.items()]
                sl.append(f"{el.id:<10}: ILMATRIX, l={el.l}, {', '.join(arg_string)}\n")
            sl.append('\n')

        preamble = ''
        if recirculators:
            if opt.add_mal:
                raise Exception('Legacy add_mal option specified but custom recirculators also present!')
            sl.append('!RECIRC \n')
            for el in recirculators:
                if self._validate_element(el, elements): continue
                sl.append(f"{el.id:<10}: RECIRC \n")
            sl.append('\n')
        else:
            if opt.add_mal:
                sl.append('RC: RECIRC \n')
                sl.append('MAL: MALIGN \n')
                sl.append('\n')
                preamble += 'MAL, RC,'

        # names = []
        # for el in markers:
        #     # name = pick_next_name(el['UNIQUENAME'], names)
        #     name = el['UNIQUENAME']
        #     sl.append('{:<10}: MARKER, FITPOINT=1\n'.format(name))
        #     # print(el['INDEX'],el['UNIQUENAME'],name)
        #     # iota.RenameElement(el['INDEX'], name)
        # sl.append('\n')

        if opt.global_aperture or opt.limiting_aperture:
            sl.append('!APERTURES\n')
            sl.append('!Default escape criterions in Elegant (from source code): SLOPE_LIMIT=1.0L, COORD_LIMIT=10.0L\n')

            if opt.global_aperture:
                v = opt.global_aperture
                preamble += ' MA1,'
                if isinstance(v, tuple):
                    xm, ym = v
                else:
                    xm, ym = 0.0254 * v, 0.0254 * v
                # xm, ym = 0.0381, 0.0381  # 1.5x aperture
                logger.warning(f'Global aperture set at {xm:.5f}/{ym:.5f}')
                # f.write('MA1: MAXAMP, X_MAX=0.025, Y_MAX=0.025, ELLIPTICAL=1 \n')
                # f.write('MA1: MAXAMP, X_MAX=0.030, Y_MAX=0.030, ELLIPTICAL=1 \n')
                # f.write('APER: RCOL, X_MAX=0.007, Y_MAX=0.007 \n')
                # f.write('MA1: MAXAMP, X_MAX=0.050, Y_MAX=0.050, ELLIPTICAL=1 \n')
                # sl.append('MA1: MAXAMP, X_MAX=0.0381, Y_MAX=0.0381, ELLIPTICAL=1 \n')  # 1.5x aperture
                # f.write('APER: ECOL, X_MAX=0.008, Y_MAX=0.008 \n')
                # sl.append('APER: ECOL, X_MAX=0.005925, Y_MAX=0.00789 \n')  # 1.5x actual NL aperture
                if xm == ym:
                    sl.append(f'MA1: MAXAMP, X_MAX={xm:+.10f}, Y_MAX={ym:+.10f} \n')
                else:
                    sl.append(f'MA1: MAXAMP, X_MAX={xm:+.10f}, Y_MAX={ym:+.10f}, ELLIPTICAL=1 \n')

            if opt.limiting_aperture:
                preamble += ' APER,'
                if opt.limiting_aperture != 1:
                    logger.warning(f'Limiting aperture scaled by ({opt.limiting_aperture})')
                    sl.append(f'! Limiting aperture scaled by ({opt.limiting_aperture}) from actual\n')
                    sl.append(f'APER: ECOL, X_MAX={3.9446881e-3 * opt.limiting_aperture:+.10f},'
                              f' Y_MAX={5.25958413e-3 * opt.limiting_aperture:+.10f}\n')
                else:
                    # 1x actual NL aperture
                    logger.warning(f'Realistic DN aperture added (X_MAX={3.9446881e-3:+.10f},'
                                   f' Y_MAX={5.25958413e-3:+.10f})')
                    sl.append(f'! Limiting aperture is not scaled (i.e. is actual)\n')
                    sl.append('APER: ECOL, X_MAX=3.9446881e-3, Y_MAX=5.25958413e-3\n')

            #           'is very large (10m), and there are timing issues with CLEAN\n')
            # sl.append('!It needs change_t, which doesnt work for 4D tracking (when RF has not been setup)\n')
            # sl.append('!For now, globally limit transverse aperture, since our beampipe is <=2in\n')
            # sl.append('!Also add worst restriction point in ring (at IOR)\n')
            # sl.append('!Will have to cut bucket escapes in post-processing\n')
            # sl.append('!C0: CLEAN, DELTALIMIT=0.1\n')
            sl.append('\n')
        else:
            logger.warning(f'No apertures specified - internal elegant ones will apply!')

        # sl.append('!Full line has everything\n')

        # seq = list(iota.sequence)
        # markerdict = markers.data
        # # print(seq,markerdict)
        # dipedge_names = [el['UNIQUENAME'] for el in iota.GetElementsOfType('DIPEDGE')]
        # seq2 = []
        # for s in seq:
        #     if s not in dipedge_names:
        #         seq2.append(s)
        #         if s in markerdict:
        #             seq2.append(s)
        #             # print(s)
        # wrapstr = 'iota: LINE=(CHRG, MAL, RC, MA1, APER, {})\n'.format(', '.join(seq2))

        seq_mod = [el.id for el in lat.sequence if not isinstance(el, Edge)]
        wrapstr = f"iota: LINE=({preamble} {', '.join(seq_mod)})\n"
        sl.append(textwrap.fill(wrapstr, break_long_words=False, break_on_hyphens=False).replace('\n', ' &\n'))
        sl.append('\n\n')
        # sl.append('!Short version has no monitors or markers, from elegant forum might make parallel version faster\n')
        # seq = list(iota.sequence)
        # dipedge_names = [el['UNIQUENAME'] for el in iota.GetElementsOfType('DIPEDGE')] + \
        #                 [el['UNIQUENAME'] for el in monitors] + \
        #                 [el['UNIQUENAME'] for el in markers]
        # seq2 = []
        # for s in seq:
        #     if s not in dipedge_names:
        #         seq2.append(s)
        # wrapstr = 'iota: LINE=(CHRG, MAL, RC, MA1, APER, {})\n'.format(', '.join([el.id for el in lat.sequence]))
        # sl.append(textwrap.fill(wrapstr, break_long_words=False, break_on_hyphens=False).replace('\n', ' &\n'))
        sl.append('\n')

        remainder = view.get_sequence()
        if remainder:
            raise Exception(
                f'Have leftover elements: {[(e.id, type(e), e.__class__.__name__) for e in remainder]}')

        result = ''.join(sl)

        if save:
            if not fpath:
                logger.warning(f'File path not specified, no writes will be performed')
            else:
                print(f'Writing lattice to: {fpath}')
                with open(str(fpath), 'w', newline='\n') as f:
                    f.write(result)

        return result