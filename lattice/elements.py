import logging
from pathlib import Path

import numpy as np
from typing import Union, List, Dict
from ocelot import Element, Sextupole, MagneticLattice, Octupole, Drift, Monitor, \
    Marker, Edge, SBend, twiss, Vcor, Hcor, Multipole

logger = logging.getLogger(__name__)


class LatticeContainer:
    def __init__(self,
                 name: str,
                 lattice: List,
                 correctors: List = None,
                 monitors: List = None,
                 reset_elements_to_defaults: bool = True,
                 info: Dict = None,
                 variables: Dict = None,
                 method=None):
        if not info:
            info = {'source_file': 'unknown', 'source': 'unknown', 'pc': 0.0}
        self.name = name
        self.lattice_list = lattice
        self.correctors = correctors
        self.monitors = monitors
        self.source_file = info['source_file']
        self.source = info['source']
        self.pc = info['pc']
        self.variables = variables

        if reset_elements_to_defaults:
            print(f'WARN - resetting any nonlinear elements to 0')
            elems = [l for l in self.lattice_list if isinstance(l, Sextupole)]
            for el in elems:
                el.k2 = 0

            elems = [l for l in self.lattice_list if isinstance(l, Octupole)]
            for el in elems:
                el.k3 = 0

        if not method:
            self.lattice = MagneticLattice(tuple(self.lattice_list))
        else:
            self.lattice = MagneticLattice(tuple(self.lattice_list), method=method)

        print(f'Lattice ({self.name}) initialized')
        # self.twiss = twiss(self.lattice)

    def insert_correctors(self, destroy_skew_quads: bool = True):
        """
        Placeholder until hell freezes over and we how learn to superimpose elements :(
        """
        if not destroy_skew_quads:
            raise Exception('Cannot place thick correctors without removing skew quads!')
        raise Exception('Thick correctors on top of other elements cannot be integrated, however they are added'
                        'upon export when possible (i.e. KQUAD has VKICK/HKICK set in elegant export)')

    def rotate_lattice(self, new_start: Element):
        """
        Changes lattice origin, moving entries to the end, by analogy to a circular buffer.
        :param new_start: Element name
        :return: None
        """
        seq = self.lattice.sequence
        self.lattice.sequence_original = seq
        if new_start not in seq:
            raise Exception(f'Element ({new_start.id}) not in current lattice')
        elif seq[0] == new_start:
            print(f'Lattice already starts with {new_start.id}')
        elif len([el for el in seq if el == new_start]) > 1:
            raise Exception(
                f'Too many element matches ({len([el for el in seq if el == new_start])}) to ({new_start.id})')
        else:
            i = seq.index(new_start)
            self.lattice.sequence = seq[i:] + seq[:i]
            print(
                f'Rotated by ({i}) - starting element is now ({self.lattice.sequence[0].id}) and ending element is ({self.lattice.sequence[-1].id})')

    def replace_element(self, old_el: Element, new_el: Element):
        """
        Swaps out elements. Throws exception if more than one match found.
        :param old_el: Old element
        :param new_el: New element
        :return: None
        """
        seq = self.lattice.sequence
        if old_el not in seq:
            raise Exception(f'Element ({old_el.id}) not in current lattice')
        elif len([el for el in seq if el == old_el]) > 1:
            raise Exception(f'Too many element matches ({len([el for el in seq if el == old_el])}) to ({old_el.id})')
        else:
            if new_el.l != old_el.l:
                print(f'New length ({new_el.l}) does not match old one ({old_el.l}) - careful!')
            i = seq.index(old_el)
            seq[i] = new_el
            print(f'Inserted at ({i}) - element is now ({seq[i].id})')

    def get_response_matrix(self):
        """
        Calculates standard RM
        :return:
        """
        import ocelot.cpbd.response_matrix
        ringrm = ocelot.cpbd.response_matrix.RingRM(lattice=self.lattice,
                                                    hcors=self.get_elements(Hcor),
                                                    vcors=self.get_elements(Vcor),
                                                    bpms=self.get_elements(Monitor))
        return ringrm.calculate()

    def update_element_positions(self):
        """
        Updates the s-values of all elements
        :return: None
        """
        l = 0.0
        slist = []
        for i, el in enumerate(self.lattice.sequence):
            el.s_start = l
            el.s_mid = l + el.l / 2
            el.s_end = l + el.l
            l += el.l
            slist.append(el.s_mid)
        return slist

    def update_twiss(self):
        return twiss(self.lattice)

    def remove_markers(self):
        """
        Remove all markers from sequence.
        :return: None
        """
        len_old = len(self.lattice.sequence)
        self.lattice.sequence = [s for s in self.lattice.sequence if not isinstance(s, Marker)]
        print(f'Removed ({len_old - len(self.lattice.sequence)}) markers')

    def remove_monitors(self):
        """
        Remove all monitors from sequence. Note that this WILL BREAK THINGS, since monitors are referenced
        to drifts before insertion splitting. Will be resolved...eventually.
        :return: None
        """
        len_old = len(self.lattice.sequence)
        print([s.id for s in self.lattice.sequence if isinstance(s, Monitor)])
        self.lattice.sequence = [s for s in self.lattice.sequence if not isinstance(s, Monitor)]
        print(f'Removed ({len_old - len(self.lattice.sequence)}) monitors')

    def insert_extra_markers(self, spacing: float = 1.0):
        """
        Inserts Markers between elements, at least with specified spacing and as close to it as possible
        :param spacing: spacing in meters
        :return: None
        """
        seq = self.lattice.sequence
        seq_new = [Marker(eid=f'MARKER_START')]
        l = l_last = 0.0
        for i, el in enumerate(seq):
            l += el.l
            if (isinstance(el, Edge) and isinstance(seq[i - 1], SBend)) or isinstance(el, SBend):
                # do not want to disturb edge links
                seq_new.append(el)
                continue
            if l > l_last + spacing:
                seq_new.append(Marker(eid=f'MARKER_{i}'))
                print(
                    f'Inserted monitor at ({l:.2f}) before ({el.id}) and after ({seq[i - 1].id}), ({l - l_last:.2f}) from last one')
                l_last = l
            seq_new.append(el)
        print(f'Done - inserted ({len(seq_new) - len(seq)}) markers')
        self.lattice.sequence = seq_new

    def insert_extra_monitors(self, spacing: float = 1.0, verbose: bool = False):
        """
        Inserts virtual monitors between elements, as close as possible but above specified spacing. Useful for
        defining orbit bumps and other optimization goals.
        :param spacing: Spacing in meters
        :param verbose:
        :return:
        """
        seq = self.lattice.sequence
        seq_new = [Monitor(eid=f'MONITOR_START')]
        l = l_last = 0.0
        for i, el in enumerate(seq):
            l += el.l
            if (isinstance(el, Edge) and isinstance(seq[i - 1], SBend)) or isinstance(el, SBend):
                # do not want to disturb edge links
                seq_new.append(el)
                continue
            if l > l_last + spacing:
                seq_new.append(Monitor(eid=f'MONITOR_{i}'))
                if verbose: print(f'Inserted monitor at ({l:.2f}) before ({el.id}) and'
                                  f' after ({seq[i - 1].id}), ({l - l_last:.2f}) from last one')
                l_last = l
            seq_new.append(el)
        print(f'Inserted ({len(seq_new) - len(seq)}) monitors')
        self.lattice.sequence = seq_new

    def insert_monitors(self, monitors: List[Monitor] = None, verbose: bool = False):
        """
        Inserts Monitor type elements into lattice sequence where possible. This is necessary for 6DSim imports,
        where these elements are specified on top of regular sequence and are allowed to overlap.
        :param monitors: monitors to insert
        :param verbose:
        :return: None
        """
        if not monitors:
            monitors = self.monitors
            logger.warning(f'No monitors specified - inserting default set')
        logger.warning('Only monitors between elements can be inserted for thick lattices')

        lengths = [0.0] + [el.l for el in self.lattice.sequence]
        s = np.cumsum(lengths)
        s_dict = {k: v for k, v in zip(self.lattice.sequence, s)}
        s_inverse_dict = {v: k for k, v in zip(self.lattice.sequence, s)}
        for m in monitors:
            m.s_mid = s_dict[m.ref_el] + m.shift
            if m.s_mid > self.lattice.totalLen:
                raise Exception(f'Monitor {m} is outside lattice length at {m.s_mid}')
            else:
                if verbose: print(f'Resolved {m.id} position (ref {m.ref_el.id}) + {m.shift} = {m.s_mid}')

        a, b, c = 0, 0, 0
        rejected = []
        for m in monitors:
            lengths = [0.0] + [el.l for el in self.lattice.sequence]
            s = np.cumsum(lengths)
            s_dict = {k: v for k, v in zip(self.lattice.sequence, s)}
            s_inverse_dict = {v: k for k, v in zip(self.lattice.sequence, s)}
            # print('S:', {k.id: v for k, v in zip(self.lattice.sequence, s)})

            if m.s_mid in s or np.any(np.isclose(m.s_mid, s)):
                i = np.where(np.isclose(m.s_mid, s))[0][-1]
                # print(i,m.s)
                if verbose: print(
                    f'{m.id} - inserted at {m.s_mid}(idx {i}) between {self.lattice.sequence[i - 1].id}'
                    f' and {self.lattice.sequence[i].id}')
                self.lattice.sequence.insert(i, m)
                self.lattice.update_transfer_maps()
                a += 1
            else:
                occupant_s_idx = np.where(s < m.s_mid)[0][-1]
                occupant_s = s[occupant_s_idx]
                occupant = s_inverse_dict[occupant_s]
                if verbose: print(
                    f'{m.id} - location {m.s_mid} in collision with ({occupant.__class__.__name__} {occupant.id})'
                    f'(sidx:{occupant_s_idx}) - length {occupant.l}m, between {s[max(occupant_s_idx - 1, 0):min(occupant_s_idx + 3, len(s) - 1)]}')
                if isinstance(occupant, Drift):
                    len_before = self.lattice.totalLen
                    d_before = Drift(eid=occupant.id + '_1', l=m.s_mid - occupant_s)
                    d_after = Drift(eid=occupant.id + '_2', l=s[occupant_s_idx + 1] - m.s_mid)
                    self.lattice.sequence.insert(occupant_s_idx, d_after)
                    self.lattice.sequence.insert(occupant_s_idx, m)
                    self.lattice.sequence.insert(occupant_s_idx, d_before)
                    self.lattice.sequence.remove(occupant)
                    if verbose: print(
                        f'{m.id} - inserted at {m.s_mid}, two new drifts {d_before.l} and {d_after.l} created')
                    self.lattice.update_transfer_maps()
                    if self.lattice.totalLen != len_before:
                        raise Exception(f"Lattice length changed from {len_before} to {self.lattice.totalLen}!!!")
                    b += 1
                else:
                    if verbose: print(
                        f'Could not place monitor {m.id} at {m.s_mid} - collision with {occupant.__class__.__name__} '
                        f'{occupant.id} - length {occupant.l}m, closest edges ({np.max(s[s < m.s_mid])}|{np.min(s[s > m.s_mid])})')
                    rejected.append(m.id)
                    c += 1
        print(f'Inserted ({a}) cleanly, ({b}) with drift splitting, ({c}) rejected: {rejected}')

    def get_elements(self, el_type: Union[str, type] = None):
        """
        Gets all elements of type in current sequence
        :param el_type:
        :return:
        """
        if isinstance(el_type, str):
            return [el for el in self.lattice.sequence if el.__class__.__name__ in el_type]
        else:
            return [el for el in self.lattice.sequence if isinstance(el, el_type)]

    def get_first(self, el_name: str = None, el_type: Union[str, type] = None):
        """
        Gets first element matching any non-None conditions
        :param el_name: Element name string
        :param el_type: Class object or name string
        :return: Element
        """
        seq = self.lattice.sequence.copy()
        if el_name:
            el_name = el_name.upper()
            seq = [el for el in seq if el_name in el.id]

        if el_type:
            if isinstance(el_type, str):
                seq = [el for el in seq if el.__class__.__name__ in el_type]
            else:
                seq = [el for el in seq if isinstance(el, el_type)]

        if len(seq) > 0:
            return seq[0]
        else:
            raise Exception(f'No matches found for {el_name}|{el_type}')

    def filter(self, fun):
        """
        Filter by supplied function and return those that are True
        :param fun:
        :return:
        """
        return [el for el in self.lattice.sequence if fun(el)]

    def filter_by_id(self, id_list: List[str], loose_match=True):
        """
        Filters by trying to match element id to any string in the list
        :param id_list: List of strings
        :param loose_match: Whether to consider partial matches as ok
        :return:
        """
        if isinstance(id_list, str):
            id_list = [id_list]
        if loose_match:
            return [el for el in self.lattice.sequence if any([el.id in item for item in id_list])]
        else:
            return [el for el in self.lattice.sequence if el.id in id_list]

    # Convenience utils

    class OneTimeView:
        """
        Utility class that mirrors current sequence on creation and then removes elements anytime they
        are accessed, as if popping things from stack. Used to make sure all elements are processed during
        conversions.
        """
        def __init__(self, sequence):
            # Shallow copy only
            self.seq = sequence.copy()

        def get_elements(self, el_type):
            if isinstance(el_type, str):
                matches = [el for el in self.seq if el.__class__.__name__ in el_type]
            else:
                matches = [el for el in self.seq if isinstance(el, el_type)]
            for m in matches:
                self.seq.remove(m)
            return matches

        def get_sequence(self):
            return self.seq

    def get_onetimeview(self):
        return self.OneTimeView(self.lattice.sequence)

    # Conversion functions

    def to_elegant(self, lattice_options: Dict, lattice_path_abs: Path, dry_run: bool = False):
        """
        Calls elegant submodule library to produce elegant lattice.
        Should fail-fast if incompatible features are found.
        :return:
        """
        import pyIOTA.elegant.latticefile
        wr = pyIOTA.elegant.latticefile.Writer(options=lattice_options)
        return wr.write_lattice_ng(fpath=lattice_path_abs, box=self, save=not dry_run)


class NLLens(Element):
    """
    For our purposes, it is a drift
    l - length of drift in [m]
    """

    def __init__(self, l=0., eid=None):
        Element.__init__(self, eid)
        self.l = l


class HKPoly(Element):
    """
    Arbitrary Hamiltonian element for ELEGANT
    """

    def __init__(self, l=0., eid=None, **kwargs):
        Element.__init__(self, eid)
        self.l = l
        self.hkpoly_args = kwargs


class OctupoleInsert:
    """
    A collection of octupoles for quasi-integrable insert that can be generated into a sequence
    """

    def __init__(self):
        self.l0 = self.mu0 = None
        self.seq = []
        pass

    def configure(self, oqK=1.0, run=2, l0=1.8, mu0=0.3, otype=1, olen=0.07, nn=17):
        """
        Initializes QI configuration
        :param l: #l0     = 1.8;        # length of the straight section
        :param mu0: #mu0 = 0.3;  # phase advance over straight section
        :param run: # which run configuration to use, 1 or 2
        :param otype: # type of magnet (0) thin, (1) thick, only works for octupoles (ncut=4)
        :param olen: #olen = 0.07  # length of octupole for thick option.must be < l0 / nn
        :param nn: # number of nonlinear elements
        """
        self.l0 = l0
        self.mu0 = mu0
        tn = 0.4  # strength of nonlinear lens
        cn = 0.01  # dimentional parameter of nonlinear lens

        if run == 1:
            # Margins on the sides were present
            oqSpacing = 0.03325  # (1.8-0.022375-0.022375-17*0.07)/17
            margin = 0.022375  # extra margin at the start and end of insert
            positions = margin + (l0 - 2 * margin) / nn * (np.arange(1, nn + 1) - 0.5)
        elif run == 2:
            # Perfect spacing with half drift on each end
            oqSpacing = (l0 - olen * nn) / nn
            margin = 0.0
            positions = l0 / nn * (np.arange(1, nn + 1) - 0.5)
        else:
            raise Exception

        # musect = mu0 + 0.5
        f0, betae, alfae, betas = self.calculate_optics_parameters()

        print(
            f"QI optics: mu0:{mu0:.3f}|f0:{f0:.3f}|1/f0:{1 / f0:.3f}|"
            f"betaedge:{betae:.3f}|alphaedge:{alfae:.3f}|betastar:{betas:.3f}")
        # value, , oqK, nltype, otype;

        self.seq = seq = []
        seq.append(Drift(l=margin, eid='oQImargin'))
        for i in range(1, nn + 1):
            sn = positions[i - 1]
            bn = l0 * (1 - sn * (l0 - sn) / l0 / f0) / np.sqrt(1.0 - (1.0 - l0 / 2.0 / f0) ** 2)
            knn = tn * l0 / nn / bn ** 2
            cnll = cn * np.sqrt(bn)
            knll = knn * cnll ** 2
            k1 = knn * 2  # 1 * 2!
            k3 = knn / cn ** 2 / bn * 16  # 2 / 3 * 4!
            k3scaled = k3 * oqK

            if otype == 1:
                seq.append(Drift(l=oqSpacing / 2, eid=f'oQI{i:02}l'))
                seq.append(Octupole(l=olen, k3=k3scaled / olen, eid=f'QI{i:02}'))
                seq.append(Drift(l=oqSpacing / 2, eid=f'oQI{i:02}r'))
            elif otype == 0:
                seq.append(Drift(l=oqSpacing / 2 + olen / 2, eid=f'oQI{i:02}l'))
                seq.append(Multipole(kn=[0., 0., 0., k3scaled], eid=f'QI{i:02}'))
                seq.append(Drift(l=oqSpacing / 2 + olen / 2, eid=f'oQI{i:02}r'))
            # value, i, bn, sn, k3, k3scaled, (betas ^ 3 / bn ^ 3);
        seq.append(Drift(l=margin, eid='oQImargin'))
        l_list = [e.l for e in seq]
        assert np.isclose(sum(l_list), l0)

        s_list = []
        s = 0
        for e in seq:
            s += e.l / 2
            s_list.append(s)
            s += e.l / 2

        s_oct = [s for e, s in zip(seq, s_list) if not isinstance(e, Drift)]
        assert np.allclose(np.diff(np.array(s_oct)), oqSpacing + olen)

    def calculate_optics_parameters(self):
        """
        Calculates the key descriptive parameters of insert based on length and phase advance
        :return:f0, betae, alfae, betas
        """
        l0 = self.l0
        mu0 = self.mu0
        f0 = l0 / 4.0 * (1.0 + 1.0 / np.tan(np.pi * mu0) ** 2)
        betae = l0 / np.sqrt(1.0 - (1.0 - l0 / 2.0 / f0) ** 2)
        alfae = l0 / 2.0 / f0 / np.sqrt(1.0 - (1.0 - l0 / 2.0 / f0) ** 2)
        betas = l0 * (1 - l0 / 4.0 / f0) / np.sqrt(1.0 - (1.0 - l0 / 2.0 / f0) ** 2)
        return f0, betae, alfae, betas

    def set_current(self):
        pass

    def to_sequence(self):
        return self.seq
