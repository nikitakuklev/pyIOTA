from typing import Union

from ocelot import Element, Sextupole, MagneticLattice, Octupole, twiss, Drift, Monitor
import numpy as np


class LatticeContainer:
    def __init__(self, name: str, lattice: list, correctors: list,
                 monitors: list, reset_elements_to_defaults: bool = True, info: dict = None):
        if not info:
            info = {'source_file': 'unknown', 'source': 'unknown'}
        self.name = name
        self.lattice_list = lattice
        self.correctors = correctors
        self.monitors = monitors
        self.source_file = info['source_file']
        self.source = info['source']
        self.pc = info['pc'] or 0.0

        if reset_elements_to_defaults:
            print(f'WARN - resetting any nonlinear elements to 0')
            elems = [l for l in self.lattice_list if isinstance(l, Sextupole)]
            for el in elems:
                el.k2 = 0

            elems = [l for l in self.lattice_list if isinstance(l, Octupole)]
            for el in elems:
                el.k3 = 0

        self.lattice = MagneticLattice(tuple(self.lattice_list))

        print(f'Lattice ({self.name}) initialized')
        # self.twiss = twiss(self.lattice)

    def insert_correctors(self, destroy_skew_quads: bool = True):
        if not destroy_skew_quads:
            raise Exception('Cannot place thick correctors without removing skew quads!')
        raise Exception('Thick correctors on top of other elements cannot be integrated, however they are added upon'
                        'export when possible (i.e. KQUAD has VKICK/HKICK set in elegant export)')

    def insert_monitors(self, monitors: list = None, verbose: bool = False):
        """
        Inserts Monitor type elements into lattice sequence where possible. This is necessary for 6dsim imports, where
        these elements are specified on top of regular sequence and are allowed to overlap.
        :param monitors:
        :param verbose:
        :return:
        """
        if not monitors:
            monitors = self.monitors
            print(f'No monitors specified - using default set')
        print('WARN - only monitors between elements can be inserted for thick lattices')
        lengths = [0.0] + [el.l for el in self.lattice.sequence]
        s = np.cumsum(lengths)
        s_dict = {k: v for k, v in zip(self.lattice.sequence, s)}
        s_inverse_dict = {v: k for k, v in zip(self.lattice.sequence, s)}
        # print('S:', {k.id: v for k, v in zip(self.lattice.sequence, s)})
        # print('Sinv:', {v: k.id for k, v in zip(self.lattice.sequence, s)})
        for m in monitors:
            m.s = s_dict[m.ref_el] + m.shift
            if m.s > self.lattice.totalLen:
                raise Exception(f'Monitor {m} is outside lattice length at {m.s}')
            else:
                if verbose: print(f'Resolved {m.id} position (ref {m.ref_el.id}) + {m.shift} = {m.s}')
        a, b, c = 0, 0, 0
        rejected = []
        for m in monitors:
            lengths = [0.0] + [el.l for el in self.lattice.sequence]
            s = np.cumsum(lengths)
            s_dict = {k: v for k, v in zip(self.lattice.sequence, s)}
            s_inverse_dict = {v: k for k, v in zip(self.lattice.sequence, s)}
            # print('S:', {k.id: v for k, v in zip(self.lattice.sequence, s)})
            if m.s in s or np.any(np.isclose(m.s, s)):
                i = np.where(np.isclose(m.s, s))[0][-1]
                # print(i,m.s)
                if verbose: print(
                    f'{m.id} - inserted at {m.s}(idx {i}) between {self.lattice.sequence[i - 1].id} and {self.lattice.sequence[i].id}')
                self.lattice.sequence.insert(i, m)
                self.lattice.update_transfer_maps()
                a += 1
            else:
                occupant_s_idx = np.where(s < m.s)[0][-1]
                occupant_s = s[occupant_s_idx]
                occupant = s_inverse_dict[occupant_s]
                if verbose: print(
                    f'{m.id} - location {m.s} in collision with ({occupant.__class__.__name__} {occupant.id})(sidx:{occupant_s_idx}) - length {occupant.l}m, between {s[max(occupant_s_idx - 1, 0):min(occupant_s_idx + 3, len(s) - 1)]}')
                if isinstance(occupant, Drift):
                    len_before = self.lattice.totalLen
                    d_before = Drift(eid=occupant.id + '_1', l=m.s - occupant_s)
                    d_after = Drift(eid=occupant.id + '_2', l=s[occupant_s_idx + 1] - m.s)
                    self.lattice.sequence.insert(occupant_s_idx, d_after)
                    self.lattice.sequence.insert(occupant_s_idx, m)
                    self.lattice.sequence.insert(occupant_s_idx, d_before)
                    self.lattice.sequence.remove(occupant)
                    if verbose: print(
                        f'{m.id} - inserted at {m.s}, two new drifts {d_before.l} and {d_after.l} created')
                    self.lattice.update_transfer_maps()
                    if self.lattice.totalLen != len_before:
                        raise Exception(f"Lattice length changed from {len_before} to {self.lattice.totalLen}!!!")
                    b += 1
                else:
                    if verbose: print(
                        f'Could not place monitor {m.id} at {m.s} - collision with {occupant.__class__.__name__} {occupant.id} - length {occupant.l}m, closest edges ({np.max(s[s < m.s])}|{np.min(s[s > m.s])})')
                    rejected.append(m.id)
                    c += 1
        print(f'Inserted ({a}) cleanly, ({b}) with drift splitting, ({c}) rejected: {rejected}')

    def get_elements(self, el_type: Union[str, type] = None):
        if isinstance(el_type, str):
            return [el for el in self.lattice.sequence if el.__class__.__name__ in el_type]
        else:
            return [el for el in self.lattice.sequence if isinstance(el, el_type)]

    def get_first(self, el_name: str = None, el_type: Union[str, type] = None):
        seq = self.lattice.sequence.copy()
        if el_name:
            seq = [el for el in seq if el_name in el.id]

        if el_type:
            if isinstance(el_type, str):
                seq = [el for el in seq if el.__class__.__name__ in el_type][0]
            else:
                seq = [el for el in seq if isinstance(el, el_type)][0]

        if len(seq) > 0:
            return seq[0]
        else:
            raise Exception(f'No matches found for {el_name}|{el_type}')

    def monitors_in_sequence(self):
        return [m for m in self.lattice.sequence if isinstance(m, Monitor)]


class DNMagnet(Element):
    """
    For our purposes, it is a drift
    l - length of drift in [m]
    """

    def __init__(self, l=0., eid=None):
        Element.__init__(self, eid)
        self.l = l