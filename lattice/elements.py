from __future__ import annotations

__all__ = ['LatticeContainer', 'NLLens', 'HKPoly', 'ILMatrix', 'OctupoleInsert']

import logging
from pathlib import Path

from typing import Union, List, Dict, Type, Iterable, Callable, Optional
from ocelot.cpbd.elements import *
from ocelot import MagneticLattice, twiss, MethodTM

logger = logging.getLogger(__name__)


class LatticeContainer:
    logger = logging.getLogger('LatticeContainer')

    def __init__(self,
                 name: str,
                 lattice: List,
                 correctors: List = None,
                 monitors: List = None,
                 reset_elements_to_defaults: bool = True,
                 info: Dict = None,
                 variables: Dict = None,
                 method: MethodTM = None,
                 silent: bool = True):
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
        self.silent = silent

        if reset_elements_to_defaults:
            logger.warning(f'Resetting any nonlinear elements to 0')
            # print(f'WARN - resetting any nonlinear elements to 0')
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

        # if not silent: print(f'Lattice ({self.name}) initialized')
        if not silent: logger.info(f'Lattice ({self.name}) initialized')
        # self.twiss = twiss(self.lattice)

    def insert_correctors(self, destroy_skew_quads: bool = True):
        """
        Placeholder until hell freezes over and we how learn to superimpose elements :(
        """
        if not destroy_skew_quads:
            raise Exception('Cannot place thick correctors without removing skew quads!')
        raise Exception('Thick correctors on top of other elements cannot be integrated, however they are added'
                        'upon export when possible (i.e. KQUAD has VKICK/HKICK set in elegant export)')

    # Lattice manipulation methods

    def rotate_lattice(self, new_start: Element) -> LatticeContainer:
        """
        Changes lattice origin, moving entries to the end, by analogy to a circular buffer.
        :param new_start: New starting element
        :return: LatticeContainer
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
            if not self.silent:
                logger.info(f'Rotated by ({i}) - starting element is ({self.lattice.sequence[0].id}) and ending'
                            f' is ({self.lattice.sequence[-1].id})')
        assert len(seq) == len(self.lattice.sequence)
        return self

    def replace_elements(self, old_el: Union[Element, Iterable], new_el: Union[Element, Iterable]) -> LatticeContainer:
        """
        Swaps out elements. Throws exception if more than one match found.
        :param old_el: Old element, or list of elements
        :param new_el: New element, or list of elements
        :return: LatticeContainer
        """
        seq = self.lattice.sequence
        if isinstance(old_el, List) and isinstance(new_el, List):
            assert all(isinstance(e, Element) for e in old_el)
            assert all(isinstance(e, Element) for e in new_el)
            assert len(old_el) == len(new_el)
        elif isinstance(old_el, Element) and isinstance(new_el, Element):
            old_el = [old_el]
            new_el = [new_el]
        else:
            raise Exception(f'Both new and old parameters must be lists or Element')

        for oel, nel in zip(old_el, new_el):
            matches = [el for el in seq if el == oel]
            if len(matches) == 0:
                raise Exception(f'Element ({oel.id}) not in current lattice')
            elif len(matches) > 1:
                raise Exception(f'Too many matches ({len(matches)}) for ({oel.id})')
            else:
                if nel.l != oel.l:
                    print(f'New length ({nel.l}) does not match old one ({oel.l}) - careful!')
                i = seq.index(old_el)
                seq[i] = new_el
                if not self.silent:
                    print(f'Inserted at ({i}) - element is now ({seq[i].id})')
        return self

    def transmute_elements(self,
                           elements: Union[Element, Iterable[Element]],
                           new_type: Type[Element],
                           verbose: bool = False) -> LatticeContainer:
        """
        Transmutes element type to new one. Only length and id are preserved.
        :param elements: Elements to transmute
        :param new_type: New element type
        :param verbose:
        :return: LatticeContainer
        """
        seq = self.lattice.sequence
        l_seq = len(seq)
        if isinstance(elements, List) and isinstance(elements, List):
            assert all(isinstance(e, Element) for e in elements)
        elif isinstance(elements, Element):
            elements = [elements]
        else:
            raise Exception(f'Both new and old parameters must be lists or Element')

        for target in elements:
            matches = [el for el in seq if el == target]
            if len(matches) == 0:
                raise Exception(f'Element ({target.id}) not in current lattice')
            elif len(matches) > 1:
                raise Exception(f'Too many matches ({len(matches)}) for ({target.id})')
            else:
                new_el = new_type(eid=target.id)
                if hasattr(target, 'l'):
                    new_el.l = target.l
                else:
                    raise Exception(f'Target ({target.id}) has no length attribute - is it an Element?')
                i = seq.index(target)
                seq[i] = new_el
                if verbose: print(f'Transmuted ({new_el.id}) at pos ({i}) from ({matches[0].__class__.__name__}) '
                                  f'to ({seq[i].__class__.__name__})')
        assert l_seq == len(seq)
        # if not self.silent: print(f'Transmuted ({len(elements)}) elements')
        if not self.silent: logger.info(f'Transmuted ({len(elements)}) elements')
        return self

    def split_elements(self, elements: Union[Element, Iterable[Element]], n_parts: int = 2) -> LatticeContainer:
        """
        Splits elements into several parts, scaling parameters appropriately
        :param elements: Element or list of elements to be split
        :param n_parts: Number of parts to split into
        :return: LatticeContainer
        """
        from copy import deepcopy
        seq = self.lattice.sequence
        if isinstance(elements, List) and isinstance(elements, List):
            assert all(isinstance(e, Element) for e in elements)
        elif isinstance(elements, Element):
            elements = [elements]
        else:
            raise Exception(f'Both new and old parameters must be lists or Element')

        scaled_parameters = ['l', 'k1', 'k2', 'k3', 'k4']
        seq_new = seq.copy()
        for el in elements:
            el_list = [deepcopy(el) for i in range(n_parts)]
            for i, e in enumerate(el_list):
                for s in scaled_parameters:
                    if hasattr(e, s):
                        setattr(e, s, getattr(e, s) / n_parts)
                e.id = el.id + f'_{i}'
            idx = seq_new.index(el)
            seq_new.remove(el)
            seq_new[idx:idx] = el_list  # means insert here
        assert len(seq_new) == len(seq) + len(elements) * (n_parts - 1)
        if not self.silent:
            logger.info(f'Split elements ({[el.id for el in elements]}) into ({n_parts}) parts - seq length'
                        f' ({len(seq)}) -> ({len(seq_new)})')
        self.lattice.sequence = seq_new
        return self

    def insert_elements(self,
                        elements: Union[Element, Iterable[Element]],
                        before: Element = None,
                        after: Element = None) -> LatticeContainer:
        """
        Inserts element before or after another element
        :param elements: Elements to insert
        :param before: Insert before this element
        :param after: Insert after this element (if before is not specified)
        """
        seq = self.lattice.sequence
        seq_new = []
        assert before or after
        assert not (before and after)
        target = before or after
        if not isinstance(elements, list):
            elements = [elements]
        for i, el in enumerate(seq):
            if el == target:
                if before:
                    seq_new.extend(elements)
                    seq_new.append(el)
                else:
                    seq_new.append(el)
                    seq_new.extend(elements)
            else:
                seq_new.append(el)
        logger.info(f'Inserted ({len(seq_new) - len(seq)}) markers')
        self.lattice.sequence = seq_new
        return self

    def remove_markers(self):
        """
        Remove all markers
        :return: LatticeContainer
        """
        len_old = len(self.lattice.sequence)
        self.lattice.sequence = [s for s in self.lattice.sequence if not isinstance(s, Marker)]
        logger.info(f'Removed ({len_old - len(self.lattice.sequence)}) markers')
        return self

    def remove_monitors(self):
        """
        Remove all monitors from sequence. Note that this WILL BREAK THINGS, since monitors are referenced
        to drifts before insertion splitting. Will be resolved...eventually.
        :return: None
        """
        len_old = len(self.lattice.sequence)
        print([s.id for s in self.lattice.sequence if isinstance(s, Monitor)])
        self.lattice.sequence = [s for s in self.lattice.sequence if not isinstance(s, Monitor)]
        logger.info(f'Removed ({len_old - len(self.lattice.sequence)}) monitors')

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
        Updates the s-values of all elements. Does not overwrite default ocelot 's' parameter - use 's_mid' instead
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

    def update_twiss(self, n_points: int = None, update_maps: bool = True):
        if update_maps:
            self.lattice.update_transfer_maps()
        return twiss(self.lattice, nPoints=n_points)

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
        # print(f'Inserted ({a}) cleanly, ({b}) with drift splitting, ({c}) rejected: {rejected}')
        logger.info(f'Inserted ({a}) cleanly, ({b}) with drift splitting, ({c}) rejected: {rejected}')

    def merge_drifts(self, exclusions: List[Drift] = None, verbose: bool = False, silent: bool = False):
        """
        Merges consecutive drifts in the lattice, except those in exclusions list
        :param exclusions:
        :param verbose:
        :return:
        """
        seq = self.lattice.sequence
        if exclusions:
            assert all(isinstance(el, Drift) for el in exclusions)
        else:
            exclusions = []
        seq_new = []
        l_new = 0
        l_total = sum([el.l for el in self.lattice.sequence])
        cnt = 0
        name_new = ''
        drift_mode = False
        for el in self.lattice.sequence:
            is_exclusion = el in exclusions
            if isinstance(el, Drift) and not is_exclusion:
                if drift_mode:
                    l_new += el.l
                    name_new += '_' + el.id
                    if verbose: print(f'Found consecutive drift ({el.id}) - length ({el.l:.5f})')
                else:
                    drift_mode = True
                    l_new += el.l
                    name_new += el.id
                    if verbose: print(f'Found first drift ({el.id}) - length ({el.l:.5f})')
            else:
                if is_exclusion:
                    print(f'Skipping drift ({el.id}) since it is excluded')
                if drift_mode:
                    seq_new.append(Drift(l=l_new, eid=name_new))
                    if verbose: print(f'Consecutive drifts ended - creating ({name_new}) of length ({l_new:.5f})')
                    name_new = ''
                    l_new = 0
                    drift_mode = False
                    cnt += 1
                seq_new.append(el)
        if l_new != 0:
            if verbose: print(f'Sequence ended - creating ({name_new}) of length ({l_new})')
            seq_new.append(Drift(l=l_new, eid=name_new))

        if not self.silent: print(
            f'Reduced element count from ({len(seq)}) to ({len(seq_new)}), ({cnt}) drifts remaining')
        if not np.isclose(l_total, sum([el.l for el in seq_new])):
            raise Exception(f'New sequence length ({sum([el.l for el in seq_new])} different from old ({l_total})!!!')
        self.lattice.sequence = seq_new

    # Getters/setters

    def get_first(self,
                  el_name: str = None,
                  el_type: Union[str, type] = None,
                  exact: bool = False,
                  singleton_only: bool = False,
                  last: bool = False) -> Element:
        """
        Gets first element matching any non-None conditions
        :param last:
        :param singleton_only:
        :param exact:
        :param el_name: Element name string
        :param el_type: Class object or name string
        :return: Element
        """
        seq = self.lattice.sequence.copy()
        if el_name:
            el_name = el_name.upper()
            if exact:
                seq = [el for el in seq if el_name == el.id.upper()]
            else:
                seq = [el for el in seq if el_name in el.id.upper()]

        if el_type:
            if isinstance(el_type, str):
                seq = [el for el in seq if el.__class__.__name__ in el_type]
            else:
                seq = [el for el in seq if isinstance(el, el_type)]

        if len(seq) > 0:
            if singleton_only:
                if len(seq) > 1:
                    raise Exception(f'Multiple matches found for (name:{el_name}|type:{el_type})')
            if last:
                return seq[-1]
            else:
                return seq[0]
        else:
            raise Exception(f'No matches found for (name:{el_name}|type:{el_type})')

    def get_last(self,
                 el_name: str = None,
                 el_type: Union[str, type] = None,
                 exact: bool = False,
                 singleton_only: bool = False):
        return self.get_first(el_name, el_type, exact, singleton_only, last=True)

    def get_elements(self, el_type: Union[str, type] = None) -> List[Optional[type]]:
        """
        Gets all elements of type in sequence
        :param el_type:
        :return: List of elements
        """
        if isinstance(el_type, str):
            return [el for el in self.lattice.sequence if el.__class__.__name__ in el_type]
        else:
            return [el for el in self.lattice.sequence if isinstance(el, el_type)]

    def get_all(self, el_type: Union[str, type] = None) -> List[Optional[type]]:
        return self.get_elements(el_type)

    def __getitem__(self, item) -> Element:
        """
        Gets element by id
        :param item: Element name string
        :return: Element
        """
        return self.get_first(el_name=item, exact=True, singleton_only=True)

    def filter(self, fun: Callable) -> List[Element]:
        """
        Filter elements by supplied function and return those that are True
        :param fun: Function to call
        :return:
        """
        return [el for el in self.lattice.sequence if fun(el)]

    def filter_elements(self, el_name: str = None, el_type: Union[str, type] = None) -> List[Element]:
        """
        Filter elements by type and regex
        :return:
        """
        import re
        el_list = self.lattice.sequence
        if el_type:
            if isinstance(el_type, str):
                el_list = [el for el in self.lattice.sequence if el.__class__.__name__ in el_type]
            else:
                el_list = [el for el in self.lattice.sequence if isinstance(el, el_type)]
        if el_name:
            r = re.compile(el_name.upper(), re.IGNORECASE)
            new_list = []
            for el in el_list:
                if r.match(el.id):
                    new_list.append(el)
            el_list = new_list
        return el_list

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

        def get_elements(self, el_type: Element):
            if isinstance(el_type, str):
                raise Exception()  # this is deprecated since had to do OCELOT classname bypasses
                # matches = [el for el in self.seq if el.__class__.__name__ in el_type]
            else:
                matches = [el for el in self.seq if type(el) == el_type]  # match type exactly, no inheritance
                # print(el_type, len(matches), [m.id for m in matches], type(self.seq[0]))
            for m in matches:
                self.seq.remove(m)
            return matches

        def get_sequence(self):
            return self.seq

    def get_onetimeview(self):
        return self.OneTimeView(self.lattice.sequence)

    # Conversion functions

    def to_elegant(self, fpath: Path, lattice_options: Dict, dry_run: bool = False):
        """
        Calls elegant submodule library to produce elegant lattice.
        Should fail-fast if incompatible features are found.
        :return:
        """
        import pyIOTA.elegant as elegant
        assert isinstance(fpath, Path) and isinstance(lattice_options, dict)
        wr = elegant.Writer(options=lattice_options)
        return wr.write_lattice_ng(fpath=fpath, box=self, save=not dry_run)


class NLLens(Element):
    """
    For our purposes, it is a drift
    l - length of drift in [m]
    """

    def __init__(self, l: float = 0.0, eid: str = None):
        Element.__init__(self, eid)
        self.l = l


class HKPoly(Element):
    """
    Arbitrary Hamiltonian element for ELEGANT
    """

    def __init__(self, l: float = 0.0, eid: str = None, **kwargs: float):
        Element.__init__(self, eid)
        self.l = l
        self.hkpoly_args = kwargs


class ILMatrix(Matrix):
    """
    ILMATRIX element for ELEGANT
    """

    @property
    def __class__(self):
        """
        OCELOT uses a stupid way to check objects by class name, so we have to fake ourselves to be the superclass
        :return:
        """
        return Matrix

    def __init__(self, l: float = 0.0, eid: str = None, **kwargs: float):
        self.l = l
        self.id = eid
        self.extra_args = kwargs
        mdict = {}
        for i, plane in enumerate(['x', 'y']):
            offset = i * 2 + 1
            sin_phi = np.sin(kwargs['nux'] * 2 * np.pi)
            # if np.isclose(sin_phi, 0, rtol=1.e-12, atol=1.e-12):
            #    sin_phi = 0.0
            cos_phi = np.cos(kwargs['nux'] * 2 * np.pi)
            alpha1 = kwargs.get('alphax', 0.0)
            alpha2 = -alpha1  # Tinsert
            beta1 = kwargs.get('betax', 0.0)
            mdict[f'r{offset}{offset}'] = cos_phi + alpha1 * sin_phi
            mdict[f'r{offset + 1}{offset + 1}'] = cos_phi - alpha1 * sin_phi
            mdict[f'r{offset}{offset + 1}'] = beta1 * sin_phi
            mdict[f'r{offset + 1}{offset}'] = -((1 + alpha1 * alpha2) / beta1) * sin_phi + (
                    (alpha1 - alpha2) / beta1) * cos_phi
        mdict['r55'] = 1.0
        mdict['r66'] = 1.0
        super().__init__(eid=eid, l=l, **mdict)


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
