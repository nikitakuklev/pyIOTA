class BPMS:

    @staticmethod
    def to_acnet(names: list):
        return ['N:' + n for n in names]

    @staticmethod
    def from_acnet(names: list):
        return [n[2:] if n[0:2] == 'N:' else n for n in names]

    @staticmethod
    def add_axis(names: list, axis: str):
        assert axis in ['H', 'V', 'S']
        return [n + axis for n in names]

    LATTICE_NAMES = ['IBB2R', 'IBC1R', 'IBC2R', 'IBD1R', 'IBD2R', 'IBE1R', 'IBE2R', 'IBE2L', 'IBE1L', 'IBD2L', 'IBD1L',
                     'IBC2L', 'IBC1L', 'IBB2L', 'IBB1L', 'IBA3L', 'IBA2L', 'IBA1C', 'IBA2R', 'IBA3R', 'IBB1R']

    ACNET_NAMES = to_acnet.__func__(LATTICE_NAMES)

    H = add_axis.__func__(LATTICE_NAMES, 'H')
    HA = to_acnet.__func__(H)

    V = add_axis.__func__(LATTICE_NAMES, 'V')
    VA = to_acnet.__func__(V)

    S = add_axis.__func__(LATTICE_NAMES, 'S')
    SA = to_acnet.__func__(S)

    ALL = H+V+S
    ALLA = HA+VA+SA


class CORRECTORS:
    __combfun = ['A1R', 'A2R', 'B1R', 'B2R', 'C1R', 'C2R', 'D1R', 'D2R', 'E1R', 'E2R',
                 'E2L', 'E1L', 'D2L', 'D1L', 'C2L', 'C1L', 'B2L', 'B1L', 'A2L', 'A1L']

    DIPOLE_SHIMS = ['N:IHM' + str(i) + 'LI' for i in range(1, 5)] + ['N:IHM' + str(i) + 'RI' for i in range(1, 5)]

    COMBINED_COILS = [i for sl in [['N:I1' + k + 'I', 'N:I2' + k + 'I', 'N:I3' + k + 'I', 'N:I4' + k + 'I'] for k in __combfun] for i in sl]

    COMBINED_VIRTUAL = [i for sl in [['N:IV' + k + 'I', 'N:IH' + k + 'I'] for k in __combfun] for i in sl]

    OTHER_CORRECTORS = ['N:IBMPLI', 'N:IBMPRI', 'N:IBEND', 'N:ILAM', 'N:IHLAMI']

    COMBINED_COILS_AND_DIPOLE_SHIMS = COMBINED_COILS + DIPOLE_SHIMS

    COMBINED_VIRTUAL_AND_DIPOLE_SHIMS = COMBINED_VIRTUAL + DIPOLE_SHIMS

    COMBINED_COILS_AND_DIPOLE_SHIMS_AND_OTHER = COMBINED_COILS_AND_DIPOLE_SHIMS + OTHER_CORRECTORS

    COMBINED_VIRTUAL_AND_DIPOLE_SHIMS_AND_OTHER = COMBINED_VIRTUAL_AND_DIPOLE_SHIMS + OTHER_CORRECTORS

    ALL_COILS = COMBINED_COILS_AND_DIPOLE_SHIMS_AND_OTHER
    ALL_VIRTUAL = COMBINED_VIRTUAL_AND_DIPOLE_SHIMS_AND_OTHER

    ALL = list(set(COMBINED_COILS_AND_DIPOLE_SHIMS_AND_OTHER +
                   COMBINED_VIRTUAL_AND_DIPOLE_SHIMS_AND_OTHER))


class QUADS:
    __quad_names = ['A1R', 'A2R', 'A3R', 'A4R', 'B1R', 'B2R', 'B3R', 'B4R', 'B5R', 'B6R',
                    'C1R', 'C2R', 'C3R', 'D1R', 'D2R', 'D3R', 'D4R', 'E1R', 'E2R', 'E3C',
                    'A1L', 'A2L', 'A3L', 'A4L', 'B1L', 'B2L', 'B3L', 'B4L', 'B5L', 'B6L',
                    'C1L', 'C2L', 'C3L', 'D1L', 'D2L', 'D3L', 'D4L', 'E1L', 'E2L']
    ALL_CURRENTS = ['N:IQ' + str(i) + 'I' for i in __quad_names]
    ALL_VOLTAGES = ['N:IQ' + str(i) + 'V' for i in __quad_names]
    ALL = ALL_CURRENTS + ALL_VOLTAGES


class SKEWQUADS:
    __skew_names = ['A1R', 'A2R', 'B1R', 'B2R', 'C1R', 'C2R', 'D1R', 'D2R', 'E1R', 'E2R',
                    'A1L', 'A2L', 'B1L', 'B2L', 'C1L', 'C2L', 'D1L', 'D2L', 'E1L', 'E2L']
    ALL_CURRENTS = ['N:IK' + str(i) + 'I' for i in __skew_names]
    ALL_VOLTAGES = ['N:IK' + str(i) + 'V' for i in __skew_names]
    ALL = ALL_CURRENTS + ALL_VOLTAGES


class SEXTUPOLES:
    __sext_names = ['A1R', 'C1R', 'C2R', 'D1R', 'E1R', 'E2R',
                    'A2L', 'C1L', 'C2L', 'D1L', 'E1L', 'E2L']
    ALL_CURRENTS = ['N:IS' + str(i) + 'I' for i in __sext_names]
    ALL_VOLTAGES = ['N:IS' + str(i) + 'V' for i in __sext_names]
    ALL = ALL_CURRENTS + ALL_VOLTAGES


class OCTUPOLES:
    ALL_CURRENTS = ['N:IO' + str(i) + 'LI' for i in range(1, 19)] + ['N:OB9L6I', 'N:O4L6I', 'N:O14L6I']
    ALL_VOLTAGES = ['N:IO' + str(i) + 'LV' for i in range(1, 19)] + ['N:OB9L6V', 'N:O4L6V', 'N:O14L6V']
    ALL_CURRENTS_ACTIVE = ['N:IO' + str(i) + 'LI' for i in range(1, 9)] + ['N:OB9L6I'] + \
                          ['N:IO' + str(i) + 'LI' for i in range(10, 18)]
    ALL_VOLTAGES_ACTIVE = ['N:IO' + str(i) + 'LV' for i in range(1, 9)] + ['N:OB9L6V'] + \
                          ['N:IO' + str(i) + 'LV' for i in range(10, 18)]
    ALL = ALL_CURRENTS + ALL_VOLTAGES


class DNMAGNET:
    ALL_CURRENTS = ['N:INL{:02d}I'.format(i) for i in range(1, 19)]
    ALL_VOLTAGES = ['N:INL{:02d}I'.format(i) for i in range(1, 19)]
    ALL = ALL_CURRENTS + ALL_VOLTAGES


class OTHER:
    RF = ['N:IRFLLG', 'N:IRFMOD', 'N:IRFEAT', 'N:IRFEPC']
    KICKERS = ['N:IKPSV', 'N:IKPSVX', 'N:IKPSVD', 'N:IKPSH']


MASTER_STATE = CORRECTORS.ALL + QUADS.ALL + SKEWQUADS.ALL + SEXTUPOLES.ALL + OCTUPOLES.ALL + DNMAGNET.ALL + \
               OTHER.RF + OTHER.KICKERS
