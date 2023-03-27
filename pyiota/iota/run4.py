import itertools


class BPMS:

    @staticmethod
    def to_acnet(names: list):
        return ['N:' + n for n in names]

    @staticmethod
    def to_sixdsim(names: list) -> list:
        assert [n[:3] == 'N:I' for n in names]  # acnet names
        return [n[3:] for n in names]

    @staticmethod
    def from_sixdsim(names: list) -> list:
        assert [len(n) == 4 or len(n) == 5 for n in names]  # i.e. BC2L or BC2LH
        return ['N:I' + n for n in names]

    @staticmethod
    def from_acnet(names: list):
        return [n[2:] if n[0:2] == 'N:' else n for n in names]

    @staticmethod
    def add_axis(names: list, axis: str):
        assert axis in ['H', 'V', 'S']
        return [n + axis for n in names]

    LATTICE_NAMES = ['IBB2R', 'IBC1R', 'IBC2R', 'IBD1R', 'IBD2R', 'IBE1R', 'IBE2R', 'IBE2L',
                     'IBE1L', 'IBD2L', 'IBD1L',
                     'IBC2L', 'IBC1L', 'IBB2L', 'IBB1L', 'IBA3L', 'IBA2L', 'IBA1C', 'IBA2R',
                     'IBA3R', 'IBB1R']

    ACNET_NAMES = to_acnet.__func__(LATTICE_NAMES)

    ACNET_NAMES_LATTICE_ORDER = to_acnet.__func__(
            ['IBA1C', 'IBA2R', 'IBA3R', 'IBB1R', 'IBB2R', 'IBC1R', 'IBC2R', 'IBD1R', 'IBD2R',
             'IBE1R', 'IBE2R', 'IBE2L', 'IBE1L', 'IBD2L', 'IBD1L',
             'IBC2L', 'IBC1L', 'IBB2L', 'IBB1L', 'IBA3L', 'IBA2L'])

    H = add_axis.__func__(LATTICE_NAMES, 'H')
    HA = to_acnet.__func__(H)

    V = add_axis.__func__(LATTICE_NAMES, 'V')
    VA = to_acnet.__func__(V)

    S = add_axis.__func__(LATTICE_NAMES, 'S')
    SA = to_acnet.__func__(S)

    ALL = H + V + S
    ALLA = HA + VA + SA

    RAW_A = 'N:IBPMRA'
    RAW_B = 'N:IBPMRB'
    RAW_C = 'N:IBPMRB'
    RAW_D = 'N:IBPMRB'
    RAW_ALL = [RAW_A, RAW_B, RAW_C, RAW_D]


class DIPOLES:
    MAIN_BEND_I = ['N:IBEND']
    MAIN_BEND_V = ['N:IBENDV']

    FLUX_COMPENSATORS_I = [f'N:IBT{i}{s}I' for (i, s) in
                           itertools.product(range(1, 5), ['R', 'L'])]  # 15
    FLUX_COMPENSATORS_V = [f'N:IBT{i}{s}V' for (i, s) in
                           itertools.product(range(1, 5), ['R', 'L'])]  # 15

    ALL_I = MAIN_BEND_I + FLUX_COMPENSATORS_I
    ALL_V = MAIN_BEND_V + FLUX_COMPENSATORS_V
    ALL = ALL_I + ALL_V


class CORRECTORS:
    _combfun = ['A1R', 'A2R', 'B1R', 'B2R', 'C1R', 'C2R', 'D1R', 'D2R', 'E1R', 'E2R',
                'E2L', 'E1L', 'D2L', 'D1L', 'C2L', 'C1L', 'B2L', 'B1L', 'A2L', 'A1L']

    kHSQ = (0.1 - 0.034645) * 1.3  # calibration for current to field for panofsky magnets

    DIPOLE_TRIMS_I = ['N:IHM' + str(i) + 'LI' for i in range(1, 5)] + ['N:IHM' + str(i) + 'RI' for i
                                                                       in range(1, 5)]
    DIPOLE_TRIMS_V = ['N:IHM' + str(i) + 'LV' for i in range(1, 5)] + ['N:IHM' + str(i) + 'RV' for i
                                                                       in range(1, 5)]
    DIPOLE_TRIMS_ALL = DIPOLE_TRIMS_I + DIPOLE_TRIMS_V  # 4000

    COMBINED_COILS_I = [i for sl in [['N:I1' + k + 'I', 'N:I2' + k + 'I',
                                      'N:I3' + k + 'I', 'N:I4' + k + 'I'] for k in _combfun] for
                        i in sl]
    COMBINED_COILS_V = [i for sl in [['N:I1' + k + 'V', 'N:I2' + k + 'V',
                                      'N:I3' + k + 'V', 'N:I4' + k + 'V'] for k in _combfun] for
                        i in sl]
    COMBINED_COILS_ALL = COMBINED_COILS_I + COMBINED_COILS_V  # 4000

    VIRTUAL_H = [i for sl in [['N:IH' + k + 'I'] for k in _combfun] for i in sl]
    VIRTUAL_V = [i for sl in [['N:IV' + k + 'I'] for k in _combfun] for i in sl]
    COMBINED_VIRTUAL = [i for sl in [['N:IV' + k + 'I', 'N:IH' + k + 'I'] for k in _combfun] for i
                        in sl]

    OTHER_CORRECTORS_I = ['N:IBMPLI', 'N:IBMPRI']  # 4000
    OTHER_CORRECTORS_V = ['N:IBMPLV', 'N:IBMPRV']  # 4000

    LAMBERTSON_I = ['N:ILAM']  # 15
    LAMBERTSON_V = ['N:ILAMV']  # 15

    LAMBERTSON_HCORR_I = ['N:IHLAMI']  # 4000
    LAMBERTSON_HCORR_V = ['N:IHLAMV']  # 4000

    COMBINED_COILS_AND_DIPOLE_SHIMS_I = COMBINED_COILS_I + DIPOLE_TRIMS_I
    COMBINED_COILS_AND_DIPOLE_SHIMS_AND_OTHER_I = COMBINED_COILS_AND_DIPOLE_SHIMS_I +\
                                                  OTHER_CORRECTORS_I

    COMBINED_VIRTUAL_AND_DIPOLE_SHIMS_I = COMBINED_VIRTUAL + DIPOLE_TRIMS_I
    COMBINED_VIRTUAL_AND_DIPOLE_SHIMS_AND_OTHER_I = COMBINED_VIRTUAL_AND_DIPOLE_SHIMS_I +\
                                                    OTHER_CORRECTORS_I

    ALL_COILS = COMBINED_COILS_AND_DIPOLE_SHIMS_AND_OTHER_I
    ALL_VIRTUAL = COMBINED_VIRTUAL_AND_DIPOLE_SHIMS_AND_OTHER_I

    ALL = list(set(COMBINED_COILS_AND_DIPOLE_SHIMS_AND_OTHER_I +
                   COMBINED_VIRTUAL_AND_DIPOLE_SHIMS_AND_OTHER_I))


class QUADS:
    __quad_names = ['A1R', 'A2R', 'A3R', 'A4R', 'B1R', 'B2R', 'B3R', 'B4R', 'B5R', 'B6R',
                    'C1R', 'C2R', 'C3R', 'D1R', 'D2R', 'D3R', 'D4R', 'E1R', 'E2R',
                    'E3C',
                    'E2L', 'E1L', 'D4L', 'D3L', 'D2L', 'D1L', 'C3L', 'C2L', 'C1L', 'B6L',
                    'B5L', 'B4L', 'B3L', 'B2L', 'B1L', 'A4L', 'A3L', 'A2L', 'A1L']
    ALL_CURRENTS = ['N:IQ' + str(i) + 'I' for i in __quad_names]
    ALL_VOLTAGES = ['N:IQ' + str(i) + 'V' for i in __quad_names]
    ALL = ALL_CURRENTS + ALL_VOLTAGES


class SKEWQUADS:
    __skew_names = ['A1R', 'A2R', 'B1R', 'B2R', 'C1R', 'C2R', 'D1R', 'D2R', 'E1R', 'E2R',
                    'E2L', 'E1L', 'D2L', 'D1L', 'C2L', 'C1L', 'B2L', 'B1L', 'A2L', 'A1L']
    ALL_CURRENTS = ['N:IK' + str(i) + 'I' for i in __skew_names]
    ALL_VOLTAGES = ['N:IK' + str(i) + 'V' for i in __skew_names]
    ALL = ALL_CURRENTS + ALL_VOLTAGES


class SEXTUPOLES:
    __sext_names = ['A1R', 'C1R', 'C2R', 'D1R', 'E1R', 'E2R',
                    'E2L', 'E1L', 'D1L', 'C2L', 'C1L', 'A2L']
    #__sext_names_v2 = ['C1R', 'C2R', 'C1L', 'C2L']
    ALL_CURRENTS = ['N:IS' + str(i) + 'I' for i in __sext_names] #+ \
                   #['N:TS' + str(i) + 'I' for i in __sext_names_v2]
    ALL_VOLTAGES = ['N:IS' + str(i) + 'V' for i in __sext_names] #+ \
                   #['N:TS' + str(i) + 'V' for i in __sext_names_v2]
    ALL = ALL_CURRENTS + ALL_VOLTAGES


class OCTUPOLES:
    ALL_CURRENTS = ['N:IO' + str(i) + 'LI' for i in range(1, 19)] + ['N:OB9L6I', 'N:O4L6I',
                                                                     'N:O14L6I']
    ALL_VOLTAGES = ['N:IO' + str(i) + 'LV' for i in range(1, 19)] + ['N:OB9L6V', 'N:O4L6V',
                                                                     'N:O14L6V']
    ALL_CURRENTS_ACTIVE = ['N:IO' + str(i) + 'LI' for i in range(1, 18)]
    ALL_VOLTAGES_ACTIVE = ['N:IO' + str(i) + 'LV' for i in range(1, 18)]
    ALL = ALL_CURRENTS + ALL_VOLTAGES


class DNMAGNET:
    ALL_CURRENTS = ['N:INL{:02d}I'.format(i) for i in range(1, 19)]
    ALL_VOLTAGES = ['N:INL{:02d}V'.format(i) for i in range(1, 19)]
    ALL = ALL_CURRENTS + ALL_VOLTAGES


class CONTROLS:
    TRIGGER_A5 = 'N:EA5TRG'  # $A5 timing event on reset
    TRIGGER_A6 = 'N:EA6TRG'  # $A6 timing event on reset
    VKICKER = 'N:IKPSV'
    VKICKER_USER_KNOB = 'N:IKPSVX'
    HKICKER = 'N:IKPSH'
    CHIP_PLC = 'N:IDG'
    BPM_INJ_TRIGGER = 'N:IBINJ'
    BPM_ORB_TRIGGER = 'N:IBORB'
    BPM_CONFIG_DEVICE = 'N:IBPSTATD'
    BPM_RAWREAD_INDEX = 'N:IBPMIN'
    FAST_LASER_INJECTOR = 'N:LGINJ'
    FAST_LASER_SHUTTER = 'N:LGXS'

    HKICKER_RESCHARGE = 'N:IKPSHR'
    HKICKER_TRIG = 'N:IKPSHT'
    HKICKER_ONOFF_DEVICES = [HKICKER_RESCHARGE, HKICKER_TRIG]

    VKICKER_RESCHARGE = 'N:IKPSVR'
    VKICKER_TRIG = 'N:IKPSVT'
    VKICKER_ONOFF_DEVICES = [VKICKER_RESCHARGE, VKICKER_TRIG]


class OTHER:
    RF = ['N:IRFLLF', 'N:IRFLLA', 'N:IRFMOD', 'N:IRFEAT', 'N:IRFEPC']
    KICKERS = ['N:IKPSV', 'N:IKPSH', 'N:IKPSVX', 'N:IKPSVD']
    BEAM_CURRENT = 'N:IBEAM'
    BEAM_CURRENT_AVERAGE = 'N:IBEAMA'
    WCM_PARAMS = ['N:IWCMBF', 'N:IWCMBR', 'N:IWCMI', 'N:IWCMBP', 'N:IRFEPA', 'N:IRFEPP']
    BPM_ATN = ['Z:RP2ATN']
    AUX_DEVICES = [BEAM_CURRENT] + [BEAM_CURRENT_AVERAGE] + WCM_PARAMS + [CONTROLS.TRIGGER_A5] +\
                  [CONTROLS.TRIGGER_A6] + BPM_ATN


MASTER_STATUS_DEVICES = CONTROLS.VKICKER_ONOFF_DEVICES + \
                        CONTROLS.HKICKER_ONOFF_DEVICES + \
                        [CONTROLS.HKICKER] + \
                        [CONTROLS.VKICKER] + \
                        [CONTROLS.CHIP_PLC]

MASTER_STATE_CURRENTS = DIPOLES.ALL_I + CORRECTORS.ALL + \
                        QUADS.ALL_CURRENTS + \
                        SKEWQUADS.ALL_CURRENTS + \
                        SEXTUPOLES.ALL_CURRENTS + \
                        OCTUPOLES.ALL_CURRENTS + \
                        DNMAGNET.ALL_CURRENTS + \
                        OTHER.RF + OTHER.KICKERS

MASTER_STATE = CORRECTORS.ALL + \
               QUADS.ALL + \
               SKEWQUADS.ALL + \
               SEXTUPOLES.ALL + \
               OCTUPOLES.ALL + \
               DNMAGNET.ALL + \
               OTHER.RF + OTHER.KICKERS
