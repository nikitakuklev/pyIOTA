import itertools


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

    ALL = H + V + S
    ALLA = HA + VA + SA


class DIPOLES:
    MAIN_BEND_I = ['N:IBEND']
    MAIN_BEND_V = ['N:IBENDV']

    FLUX_COMPENSATORS_I = [f'N:IBT{i}{s}I' for (i,s) in itertools.product(range(1, 5), ['R', 'L'])] #15
    FLUX_COMPENSATORS_V = [f'N:IBT{i}{s}V' for (i,s) in itertools.product(range(1, 5), ['R', 'L'])] #15

    ALL_I = MAIN_BEND_I + FLUX_COMPENSATORS_I
    ALL_V = MAIN_BEND_V + FLUX_COMPENSATORS_V


class CORRECTORS:
    __combfun = ['A1R', 'A2R', 'B1R', 'B2R', 'C1R', 'C2R', 'D1R', 'D2R', 'E1R', 'E2R',
                 'E2L', 'E1L', 'D2L', 'D1L', 'C2L', 'C1L', 'B2L', 'B1L', 'A2L', 'A1L']

    DIPOLE_TRIMS_I = ['N:IHM' + str(i) + 'LI' for i in range(1, 5)] + ['N:IHM' + str(i) + 'RI' for i in range(1, 5)]
    DIPOLE_TRIMS_V = ['N:IHM' + str(i) + 'LV' for i in range(1, 5)] + ['N:IHM' + str(i) + 'RV' for i in range(1, 5)]
    DIPOLE_TRIMS_ALL = DIPOLE_TRIMS_I + DIPOLE_TRIMS_V # 4000

    COMBINED_COILS_I = [i for sl in
                        [['N:I1' + k + 'I', 'N:I2' + k + 'I', 'N:I3' + k + 'I', 'N:I4' + k + 'I'] for k in __combfun] for
                        i in sl]
    COMBINED_COILS_V = [i for sl in
                      [['N:I1' + k + 'V', 'N:I2' + k + 'V', 'N:I3' + k + 'V', 'N:I4' + k + 'V'] for k in __combfun] for
                      i in sl]
    COMBINED_COILS_ALL = COMBINED_COILS_I + COMBINED_COILS_V #4000

    COMBINED_VIRTUAL = [i for sl in [['N:IV' + k + 'I', 'N:IH' + k + 'I'] for k in __combfun] for i in sl]

    OTHER_CORRECTORS_I = ['N:IBMPLI', 'N:IBMPRI'] #4000
    OTHER_CORRECTORS_V = ['N:IBMPLV', 'N:IBMPRV'] #4000

    LAMBERTSON_I = ['N:ILAM'] #15
    LAMBERTSON_V = ['N:ILAMV'] #15

    LAMBERTSON_HCORR_I = ['N:IHLAMI'] #4000
    LAMBERTSON_HCORR_V = ['N:IHLAMV'] #4000

    COMBINED_COILS_AND_DIPOLE_SHIMS_I = COMBINED_COILS_I + DIPOLE_TRIMS_I
    COMBINED_COILS_AND_DIPOLE_SHIMS_AND_OTHER_I = COMBINED_COILS_AND_DIPOLE_SHIMS_I + OTHER_CORRECTORS_I

    COMBINED_VIRTUAL_AND_DIPOLE_SHIMS_I = COMBINED_VIRTUAL + DIPOLE_TRIMS_I
    COMBINED_VIRTUAL_AND_DIPOLE_SHIMS_AND_OTHER_I = COMBINED_VIRTUAL_AND_DIPOLE_SHIMS_I + OTHER_CORRECTORS_I

    ALL_COILS = COMBINED_COILS_AND_DIPOLE_SHIMS_AND_OTHER_I
    ALL_VIRTUAL = COMBINED_VIRTUAL_AND_DIPOLE_SHIMS_AND_OTHER_I

    ALL = list(set(COMBINED_COILS_AND_DIPOLE_SHIMS_AND_OTHER_I +
                   COMBINED_VIRTUAL_AND_DIPOLE_SHIMS_AND_OTHER_I))


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
    ALL_VOLTAGES = ['N:INL{:02d}V'.format(i) for i in range(1, 19)]
    ALL = ALL_CURRENTS + ALL_VOLTAGES


class OTHER:
    RF = ['N:IRFLLF', 'N:IRFLLA', 'N:IRFMOD', 'N:IRFEAT', 'N:IRFEPC']
    KICKERS = ['N:IKPSV', 'N:IKPSH', 'N:IKPSVX', 'N:IKPSVD']
    BEAM_CURRENT = 'N:IBEAM'
    BEAM_CURRENT_AVERAGE = 'N:IBEAMA'
    WCM_PARAMS = ['N:IWCMBF','N:IWCMBR','N:IWCMI','N:IWCMBP','N:IRFEPA','N:IRFEPP']
    AUX_DEVICES = [BEAM_CURRENT] + [BEAM_CURRENT_AVERAGE] + WCM_PARAMS


class CONTROLS:
    TRIGGER_A5 = 'N:EA5TRG'  # $A5 timing event on reset
    TRIGGER_A6 = 'N:EA6TRG'  # $A6 timing event on reset
    VKICKER = 'N:IKPSV'
    HKICKER = 'N:IKPSH'
    CHIP_PLC = 'N:IDG'
    BPM_INJ_TRIGGER = 'N:IBINJ'
    BPM_ORB_TRIGGER = 'N:IBORB'
    BPM_CONFIG_DEVICE = 'N:IBPSTATD'
    FAST_LASER_INJECTOR = 'N:LGINJ'
    FAST_LASER_SHUTTER = 'N:LGXS'

    HKICKER_RESCHARGE = 'N:IKPSHR'
    HKICKER_TRIG = 'N:IKPSHT'
    HKICKER_ONOFF_DEVICES = [HKICKER_RESCHARGE, HKICKER_TRIG]

    VKICKER_RESCHARGE = 'N:IKPSVR'
    VKICKER_TRIG = 'N:IKPSVT'
    VKICKER_ONOFF_DEVICES = [VKICKER_RESCHARGE, VKICKER_TRIG]


MASTER_STATE_CURRENTS = DIPOLES.ALL_I + CORRECTORS.ALL + QUADS.ALL_CURRENTS + SKEWQUADS.ALL_CURRENTS + SEXTUPOLES.ALL_CURRENTS + \
                        OCTUPOLES.ALL_CURRENTS + DNMAGNET.ALL_CURRENTS + OTHER.RF + OTHER.KICKERS

MASTER_STATE = CORRECTORS.ALL + QUADS.ALL + SKEWQUADS.ALL + SEXTUPOLES.ALL + OCTUPOLES.ALL + DNMAGNET.ALL + \
               OTHER.RF + OTHER.KICKERS
