import numpy as np

from sixdsim import Knob, parse_knobs


def test_knob():
    k1 = Knob.from_dict({'x1': 0.1, 'x2': 3.0})
    k2 = Knob.from_dict({'x1': 0.8, 'x2': -30.0})
    assert k1.vars['x1'].acnet_var == 'x1'

    ka1 = Knob.from_dict({'x2': 10.0, '$N_BLABLA': 0.8, '$N_BLA2': -30.0})
    assert list(ka1.vars.keys()) == ['x2', '$N_BLABLA', '$N_BLA2']

    # assert list(ka1.vars.keys()) == ['N:BLABLA', 'N:BLA2']

    kminus = k1 - ka1
    assert len(kminus.vars) == 4
    assert kminus.vars['x2'].value == -7.0
    assert kminus.vars['$N_BLABLA'].value == -0.8
    assert kminus.vars['$N_BLA2'].value == 30.0

    kminus2 = 1 - ka1
    assert len(kminus2.vars) == 3
    np.allclose([kminus2.vars[x].value for x in ['x2', '$N_BLABLA', '$N_BLA2']],
                [-9.0, 0.2, 31.0], rtol=0)

    kminus3 = -(ka1 - 1)
    assert len(kminus3.vars) == 3
    np.allclose([kminus3.vars[x].value for x in ['x2', '$N_BLABLA', '$N_BLA2']],
                [-9.0, 0.2, 31.0], rtol=0)

    kminus4 = -(-2 * ka1 - 1) / 2
    assert len(kminus4.vars) == 3
    np.allclose([kminus4.vars[x].value for x in ['x2', '$N_BLABLA', '$N_BLA2']],
                [21 / 2, 2.6 / 2, -59 / 2], rtol=0)

    k3 = k1 + k2
    assert len(k3.vars) == 2
    assert k3.vars['x1'].value == 0.9
    assert k3.vars['x2'].value == -27.0

    k4 = k1 + ka1
    assert len(k4.vars) == 4
    assert k4.vars['x1'].value == 0.1
    assert k4.vars['x2'].value == 13.0
    assert k4.vars['$N_BLABLA'].value == 0.8

    k5 = k1.copy()
    k5 = k5.only_keep_shared(k2)
    assert len(k5.vars) == 2

    k6 = k1.copy()
    k6 = k6.only_keep_shared(ka1)
    assert len(k6.vars) == 1

    k7 = k1.copy()
    k7.shuffle()
    assert set(k7.vars.keys()) == {'x1', 'x2'}

    assert k1.vars == k1.copy().vars


def test_knob_parse():
    knobs = parse_knobs('files/20230814_IOTA_und_150MeV_2_orbCorrInTMP.knobs')
    assert len(knobs) == 15
    assert 'IOTA_BR_dY_pm1mm' in knobs.keys()
    assert len(knobs['IOTA_BR_dY_pm1mm']) == 6

    knob = knobs['IOTA_Trims_tmp']
    assert len(knob) == 49

    def check_no_dollar(knob):
        for k in knob.vars.keys():
            assert '$' not in k

    knob_abs_offset = Knob.from_dict({'blabababababa': 0.5, 'N:IHC1RI': 0.7})

    knobc = knob.copy()
    assert len(knobc.vars) == 49
    check_no_dollar(knobc)
    knobc.only_keep_shared(knob_abs_offset)
    assert knobc.vars.keys() == {'N:IHC1RI'}

    knob2 = knob + knob_abs_offset
    check_no_dollar(knob2)
    assert len(knob2.vars) == 50
    assert knob2['N:IHC1RI'] == knob['N:IHC1RI'] + 0.7
    assert knob2['blabababababa'] == 0.5

    knob3 = knob + 0.5
    check_no_dollar(knob3)
    assert len(knob3.vars) == 49
    assert set(knob3.vars.keys()) == set(knob.vars.keys())
    for k, v in knob3.vars.items():
        assert v.value == knob.vars[k].value + 0.5

    knob3 = knob * 0.5
    check_no_dollar(knob3)
    assert len(knob3.vars) == 49
    for k, v in knob3.vars.items():
        assert v.value == knob.vars[k].value * 0.5

    k = knob - (knob / 2)
    check_no_dollar(k)
    assert len(k.vars) == 49
    for k, v in k.vars.items():
        assert v.value == knob.vars[k].value * 0.5

    knob_delta = knob - knob3
    check_no_dollar(knob_delta)
    assert len(knob_delta.vars) == 49
    for k, v in knob_delta.vars.items():
        assert v.value == knob.vars[k].value - knob3.vars[k].value

    knob_prune = (knob + 1e-4) - knob
    knob_prune.prune(tol=1e-3)
    assert len(knob_prune) == 0

    knob_prune = (knob + 1e-4) - knob
    knob_prune.vars['N:IHC1RI'].value += 1.0
    knob_prune.prune(tol=1e-3)
    assert len(knob_prune) == 1
