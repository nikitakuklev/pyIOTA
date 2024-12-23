import logging
import time
from copy import copy

import numpy as np
import pytest

from acnet import ArrayDevice, ArrayDeviceSet, DPM, DoubleDevice, DoubleDeviceSet, StatusDevice, \
    StatusDeviceSet
from acnet.adapters import ACNETSettings, AdapterManager, FakeAdapter
from acnet.data import ArrayDataResponse, StatusDataResponse
from acnet.drf2 import *
from acnet.drf2.drf2 import DiscreteRequest
from acnet.drf2.event import ClockEvent, DefaultEvent, ImmediateEvent, PeriodicEvent
from acnet.errors import ACNETError, ACNETTimeoutError

logger = logging.getLogger(__name__)


class DummyHTTPServer():
    def __init__(self):
        pass


def test_dpm():
    dpm = DPM()
    d = DoubleDevice('N:FAKE')
    ds = DoubleDeviceSet(name='test', members=[d], adapter=dpm)
    results = ds.read()
    assert len(results) == 1
    logger.info(f'{results}')


def test_dpm_live():
    dpm = DPM()
    d = DoubleDevice('Z:ACLTST')
    ds = DoubleDeviceSet(name='test', members=[d], adapter=dpm)
    results = ds.read()
    assert len(results) == 1
    logger.info(f'{results}')


def test_dpm_kerberos():
    import gssapi
    creds = gssapi.creds.Credentials(usage='initiate')


def test_device_array():
    d1 = ArrayDevice('N:IBB1RH')
    d2 = ArrayDevice('N:IBB1RV')
    d3 = StatusDevice('Z|ACLTST')
    d4 = DoubleDevice('M:OUTTMP')
    dd = DoubleDeviceSet(members=[d4], adapter=FakeAdapter())
    ds = StatusDeviceSet(members=[d3], adapter=FakeAdapter())
    dbpm = ArrayDeviceSet(members=[d1, d2],
                          enforce_array_length=1100,
                          adapter=FakeAdapter())

    dbpm.add(ArrayDevice('N:IBB1RS'))
    assert 'N:IBB1RS' in dbpm.devices
    assert list(dbpm.devices.keys()) == ['N:IBB1RH', 'N:IBB1RV', 'N:IBB1RS']

    dbpm.remove('N:IBB1RS')
    dbpm.remove('N:IBB1RV')
    assert list(dbpm.devices.keys()) == ['N:IBB1RH']
    dbpm.remove('N:IBB1RH')
    assert list(dbpm.devices.keys()) == []
    with pytest.raises(ValueError):
        dbpm.remove('FAKEBPM')

class TestFailures:
    def setup_method(self, method):
        fails = ['M:OUTTMP']
        pfails = {'M:OUTTMP2': 0.99, 'M:OUTTMP2.READING@I':0.99}
        self.fa = FakeAdapter(acnet_fails=fails, failed_partial=pfails)
        self.dpm = DPM(mock=self.fa)
        AdapterManager.default_adapter = self.fa

    def test_read_failed(self):
        df = DoubleDevice('M:OUTTMP')
        d = DoubleDevice('M:OUTHUM')

        assert d.read() == 0.1
        assert df.read() is None

        r = df.read(full=True)
        assert r.fc == 99
        assert r.err == -99

    def test_read_partial_fail(self):
        df = DoubleDevice('M:OUTTMP2')

        while True:
            try:
                r = df.read()
                break
                #assert r.fc == 66
                #assert r.err == -66
            except ACNETTimeoutError:
                pass

    def test_read_failed_dpm(self):
        df = DoubleDevice('M:OUTTMP2', adapter=self.dpm)

        with pytest.raises(ACNETTimeoutError):
            r = df.read()

        with ACNETSettings.Context(n_read=200):
            r = df.read()
            assert r == 0.1

        with pytest.raises(ACNETTimeoutError):
            r = df.read()



class TestDevices:
    def setup_method(self, method):
        self.fa = FakeAdapter()

    def test_devices(self):
        AdapterManager.default_adapter = self.fa
        #print(f'{AdapterManager.default_adapter=}')
        #print(f'{self.fa=}')
        d = DoubleDevice('N:BLABLA')
        r = d.read()
        with pytest.raises(ValueError):
            d = DoubleDevice('BAD:BLABLA')

        d.set(0.5)
        r = 0.5
        r2 = d.read_setpoint()
        assert 0.5 == r2

        r3 = d.read()
        assert np.abs(r - r3) < 1e-4

        r4 = d.read_readback()
        assert r3 == r4
        assert np.abs(r - r4) < 1e-4

        s = StatusDevice('Z|ACLTST2')
        s.read()
        assert s.is_on()

        s.set_off()
        s.read()
        assert s.is_off()
        assert s.off


    def test_devicesets(self):
        d4 = DoubleDevice('M:OUTTMP')
        d5 = DoubleDevice('M:OUTHUM')
        dd = DoubleDeviceSet(members=[d4, d5], adapter=self.fa)


class TestFakeAdapter:
    def setup_method(self, method):
        d1 = ArrayDevice('N:IBB1RH')
        d2 = ArrayDevice('N:IBB1RV')
        d3 = StatusDevice('Z|ACLTST')
        d3b = StatusDevice('Z|ACLTST2')
        d4 = DoubleDevice('M:OUTTMP')
        d5 = DoubleDevice('M:OUTHUM')
        fa = FakeAdapter()
        dd = DoubleDeviceSet(members=[d4, d5], adapter=fa)
        dds = DoubleDeviceSet(members=[d4.copy(), d5.copy()], adapter=fa, settings=True)
        dd_rb = DoubleDeviceSet(members=['M:BLABLAA'], adapter=fa)
        dd_setp = DoubleDeviceSet(members=['M:BLABLAA'], adapter=fa, settings=True)
        ds = StatusDeviceSet(members=[d3, d3b], adapter=fa)
        dbpm = ArrayDeviceSet(members=[d1, d2],
                              enforce_array_length=1100,
                              adapter=fa)
        self.dbpm = dbpm
        self.dd = dd
        self.dds = dds
        self.dd_rb = dd_rb
        self.dd_setp = dd_setp
        self.ds = ds

    def test_read_array(self):
        # {'N:IBB1RH', 'N:IBB1RV'}
        r = self.dbpm.read()
        assert len(r) == 2
        assert all(len(v) == 10 for v in r)
        assert all(isinstance(v, np.ndarray) for v in r)

        r2 = self.dbpm.read(full=True)
        assert isinstance(r2[0], ArrayDataResponse)

        device = list(self.dbpm.devices.values())[0]
        print(device.read(adapter=self.dbpm.adapter)[0:5])
        assert device.value_string is not None

    def test_set_array(self):
        r = self.dbpm.read()
        with pytest.raises(ACNETError):
            r2 = self.dbpm.set([[1.0, 2.0], [3.0, 4.0]])

    def test_read_status(self):
        r = self.ds.read()
        assert len(r) == 2
        assert all(v == {'on': True, 'ready': True, 'remote': False, 'positive': False, 'ramp': True
                         } for v in r)

        r2 = self.ds.read(full=True)
        assert isinstance(r2[0], StatusDataResponse)

    def test_set_status(self):
        r = self.ds.read()
        r2 = self.ds.set(['OFF', 'OFF'])
        rb = self.ds.read()
        assert all(
                v == {'on': False, 'ready': True, 'remote': False, 'positive': False, 'ramp': True
                      } for v in rb)

    def test_set_double(self):
        r = self.dd.read()
        r2 = self.dd.set([5.0, 5.5])
        rb = self.dds.read()
        assert r2 == 2
        assert rb == [5.0, 5.5]
        r2 = self.dd.set([5.1, 5.6])
        assert r2 == 2
        rb = self.dd.read()
        sp = self.dds.read()
        assert sp == [5.1, 5.6]
        assert np.all(np.isclose([5.1, 5.6], rb, atol=1e-4, rtol=0.0))

    def test_ts(self):
        r = self.dbpm.read()
        tbefore = [d.last_update for k, d in self.dbpm.devices.items()]
        time.sleep(0.01)
        r = self.dbpm.read()
        for t1, t2 in zip(tbefore, [d.last_update for k, d in self.dbpm.devices.items()]):
            assert t2 > t1


def test_drf_parse_full():
    # print(PATTERN_FULL.match('N:I2B1RI'))
    f = parse_request
    d = 'N:I2B1RI'
    r = f('N:I2B1RI').parts
    assert r == ('N:I2B1RI', DRF_PROPERTY.READING, parse_range(None),
                 DRF_FIELD.SCALED, parse_event(None))
    assert f(d).to_canonical() == 'N:I2B1RI.READING'
    assert f(d).to_qualified() == 'N:I2B1RI'

    d = 'N_I2B1RI'
    r = f(d).parts
    assert r == ('N:I2B1RI', DRF_PROPERTY.SETTING, parse_range(None),
                 DRF_FIELD.SCALED, parse_event(None))
    assert f(d).to_canonical() == 'N:I2B1RI.SETTING'
    assert f(d).to_qualified() == 'N_I2B1RI'

    d = 'N|I2B1RI'
    r = f(d).parts
    assert r == ('N:I2B1RI', DRF_PROPERTY.STATUS, parse_range(None),
                 DRF_FIELD.ALL, parse_event(None))
    assert f(d).to_canonical() == 'N:I2B1RI.STATUS'
    assert f(d).to_qualified() == 'N|I2B1RI'

    d = 'N:I2B1RI@p,500'
    r = f(d).parts
    assert r == ('N:I2B1RI', DRF_PROPERTY.READING, parse_range(None),
                 DRF_FIELD.SCALED, PeriodicEvent('p,500', 'P'))
    assert f(d).to_canonical() == 'N:I2B1RI.READING@p,500'
    assert f(d).to_qualified() == 'N:I2B1RI@p,500'

    d = 'N_I2B1RI@p,500'
    r = f(d).parts
    assert r == ('N:I2B1RI', DRF_PROPERTY.SETTING, parse_range(None),
                 DRF_FIELD.SCALED, PeriodicEvent('p,500', 'P'))
    assert f(d).to_canonical() == 'N:I2B1RI.SETTING@p,500'
    assert f(d).to_qualified() == 'N_I2B1RI@p,500'

    d = 'N:I2B1RI[:]@p,500'
    r = f(d).parts
    assert r == ('N:I2B1RI', DRF_PROPERTY.READING, DRF_RANGE('full', None, None),
                 DRF_FIELD.SCALED, PeriodicEvent('p,500', 'P'))
    assert f(d).to_canonical() == 'N:I2B1RI.READING[:]@p,500'
    assert f(d).to_qualified() == 'N:I2B1RI[:]@p,500'

    d = 'N:I2B1RI[]@p,500'
    r = f(d).parts
    assert r == ('N:I2B1RI', DRF_PROPERTY.READING, DRF_RANGE('full', None, None),
                 DRF_FIELD.SCALED, PeriodicEvent('p,500', 'P'))
    assert f(d).to_canonical() == 'N:I2B1RI.READING[:]@p,500'
    assert f(d).to_qualified() == 'N:I2B1RI[:]@p,500'

    d = 'N:I2B1RI[:2048]@I'
    r = f(d).parts
    assert r == ('N:I2B1RI', DRF_PROPERTY.READING, DRF_RANGE('std', None, 2048),
                 DRF_FIELD.SCALED, ImmediateEvent('I', 'I'))
    assert f(d).to_canonical() == 'N:I2B1RI.READING[:2048]@I'
    assert f(d).to_qualified() == 'N:I2B1RI[:2048]@I'

    d = 'N:I2B1RI.SETTING[50:]@I'
    r = f(d).parts
    assert r == ('N:I2B1RI', DRF_PROPERTY.SETTING, DRF_RANGE('std', 50, None),
                 DRF_FIELD.SCALED, ImmediateEvent('I', 'I'))
    assert f(d).to_canonical() == 'N:I2B1RI.SETTING[50:]@I'
    assert f(d).to_qualified() == 'N_I2B1RI[50:]@I'

    d = 'N_I2B1RI.SETTING[50:]@I'
    r = f(d).parts
    assert r == ('N:I2B1RI', DRF_PROPERTY.SETTING, DRF_RANGE('std', 50, None),
                 DRF_FIELD.SCALED, ImmediateEvent('I', 'I'))
    assert f(d).to_canonical() == 'N:I2B1RI.SETTING[50:]@I'
    assert f(d).to_qualified() == 'N_I2B1RI[50:]@I'

    d = 'N_I2B1RI.SETTING[50]@e,AE,e,1000'
    r = f(d).parts
    assert r == ('N:I2B1RI', DRF_PROPERTY.SETTING, DRF_RANGE('single', 50, None),
                 DRF_FIELD.SCALED, ImmediateEvent('I', 'I'))
    assert f(d).to_canonical() == 'N:I2B1RI.SETTING[50]@e,AE,e,1000'
    assert f(d).to_qualified() == 'N_I2B1RI[50]@e,AE,e,1000'

    d = 'Z:CACHE[50:]'
    r = f(d).parts
    assert r == ('Z:CACHE', DRF_PROPERTY.READING, DRF_RANGE('std', 50, None),
                 DRF_FIELD.SCALED, None)
    assert f(d).to_canonical() == 'Z:CACHE.READING[50:]'
    assert f(d).to_qualified() == 'Z:CACHE[50:]'

    d = 'E:TRTGTD@e,AE,e,1000'
    r = f(d).parts
    assert r == ('E:TRTGTD', DRF_PROPERTY.READING, parse_range(None),
                 DRF_FIELD.SCALED, ClockEvent('e,AE,e,1000', 'E'))
    assert f(d).to_canonical() == 'E:TRTGTD.READING@e,AE,e,1000'
    assert f(d).to_qualified() == 'E:TRTGTD@e,AE,e,1000'


def test_drf_device_parse():
    r = parse_device('N:I2B1RI')
    assert r.canonical_string == 'N:I2B1RI'

    r = parse_device('N_I2B1RI')
    assert r.canonical_string == 'N:I2B1RI'

    r = parse_device('N:I2B1RI@p,1000')
    assert r.canonical_string == 'N:I2B1RI@p,1000'

    r = parse_device('N_I2B1RI@p,1000')
    assert r.canonical_string == 'N:I2B1RI@p,1000'

    r = get_qualified_device('N:I2B1RI', DRF_PROPERTY.SETTING)
    assert r == 'N_I2B1RI'


def test_drf_sorting():
    devs = ['N:I2B1RI', 'N_I2B1RI', 'N:I2B1RI@p,1000', 'N:I2B1RI[]@p,1000']
    d, s, a = AdapterManager.sort_drf_strings(devs)
