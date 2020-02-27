from unittest import TestCase
import time
import numpy as np
import unittest

from .frontends import *
import pyIOTA.iota as iota


class TestWrap(TestCase):
    sync = True

    def setUp(self):
        t = 2
        if t == 0:
            d1 = BPMDevice('N:IBB1RH', '')
            d2 = BPMDevice('N:IBB1RV', '')
            dbpm = BPMDeviceSet('bpmset', members=[d1, d2], enforce_array_length=1100)
            dbpm.adapter = ACL(fallback=True)
            self.dbpm = dbpm
        elif t == 1:
            d3 = BPMDevice('N:IBB1RH', '')
            d4 = BPMDevice('N:IBB1RV', '')
            dbpm = BPMDeviceSet('bpmset', members=[d3, d4], enforce_array_length=1100)
            dbpm.adapter = ACL()
            self.dbpm = dbpm
        elif t == 2:
            d1 = BPMDevice('N:IBB1RH')
            d2 = BPMDevice('N:IBB1RV')
            d3 = StatusDevice('Z:ACLTST')
            d4 = DoubleDevice('M:OUTTMP')
            dd = DeviceSet('bpmset', members=[d4], adapter=FakeAdapter())
            ds = StatusDeviceSet('bpmset', members=[d3], adapter=FakeAdapter())
            dbpm = BPMDeviceSet('bpmset', members=[d1, d2], enforce_array_length=1100, adapter=FakeAdapter())
            self.dbpm = dbpm
            self.dd = dd
            self.ds = ds


class TestWrap2(TestCase):
    sync = True

    def setUp(self):
        t = 0
        if t == 0:
            d1 = BPMDevice('N:IBB1RH', '')
            d2 = BPMDevice('N:IBB1RV', '')
            dbpm = BPMDeviceSet('bpmset', members=[d1, d2], enforce_array_length=1100)
            dbpm.adapter = ACL(fallback=True)
            self.dbpm = dbpm
        elif t == 1:
            d3 = BPMDevice('N:IBB1RH', '')
            d4 = BPMDevice('N:IBB1RV', '')
            dbpm = BPMDeviceSet('bpmset', members=[d3, d4], enforce_array_length=1100)
            dbpm.adapter = ACL()
            self.dbpm = dbpm
        elif t == 2:
            d1 = BPMDevice('N:IBB1RH')
            d2 = BPMDevice('N:IBB1RV')
            d3 = StatusDevice('Z:ACLTST')
            d4 = DoubleDevice('M:OUTTMP')
            dd = DeviceSet('bpmset', members=[d4], adapter=FakeAdapter())
            ds = StatusDeviceSet('bpmset', members=[d3], adapter=FakeAdapter())
            dbpm = BPMDeviceSet('bpmset', members=[d1, d2], enforce_array_length=1100, adapter=FakeAdapter())
            self.dbpm = dbpm
            self.dd = dd
            self.ds = ds


class TestDeviceSet(TestWrap):
    def test_device_set(self):
        print()
        devs = list(self.dbpm)
        self.assertEqual(devs, ['N:IBB1RH', 'N:IBB1RV'])

    def test_start_oneshot(self):
        print()
        d1 = list(self.dbpm.devices.values())[0]
        print(f'Old ts: {d1.last_update} | {d1.value[0:5]}')
        old_ts = d1.last_update
        self.dbpm.readonce()
        print(f'New ts: {d1.last_update} | {d1.value[0:5]}')
        self.assertTrue(old_ts < d1.last_update)

    def test_start_oneshot2(self):
        print()
        devs = [BPMDevice(b) for b in iota.BPMS.HA]
        dstest = BPMDeviceSet(name='bpms', members=devs, enforce_array_length=1000)
        dstest.adapter = ACL(fallback=False)
        dstest.readonce()
        print(devs)

    def test_check_acquisition_supported(self):
        print()
        self.assertTrue(self.dbpm.check_acquisition_supported('oneshot'))
        self.assertRaises(Exception, self.dbpm.check_acquisition_supported, 'fakeshot')
        self.assertTrue(self.dbpm.check_acquisition_available('oneshot'))


class TestBPMDeviceSet(TestWrap):
    def test_add(self):
        print()
        self.dbpm.add(BPMDevice('N:IBB1RS', ''))
        self.assertTrue('N:IBB1RS' in self.dbpm.devices)
        self.assertEqual(list(self.dbpm.devices.keys()), ['N:IBB1RH', 'N:IBB1RV', 'N:IBB1RS'])

    def test_remove(self):
        print()
        self.dbpm.remove('N:IBB1RV')
        self.assertEqual(list(self.dbpm.devices.keys()), ['N:IBB1RH'])
        self.dbpm.remove('N:IBB1RH')
        self.assertEqual(list(self.dbpm.devices.keys()), [])
        self.assertRaises(ValueError, self.dbpm.remove, 'FAKEBPM')


class TestFakeAdapter(TestWrap):
    def test_readonce(self):
        print()
        device = list(self.dbpm.devices.values())[0]
        print(device.read(adapter=self.dbpm.adapter)[0:5])
        self.assertTrue(device.value_string is not None)
        self.assertTrue(device.value_tuple)

    def test_readonce2(self):
        # print(self.ds.devices)
        # print([d.name for d in self.ds.devices.values()])
        self.assertTrue(self.dbpm.readonce() == 2)
        tbefore = [d.last_update for k, d in self.dbpm.devices.items()]
        time.sleep(0.01)
        self.assertTrue(self.dbpm.readonce() == 2)
        # for k,d in self.ds.devices.items():
        #     print(k, d.value)
        for t1, t2 in zip(tbefore, [d.last_update for k, d in self.dbpm.devices.items()]):
            self.assertGreater(t2, t1)
        for k, d in self.dbpm.devices.items():
            self.assertTrue(len(d.value) == 100)
            self.assertTrue(isinstance(d.value, np.ndarray))

    @unittest.expectedFailure
    def test_set(self):
        self.dbpm.set([0.1, 0.2])

    def test_set2(self):
        self.dd.set([0.1])
        self.assertTrue(self.dd.readonce(settings=True) == 1)
        dev = list(self.dd.devices.values())[0]
        self.assertEqual(0.1, dev.value)

    def test_set3(self):
        print()
        dev = list(self.ds.devices.values())[0]
        self.assertFalse(dev.on)
        self.ds.set(['ON'])
        self.assertTrue(self.ds.readonce(settings=True) == 1)
        print(dev)
        self.assertTrue(dev.on)
        self.ds.set(['RESET'])
        self.assertTrue(self.ds.readonce(settings=True) == 1)
        self.assertTrue(dev.ready)


class TestACLAsync(TestWrap):
    def test_start_oneshot(self):
        d1 = list(self.dbpm.devices.values())[0]
        print(f'\nOld ts: {d1.last_update} | {d1.value}')
        self.dbpm.readonce()
        print(f'New ts: {d1.last_update} | {d1.value}')


class TestACL(TestWrap2):
    def test_del(self):
        del self.dbpm.adapter

    def test_readonce(self):
        print()
        print(f'TestACL-{self.dbpm.devices}-{self.dbpm.adapter.supported}-{self.dbpm.adapter.name}')
        device = list(self.dbpm.devices.values())[0]
        print(device.read()[0:5])
        self.assertTrue(device.value_string is not None)

    def test_readonce2(self):
        # print(self.ds.devices)
        # print([d.name for d in self.ds.devices.values()])
        self.assertTrue(self.dbpm.readonce() == 2)
        tbefore = [d.last_update for k, d in self.dbpm.devices.items()]
        time.sleep(0.01)
        self.assertTrue(self.dbpm.readonce() == 2)
        # for k,d in self.ds.devices.items():
        #     print(k, d.value)
        for t1, t2 in zip(tbefore, [d.last_update for k, d in self.dbpm.devices.items()]):
            self.assertGreater(t2, t1)
        for k, d in self.dbpm.devices.items():
            self.assertTrue(len(d.value) == 100)
            self.assertTrue(isinstance(d.value, np.ndarray))

    @unittest.expectedFailure
    def test_set(self):
        self.dbpm.set([0.1, 0.2])

    def test_set2(self):
        self.dd.set([0.1])
        self.assertTrue(self.dd.readonce(settings=True) == 1)
        dev = list(self.dd.devices.values())[0]
        self.assertEqual(0.1, dev.value)

    def test_set3(self):
        print()
        dev = list(self.ds.devices.values())[0]
        self.assertFalse(dev.on)
        self.ds.set(['ON'])
        self.assertTrue(self.ds.readonce(settings=True) == 1)
        print(dev)
        self.assertTrue(dev.on)
        self.ds.set(['RESET'])
        self.assertTrue(self.ds.readonce(settings=True) == 1)
        self.assertTrue(dev.ready)
