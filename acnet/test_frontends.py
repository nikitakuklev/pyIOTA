from unittest import TestCase
from .frontends import BPMDeviceSet, BPMDevice, ACL
#from ..iota.run2 import BPMS, MAGNETS
#from pyIOTA import iota
import pyIOTA.iota


class TestWrap(TestCase):
    def setUp(self):
        d1 = BPMDevice('N:IBB1RH', '', array_length=1500)
        d2 = BPMDevice('N:IBB1RV', '', array_length=1500)
        ds = BPMDeviceSet('bpmset', members=[d1, d2], enforce_array_length=1100)
        ds.adapter = ACL(fallback=True)
        self.ds = ds

        d3 = BPMDevice('N:IBB2RH', '', array_length=1500)
        d4 = BPMDevice('N:IBB2RV', '', array_length=1500)
        ds_async = BPMDeviceSet('bpmset', members=[d3, d4], enforce_array_length=1100)
        ds_async.adapter = ACL()
        self.ds_async = ds_async


class TestDeviceSet(TestWrap):
    def test_device_set(self):
        devs = list(self.ds)
        self.assertEqual(devs, ['N:IBB1RH', 'N:IBB1RV'])

    def test_start_oneshot(self):
        d1 = list(self.ds.devices.values())[0]
        print(f'\nOld ts: {d1.last_update} | {d1.value}')
        old_ts = d1.last_update
        self.ds.start_oneshot()
        print(f'New ts: {d1.last_update} | {d1.value}')
        self.assertTrue(old_ts != d1.last_update)

    def test_start_oneshot2(self):
        devs = [BPMDevice(b) for b in iota.BPMS.HA]
        dstest = BPMDeviceSet(name='bpms', members=devs, enforce_array_length=1000)
        dstest.adapter = ACL(fallback=False)
        dstest.start_oneshot()
        print(devs)

    def test_check_acquisition_supported(self):
        self.assertTrue(self.ds.check_acquisition_supported('oneshot'))
        self.assertFalse(self.ds.check_acquisition_supported('fakemethod'))


class TestACLAsync(TestWrap):
    def test_start_oneshot(self):
        d1 = list(self.ds_async.devices.values())[0]
        print(f'\nOld ts: {d1.last_update} | {d1.value}')
        self.ds_async.start_oneshot()
        print(f'New ts: {d1.last_update} | {d1.value}')


class TestACL(TestWrap):
    def test_del(self):
        del self.ds.adapter


class TestBPMDeviceSet(TestWrap):
    def test_add(self):
        self.ds.add(BPMDevice('N:IBB1RS', '', array_length=1500))
        self.assertTrue('N:IBB1RS' in self.ds.devices)
        self.assertEqual(list(self.ds.devices.keys()), ['N:IBB1RH', 'N:IBB1RV', 'N:IBB1RS'])

    def test_remove(self):
        self.ds.remove('N:IBB1RV')
        self.assertEqual(list(self.ds.devices.keys()), ['N:IBB1RH'])

    def test_remove2(self):
        self.ds.remove('N:IBB1RV')
        self.ds.remove('N:IBB1RH')
        self.assertEqual(list(self.ds.devices.keys()), [])

    def test_remove3(self):
        self.ds.remove('BLA')
        self.assertEqual(list(self.ds.devices.keys()), ['N:IBB1RH', 'N:IBB1RV'])



