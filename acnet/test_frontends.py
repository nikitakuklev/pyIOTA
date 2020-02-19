from unittest import TestCase
from .frontends import BPMDeviceSet, BPMDevice


class TestWrap(TestCase):
    def setUp(self):
        d1 = BPMDevice('N:IBB2RV','',array_length=1500)
        d2 = BPMDevice('N:IBB1RV','',array_length=1500)
        ds = BPMDeviceSet('bpmset',members=[d1,d2],enforce_array_length=1100)
        self.ds = ds

class TestDeviceSet(TestWrap):
    def test_start_oneshot(self):
        pass#self.fail()


class TestBPMDeviceSet(TestWrap):
    def test_add(self):
        self.ds.add(BPMDevice('N:IBB1RS','',array_length=1500))
        print(self.ds.devices)
        assert 'N:IBB1RS' in self.ds.devices
