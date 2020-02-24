from acnet.frontends import DeviceSet, StatusDevice
import time


def get_bpms_orbit_mode(ds: DeviceSet, trigger: StatusDevice):
    print(f'BPMS: {ds}')
    print(f'TRIG: {trigger}')
    trigger.reset()
    time.sleep(0.01)
    ds.readonce()
0
