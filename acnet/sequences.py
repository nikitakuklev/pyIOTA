from acnet.frontends import DeviceSet, StatusDevice, DoubleDevice
import time
import pyIOTA.iota as iota
import numpy as np


def get_bpms_orbit_mode(ds: DeviceSet, bpm_arm: StatusDevice, trigger: StatusDevice):
    print(f'BPMS: {ds}')
    print(f'TRIG: {trigger}')
    print(f'Firing in orbit mode!')
    trigger.reset()
    trigger.reset()
    time.sleep(0.01)
    ds.readonce()


def kick_vertical(vertical_kv=0.0, restore=False, debug=False):
    print('Kicking vertical only')

    plc = StatusDevice(iota.CONTROLS.CHIP_PLC)
    plc.read()
    if not plc.on:
        raise Exception("Chip PLC is not on???")
    if plc.remote:
        plc.set("Trig:Circ")

    if debug: print('>>Enabling vertical kicker')
    vkicker_status = StatusDevice(iota.CONTROLS.VKICKER)
    if not vkicker_status.on:
        vkicker_status.set_on()
    if not vkicker_status.ready:
        vkicker_status.reset()

    if debug: print('>>Checking $A5')
    a5 = StatusDevice(iota.CONTROLS.TRIGGER_A5)
    a5.read()
    if not a5.on:
        a5.set_on()

    if debug: print('>>Setting vertical kicker')
    vkicker = DoubleDevice(iota.CONTROLS.VKICKER)
    vkicker.read()
    if np.abs(vkicker.value - vertical_kv) > 0.05:
        vkicker.set(vertical_kv)
        time.sleep(0.1)
        for i in range(100):
            delta = np.abs(vkicker.read() - vertical_kv)
            if delta > 0.05:
                if i < 10:
                    time.sleep(0.1)
                    continue
                else:
                    raise Exception(f'Failed to set kicker - final delta: {delta}')
            else:
                break


    a5cnt = DoubleDevice(iota.CONTROLS.TRIGGER_A5)
    a5_initial_val = a5cnt.read()

    if debug: print('>>Firing')
    StatusDevice(iota.CONTROLS.BPM_INJ_TRIGGER).reset()
    a5.reset()

    #Await actual fire event
    for i in range(20):
        if a5cnt.read() > a5_initial_val:
            print('$A5 received - kick complete')
            return
        else:
            print(f'Awaiting $A5 {i}/{20}')
            time.sleep(0.1)

    raise Exception('$A5 never received - something went wrong!')


