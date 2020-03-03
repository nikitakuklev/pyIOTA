from .frontends import DeviceSet, StatusDevice, DoubleDevice
import time
import pyIOTA.iota as iota
import numpy as np


def trigger_bpms_orbit_mode(debug: bool = False):
    if debug: print('>>Resetting orbit trigger')
    StatusDevice(iota.CONTROLS.BPM_ORB_TRIGGER).reset()


def kick(vertical_kv: float = 0.0, horizontal_kv: float = 0.0,
         restore: bool = False, debug: bool = False):
    assert 1.0 > vertical_kv >= 0.0
    assert 3.0 > horizontal_kv >= 0.0

    if debug: print('>>Setting up Chip PLC')
    plc = StatusDevice(iota.CONTROLS.CHIP_PLC)
    plc.read()
    if not plc.on:
        raise Exception("Chip PLC is not on???")
    if plc.remote:
        plc.set("Trig:Circ")

    if vertical_kv > 0.0:
        if debug: print('>>Enabling vertical kicker')
        vkicker_status = StatusDevice(iota.CONTROLS.VKICKER)
        vkicker_status.read()
        if not vkicker_status.on:
            vkicker_status.set_on()
        if not vkicker_status.ready:
            vkicker_status.reset()

    if horizontal_kv > 0.0:
        if debug: print('>>Enabling horizontal kicker')
        hkicker_status = StatusDevice(iota.CONTROLS.HKICKER)
        hkicker_status.read()
        if not hkicker_status.on:
            hkicker_status.set_on()
        if not hkicker_status.ready:
            hkicker_status.reset()

    if debug: print('>>Checking $A5')
    a5 = StatusDevice(iota.CONTROLS.TRIGGER_A5)
    a5.read()
    if not a5.on:
        a5.set_on()

    if vertical_kv > 0.0:
        if debug: print('>>Setting vertical kicker')
        vkicker = DoubleDevice(iota.CONTROLS.VKICKER)
        vkicker.read()
        if np.abs(vkicker.value - vertical_kv) > 0.01:
            vkicker.set(vertical_kv)
            time.sleep(0.1)
            for i in range(100):
                delta = np.abs(vkicker.read() - vertical_kv)
                if delta > 0.05:
                    if i < 10:
                        time.sleep(0.1)
                        continue
                    else:
                        raise Exception(f'>>Failed to set kicker - final delta: {delta}')
                else:
                    if debug: print(f'>>Kicker setting ok - delta {delta}')
                    break

    if horizontal_kv > 0.0:
        if debug: print('>>Setting horizontal kicker')
        hkicker = DoubleDevice(iota.CONTROLS.HKICKER)
        hkicker.read()
        if np.abs(hkicker.value - horizontal_kv) > 0.01:
            hkicker.set(horizontal_kv)
            time.sleep(0.1)
            for i in range(100):
                delta = np.abs(hkicker.read() - horizontal_kv)
                if delta > 0.05:
                    if i < 10:
                        time.sleep(0.1)
                        continue
                    else:
                        raise Exception(f'>>Failed to set kicker - final delta: {delta}')
                else:
                    if debug: print(f'>>Kicker setting ok - delta {delta}')
                    break

    a5cnt = DoubleDevice(iota.CONTROLS.TRIGGER_A5)
    a5_initial_val = a5cnt.read()
    if debug: print(f'Initial $A5 cnt: {a5_initial_val}')

    if debug: print('>>Firing')
    StatusDevice(iota.CONTROLS.BPM_INJ_TRIGGER).reset()
    a5.reset()
    # Await actual fire event
    t0 = time.time()
    for i in range(20):
        a5_new_val = a5cnt.read()
        if a5_new_val > a5_initial_val:
            print(f'>>$A5 received (new: {a5_new_val}) - kick complete')
            return
        else:
            print(f'>>Awaiting $A5 {i}/{20} ({time.time() - t0:.3f}s)')
            time.sleep(0.1)

    raise Exception('$A5 never received - something went wrong!')


def kick_vertical(vertical_kv: float = 0.0, restore: bool = False, debug: bool = False):
    assert 1.0 > vertical_kv >= 0.0

    if debug: print('>>Setting up Chip PLC')
    plc = StatusDevice(iota.CONTROLS.CHIP_PLC)
    plc.read()
    if not plc.on:
        raise Exception("Chip PLC is not on???")
    if plc.remote:
        plc.set("Trig:Circ")

    if debug: print('>>Enabling vertical kicker')
    vkicker_status = StatusDevice(iota.CONTROLS.VKICKER)
    vkicker_status.read()
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
    if np.abs(vkicker.value - vertical_kv) > 0.01:
        vkicker.set(vertical_kv)
        time.sleep(0.1)
        for i in range(100):
            delta = np.abs(vkicker.read() - vertical_kv)
            if delta > 0.05:
                if i < 10:
                    time.sleep(0.1)
                    continue
                else:
                    raise Exception(f'>>Failed to set kicker - final delta: {delta}')
            else:
                if debug: print(f'>>Kicker setting ok - delta {delta}')
                break

    a5cnt = DoubleDevice(iota.CONTROLS.TRIGGER_A5)
    a5_initial_val = a5cnt.read()
    if debug: print(f'Initial $A5 cnt: {a5_initial_val}')

    if debug: print('>>Firing')
    StatusDevice(iota.CONTROLS.BPM_INJ_TRIGGER).reset()
    a5.reset()
    #bpmplc.set('ARMOrbit', adapter=relay)
    # Await actual fire event
    t0 = time.time()
    for i in range(20):
        a5_new_val = a5cnt.read()
        if a5_new_val > a5_initial_val:
            print(f'>>$A5 received (new: {a5_new_val}) - kick complete')
            return
        else:
            print(f'>>Awaiting $A5 {i}/{20} ({time.time() - t0:.3f}s)')
            time.sleep(0.1)

    raise Exception('$A5 never received - something went wrong!')


def kick_horizontal(horizontal_kv: float = 0.0, restore: bool = False, debug: bool = False):
    assert 3.0 > horizontal_kv >= 0.0

    if debug: print('>>Setting up Chip PLC')
    plc = StatusDevice(iota.CONTROLS.CHIP_PLC)
    plc.read()
    if not plc.on:
        raise Exception("Chip PLC is not on???")
    if plc.remote:
        plc.set("Trig:Circ")

    if debug: print('>>Enabling vertical kicker')
    vkicker_status = StatusDevice(iota.CONTROLS.HKICKER)
    vkicker_status.read()
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
    vkicker = DoubleDevice(iota.CONTROLS.HKICKER)
    vkicker.read()
    if np.abs(vkicker.value - horizontal_kv) > 0.09:
        vkicker.set(horizontal_kv)
        time.sleep(0.1)
        for i in range(100):
            delta = np.abs(vkicker.read() - horizontal_kv)
            if delta > 0.05:
                if i < 10:
                    time.sleep(0.1)
                    continue
                else:
                    raise Exception(f'>>Failed to set kicker - final delta: {delta}')
            else:
                if debug: print(f'>>Kicker setting ok - delta {delta}')
                break

    a5cnt = DoubleDevice(iota.CONTROLS.TRIGGER_A5)
    a5_initial_val = a5cnt.read()
    if debug: print(f'Initial $A5 cnt: {a5_initial_val}')

    if debug: print('>>Firing')
    StatusDevice(iota.CONTROLS.BPM_INJ_TRIGGER).reset()
    a5.reset()
    #bpmplc.set('ARMOrbit', adapter=relay)
    # Await actual fire event
    t0 = time.time()
    for i in range(20):
        a5_new_val = a5cnt.read()
        if a5_new_val > a5_initial_val:
            print(f'>>$A5 received (new: {a5_new_val}) - kick complete')
            return
        else:
            print(f'>>Awaiting $A5 {i}/{20} ({time.time() - t0:.3f}s)')
            time.sleep(0.1)

    raise Exception('$A5 never received - something went wrong!')