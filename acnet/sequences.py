import uuid
from pathlib import Path

import pyIOTA
from .frontends import DeviceSet, StatusDevice, DoubleDevice, DoubleDeviceSet, BPMDevice, BPMDeviceSet, ACNETRelay, ACL, \
    StatusDeviceSet
import time, datetime
import pyIOTA.iota as iota
from pyIOTA.sixdsim.io import Knob
import pyIOTA.acnet.utils as acutils
import numpy as np
import pandas as pd


def trigger_bpms_orbit_mode(debug: bool = False):
    if debug: print('>>Resetting orbit trigger')
    StatusDevice(iota.CONTROLS.BPM_ORB_TRIGGER).reset()


def get_bpm_data(bpm_ds=None, mode: str = 'tbt', kickv: float = np.nan, kickh: float = np.nan,
                 read_beam_current: bool = True, read_state: bool = True, read_aux=True, state_ds=None, status_ds=None,
                 check_sequence_id: bool = True, last_sequence_id: bool = None, save: bool = False, fpath: Path = None,
                 save_repeats=False,
                 custom_state_parameters: dict = None):
    state = {}
    if read_state:
        if state_ds is None:
            knobs_to_save = iota.run2.MASTER_STATE_CURRENTS  # all iota magnets + RF + kickers
            state_ds = DoubleDeviceSet(name='state', members=[DoubleDevice(d) for d in knobs_to_save])
        nstate = state_ds.readonce(settings=True, verbose=False)
        state.update({d.name + '.SETTING': d.value for d in state_ds.devices.values()})
    else:
        nstate = 0
    state['kickv'] = kickv
    state['kickh'] = kickh

    if read_beam_current:
        state[iota.run2.OTHER.BEAM_CURRENT_AVERAGE + '.READING'] = DoubleDevice(
            iota.run2.OTHER.BEAM_CURRENT_AVERAGE).read()

    if read_aux:
        aux = iota.run2.OTHER.AUX_DEVICES
        state_ds = DoubleDeviceSet(name='state', members=[DoubleDevice(d) for d in aux])
        nstate2 = state_ds.readonce(verbose=False)
        state.update({d.name + '.READING': d.value for d in state_ds.devices.values()})
        nstate += nstate2

        statuses = iota.run2.MASTER_STATUS_DEVICES
        status_ds = StatusDeviceSet(name='status', members=[StatusDevice(d) for d in statuses])
        nstate3 = status_ds.readonce(verbose=False)
        state.update({d.name + '.STATUS': d.value for d in status_ds.devices.values()})
        nstate += nstate3

    if bpm_ds is None:
        bpms = [BPMDevice(b) for b in iota.BPMS.ALLA]
        bpm_ds = BPMDeviceSet(name='bpms', members=bpms, adapter=ACNETRelay(method=1), enforce_array_length=None)
    nbpm = bpm_ds.readonce(verbose=False)
    if mode == 'tbt':
        bpm_ds.array_length = None
        data = {d.name: d.value for d in bpm_ds.devices.values()}
    else:
        bpm_ds.array_length = 2048
        data = {d.name: d.value for d in bpm_ds.devices.values()}

    state['aq_timestamp'] = datetime.datetime.utcnow().timestamp()
    state['run_uuid'] = str(uuid.uuid4())

    repeat_data = False
    mixed_data = False
    if check_sequence_id:
        if len(data) == 0:
            val_last_ret = np.nan
        for i, (k, v) in enumerate(data.items()):
            val_last_ret = int(v[0])
            if last_sequence_id is None:
                break
            else:
                if int(v[0]) == last_sequence_id:
                    print('Sequence number did not change! Skipping!')
                    repeat_data = True
                    break

        for i, (k, v) in enumerate(data.items()):
            if i == 0:
                val_seq = int(v[0])
            else:
                if val_seq != int(v[0]):
                    print(f'Sequence number is not uniform - {val_seq} vs {int(v[0])} on BPM {k}(#{i}) (wont be saved)')
                    mixed_data = True
                    # raise Exception
                    break
    else:
        val_seq = -1
        val_last_ret = -1

    custom_state_parameters = {} if custom_state_parameters is None else custom_state_parameters
    datadict = [{'idx': 0.0, 'kickv': kickv, 'kickh': kickh, 'state': state,
                 'custom': custom_state_parameters, **data}]
    df = pd.DataFrame(data=datadict)

    if (save_repeats or not repeat_data) and save and not mixed_data:
        savedate = datetime.datetime.now().strftime("%Y_%m_%d")
        fpath = Path(fpath) / f'{savedate}'
        acutils.save_data_tbt(fpath, df)
        saved_ok = True
    else:
        saved_ok = False

    return data, state, val_last_ret, saved_ok


def transition(final_state: Knob, steps: int = 5, verbose: bool = True, extra_final_setting=False,
               retry=True, retry_limit=10):
    """
    Transition sequencing - moves towards knob state in uniform, smaller steps.
    :param extra_final_setting:
    :param final_state:
    :param steps:
    :param verbose:
    :return:
    """
    t0 = time.time()
    if not final_state.absolute:
        raise
        if verbose: print(f'Add relative knob ({final_state}) in ({steps}) steps')
        initial_state = final_state.copy()
        initial_state.read_current_state()
        if verbose: print(f'Current state read OK')
        delta_knob = final_state / steps
        for step in range(1, steps + 1):
            if verbose: print(f'{step}/{steps} ', end='')
            intermediate_state = initial_state + delta_knob * step
            intermediate_state.set(verbose=verbose)
            time.sleep(0.1)
        if extra_final_setting:
            time.sleep(0.5)
            (initial_state + final_state).set(verbose=verbose)
        if verbose: print(f'Done')
    else:
        # if verbose: print(f'Transitioning to ({final_state}) in ({steps}) steps')
        print(f'Transitioning to ({final_state})(abs:{final_state.absolute}) in ({steps}) steps')
        initial_state = final_state.copy().read_current_state()
        if verbose: print(f'\nCurrent state read OK')
        delta_knob = ((final_state - initial_state) / steps).prune(tol=1e-4)
        if verbose: print(f'\nTo change', delta_knob.vars)
        if delta_knob.is_empty():
            if verbose: print(f'No changes necessary!')
            return
        initial_state_pruned = initial_state.copy().only_keep_shared(delta_knob)
        for step in range(1, steps + 1):
            # if verbose: print(f'{step}/{steps} ', end='')
            print(f'step {step}/{steps}...', end='')
            intermediate_state = initial_state_pruned + delta_knob * step
            intermediate_state.set(verbose=verbose)
            time.sleep(0.5)
        print('done!')

        if extra_final_setting:
            time.sleep(0.5)
            extra_state = final_state.copy()
            extra_state.read_current_state()
            if verbose: print('\nExtra state read:\n', [(k.var, k.value) for k in extra_state.vars.values()])
            extra_delta = final_state - extra_state
            to_change_extra = extra_delta.prune(tol=1e-4)
            if verbose: print(f'\nTo change ({to_change_extra})')
            if to_change_extra.is_empty():
                if verbose: print(f'No changes necessary!')
                return
            if verbose: print('\nExtra delta:\n', [(k.var, k.value) for k in extra_delta.vars.values()])
            extra_state_pruned = final_state.copy()
            extra_state_pruned.vars = {k: v for k, v in extra_state_pruned.vars.items() if k in to_change_extra.vars}
            extra_state_pruned.set(verbose=verbose)

        if retry:
            for i in range(retry_limit):
                extra_state = final_state.copy().read_current_state(settings=True)
                if verbose: print('\nExtra state read:\n', [(k.var, k.value) for k in extra_state.vars.values()])
                extra_delta = (final_state - extra_state).prune(tol=1e-3)
                if extra_delta.is_empty():
                    if verbose: print(f'Retry {i}/{retry_limit} - all settings satisfied!')
                    break
                else:
                    print(f'\nExtra changes left: {extra_delta.vars}')
                extra_state_pruned = final_state.copy().only_keep_shared(extra_delta)
                extra_state_pruned.set(verbose=verbose)
                print(f'Running retry loop {i}/{retry_limit} - set {len(extra_state_pruned.vars)} devices')
        print(f'Knob {final_state.name} set in {time.time() - t0:.5f}s')


def inject_until_current(arm_bpms: bool = False, debug: bool = False, current: float = 1.0, limit: int = 10):
    ibeama = DoubleDevice(iota.run2.OTHER.BEAM_CURRENT_AVERAGE)
    for i in range(limit):
        inject(arm_bpms=arm_bpms, debug=debug)
        time.sleep(1.0)
        i = ibeama.read()
        if i < current:
            print(f'Injection loop - got {i}mA - goal met')
            break
        else:
            print(f'Injection loop - got {i}mA - retrying')


def inject(arm_bpms: bool = False, debug: bool = False):
    a6cnt = DoubleDevice(iota.CONTROLS.TRIGGER_A6)

    if debug: print('>>Setting up Chip PLC')
    plc = StatusDevice(iota.CONTROLS.CHIP_PLC)
    plc.read()
    if not plc.on:
        raise Exception("Chip PLC is not on???")
    if not plc.remote:
        plc.set("Trig:Inj")

    if debug: print('>>Checking $A6')
    a6 = StatusDevice(iota.CONTROLS.TRIGGER_A6)
    a6.read()
    if not a6.on:
        a6.set_on()

    if debug: print('>>Turning off HKICKER aux devices')
    hres = StatusDevice(iota.CONTROLS.HKICKER_RESCHARGE)
    hres.set_off()
    htrig = StatusDevice(iota.CONTROLS.HKICKER_TRIG)
    htrig.set_off()

    if debug: print('>>Turning on VKICKER aux devices')
    vres = StatusDevice(iota.CONTROLS.VKICKER_RESCHARGE)
    vres.read()
    if not vres.on:
        vres.set_on()
    vtrig = StatusDevice(iota.CONTROLS.VKICKER_TRIG)
    vtrig.read()
    if not vtrig.on:
        vtrig.set_on()

    if debug: print('>>Enabling vertical kicker')
    vkicker_status = StatusDevice(iota.CONTROLS.VKICKER)
    vkicker_status.read()
    if not vkicker_status.on:
        vkicker_status.set_on()
    if not vkicker_status.ready:
        vkicker_status.reset()

    vertical_kv = 4.15
    if debug: print('>>Setting vertical kicker')
    vkicker = DoubleDevice(iota.CONTROLS.VKICKER)
    vkicker.read()
    if np.abs(vkicker.value - vertical_kv) > 0.01:
        vkicker.set(vertical_kv)
        time.sleep(0.1)
        for i in range(100):
            delta = np.abs(vkicker.read() - vertical_kv)
            if delta > 0.15:
                if i < 10:
                    time.sleep(0.1)
                    continue
                else:
                    raise Exception(f'>>Failed to set kicker - final delta: {delta}')
            else:
                if debug: print(f'>>Kicker setting ok - delta {delta}')
                break

    shutter = StatusDevice(iota.CONTROLS.FAST_LASER_SHUTTER)
    shutter.read()
    if not shutter.on:
        print('FYI - SHUTTER CLOSED ARGGGG!!!!')
    if not shutter.ready:
        print('MPS FAULT!')
        raise Exception("MPS FAULT - ABORTING INJECTION")

    if debug: print('>>Awaiting A8')
    acl = pyIOTA.acnet.frontends.ACL(fallback=True)
    acl._raw_command("wait/nml/event/timeout=2 A8")

    if debug: print('>>Firing')
    inj = DoubleDevice(iota.CONTROLS.FAST_LASER_INJECTOR)
    if arm_bpms:
        StatusDevice(iota.CONTROLS.BPM_INJ_TRIGGER).reset()
    a6.reset()
    inj.set(1)


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
        if debug: print('>>Turning on VKICKER aux devices')
        vres = StatusDevice(iota.CONTROLS.VKICKER_RESCHARGE)
        vres.read()
        if not vres.on:
            vres.set_on()
        vtrig = StatusDevice(iota.CONTROLS.VKICKER_TRIG)
        vtrig.read()
        if not vtrig.on:
            vtrig.set_on()

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
    else:
        if debug: print('>>Turning off VKICKER aux devices')
        vres = StatusDevice(iota.CONTROLS.VKICKER_RESCHARGE)
        vres.set_off()
        vtrig = StatusDevice(iota.CONTROLS.VKICKER_TRIG)
        vtrig.set_off()

    if horizontal_kv > 0.0:
        if debug: print('>>Turning on HKICKER aux devices')
        hres = StatusDevice(iota.CONTROLS.HKICKER_RESCHARGE)
        hres.read()
        if not hres.on:
            hres.set_on()
        htrig = StatusDevice(iota.CONTROLS.HKICKER_TRIG)
        htrig.read()
        if not htrig.on:
            htrig.set_on()

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
    else:
        if debug: print('>>Turning off HKICKER aux devices')
        hres = StatusDevice(iota.CONTROLS.HKICKER_RESCHARGE)
        hres.set_off()
        htrig = StatusDevice(iota.CONTROLS.HKICKER_TRIG)
        htrig.set_off()

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

# def kick_vertical(vertical_kv: float = 0.0, restore: bool = False, debug: bool = False):
#     assert 1.0 > vertical_kv >= 0.0
#
#     if debug: print('>>Setting up Chip PLC')
#     plc = StatusDevice(iota.CONTROLS.CHIP_PLC)
#     plc.read()
#     if not plc.on:
#         raise Exception("Chip PLC is not on???")
#     if plc.remote:
#         plc.set("Trig:Circ")
#
#     if debug: print('>>Enabling vertical kicker')
#     vkicker_status = StatusDevice(iota.CONTROLS.VKICKER)
#     vkicker_status.read()
#     if not vkicker_status.on:
#         vkicker_status.set_on()
#     if not vkicker_status.ready:
#         vkicker_status.reset()
#
#     if debug: print('>>Checking $A5')
#     a5 = StatusDevice(iota.CONTROLS.TRIGGER_A5)
#     a5.read()
#     if not a5.on:
#         a5.set_on()
#
#     if debug: print('>>Setting vertical kicker')
#     vkicker = DoubleDevice(iota.CONTROLS.VKICKER)
#     vkicker.read()
#     if np.abs(vkicker.value - vertical_kv) > 0.01:
#         vkicker.set(vertical_kv)
#         time.sleep(0.1)
#         for i in range(100):
#             delta = np.abs(vkicker.read() - vertical_kv)
#             if delta > 0.05:
#                 if i < 10:
#                     time.sleep(0.1)
#                     continue
#                 else:
#                     raise Exception(f'>>Failed to set kicker - final delta: {delta}')
#             else:
#                 if debug: print(f'>>Kicker setting ok - delta {delta}')
#                 break
#
#     a5cnt = DoubleDevice(iota.CONTROLS.TRIGGER_A5)
#     a5_initial_val = a5cnt.read()
#     if debug: print(f'Initial $A5 cnt: {a5_initial_val}')
#
#     if debug: print('>>Firing')
#     StatusDevice(iota.CONTROLS.BPM_INJ_TRIGGER).reset()
#     a5.reset()
#     #bpmplc.set('ARMOrbit', adapter=relay)
#     # Await actual fire event
#     t0 = time.time()
#     for i in range(20):
#         a5_new_val = a5cnt.read()
#         if a5_new_val > a5_initial_val:
#             print(f'>>$A5 received (new: {a5_new_val}) - kick complete')
#             return
#         else:
#             print(f'>>Awaiting $A5 {i}/{20} ({time.time() - t0:.3f}s)')
#             time.sleep(0.1)
#
#     raise Exception('$A5 never received - something went wrong!')
#
#
# def kick_horizontal(horizontal_kv: float = 0.0, restore: bool = False, debug: bool = False):
#     assert 3.0 > horizontal_kv >= 0.0
#
#     if debug: print('>>Setting up Chip PLC')
#     plc = StatusDevice(iota.CONTROLS.CHIP_PLC)
#     plc.read()
#     if not plc.on:
#         raise Exception("Chip PLC is not on???")
#     if plc.remote:
#         plc.set("Trig:Circ")
#
#     if debug: print('>>Enabling vertical kicker')
#     vkicker_status = StatusDevice(iota.CONTROLS.HKICKER)
#     vkicker_status.read()
#     if not vkicker_status.on:
#         vkicker_status.set_on()
#     if not vkicker_status.ready:
#         vkicker_status.reset()
#
#     if debug: print('>>Checking $A5')
#     a5 = StatusDevice(iota.CONTROLS.TRIGGER_A5)
#     a5.read()
#     if not a5.on:
#         a5.set_on()
#
#     if debug: print('>>Setting vertical kicker')
#     vkicker = DoubleDevice(iota.CONTROLS.HKICKER)
#     vkicker.read()
#     if np.abs(vkicker.value - horizontal_kv) > 0.09:
#         vkicker.set(horizontal_kv)
#         time.sleep(0.1)
#         for i in range(100):
#             delta = np.abs(vkicker.read() - horizontal_kv)
#             if delta > 0.05:
#                 if i < 10:
#                     time.sleep(0.1)
#                     continue
#                 else:
#                     raise Exception(f'>>Failed to set kicker - final delta: {delta}')
#             else:
#                 if debug: print(f'>>Kicker setting ok - delta {delta}')
#                 break
#
#     a5cnt = DoubleDevice(iota.CONTROLS.TRIGGER_A5)
#     a5_initial_val = a5cnt.read()
#     if debug: print(f'Initial $A5 cnt: {a5_initial_val}')
#
#     if debug: print('>>Firing')
#     StatusDevice(iota.CONTROLS.BPM_INJ_TRIGGER).reset()
#     a5.reset()
#     #bpmplc.set('ARMOrbit', adapter=relay)
#     # Await actual fire event
#     t0 = time.time()
#     for i in range(20):
#         a5_new_val = a5cnt.read()
#         if a5_new_val > a5_initial_val:
#             print(f'>>$A5 received (new: {a5_new_val}) - kick complete')
#             return
#         else:
#             print(f'>>Awaiting $A5 {i}/{20} ({time.time() - t0:.3f}s)')
#             time.sleep(0.1)
#
#     raise Exception('$A5 never received - something went wrong!')
