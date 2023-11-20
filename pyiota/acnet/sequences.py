import datetime
import logging
import pathlib
import time
import uuid
from pathlib import Path

import numpy as np
from pydantic import BaseModel

import pyiota.iota as iota
from . import ACL
from pyiota.acnet.adapters import Adapter, DPM
from pyiota.acnet.devices import ArrayDevice, ArrayDeviceSet, DoubleDevice, DoubleDeviceSet, \
    StatusDevice, StatusDeviceSet

from typing import Literal, Optional, TYPE_CHECKING, Union

from pyiota.acnet.drf2 import DRF_PROPERTY
from pyiota.acnet.errors import SequencerError

if TYPE_CHECKING:
    from pyiota.sixdsim.io import Knob

logger = logging.getLogger(__name__)


class ConditionalLogger:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __call__(self, *args, **kwargs):
        if self.verbose:
            logger.debug(*args, **kwargs)


def trigger_bpms_orbit_mode(debug: bool = False):
    if debug:
        print('>>Resetting orbit trigger')
    StatusDevice(iota.CONTROLS.BPM_ORB_TRIGGER).reset()


def check_bpm_saturation(bpm_indices: list,
                         threshold: int = 65000,
                         calculate_load: bool = True,
                         turns: int = 256,
                         load_turns: int = 128
                         ) -> dict:
    bpm_index = DoubleDevice(iota.CONTROLS.BPM_RAWREAD_INDEX)
    bpm_control = StatusDevice(iota.CONTROLS.BPM_CONFIG_DEVICE)

    load = {}
    for idx in bpm_indices:
        bpm_index.set(idx)
        time.sleep(0.01)
        bpm_control.set('RawReadout')
        time.sleep(0.1)
        bpms = [ArrayDevice(plate) for plate in iota.BPMS.RAW_ALL]
        ds = ArrayDeviceSet(name='temp', members=bpms, enforce_array_length=turns)
        ds.read()
        for bpm in bpms:
            arr = np.abs(bpm.value)
            if np.abs(np.max(arr)) > threshold:
                raise Exception(
                        f'Plate ({bpm.name}) of ({idx}) exceeded limit ({threshold}): {bpm.value[:40]} (loads:{load})')
            else:
                load[str(idx) + '-' + bpm.name] = np.mean(arr[:load_turns])
    return load


class TBTData(BaseModel):
    uuid: str
    timestamp: float
    metadata: dict[str, Optional[Union[float, int, str, np.ndarray]]]
    state: dict[str, Optional[Union[float, int, str, np.ndarray]]]
    bpm_data: dict[str, np.ndarray]
    data_version: int = 3
    data_type: str

    class Config:
        arbitrary_types_allowed = True

    def to_hdf5(self, fpath: pathlib.Path, overwrite: bool = False):
        fpath = pathlib.Path(fpath)
        import h5py
        h5py.get_config().track_order = True

        def _check_value(val):
            return h5py.Empty("f") if val is None else val

        logger.debug(f'Full save path: {fpath}')
        if fpath.exists() and not overwrite:
            print(f'Path {fpath} exists, aborting')
            raise Exception("File already exists")
        with h5py.File(str(fpath), 'w', libver='latest') as f:
            bpmgr = f.create_group('bpm')
            for (k, v) in self.bpm_data.items():
                bpmgr.create_dataset(k, data=v, compression='gzip',
                                     compression_opts=9,
                                     shuffle=True, fletcher32=True)

            stategr = f.create_group('state')
            for (k, v) in self.state.items():
                # try:
                stategr.attrs[k] = _check_value(v)
                # except:
                #    logger.warning(f'Failed to set {k} = {v}')
            # print(stategr.name, len(stategr.attrs))
            metagr = f.create_group('metadata')
            for (k, v) in self.metadata.items():
                metagr.attrs[k] = v
            f.attrs['timestamp'] = self.timestamp
            f.attrs['timestamp_save'] = datetime.datetime.utcnow().timestamp()
            f.attrs['data_type'] = self.data_type
            f.attrs['uuid'] = self.uuid

    @staticmethod
    def from_hdf5(fpath: pathlib.Path, load_bpms:bool = True):
        import h5py
        with h5py.File(str(fpath), 'r') as f:
            if load_bpms:
                bpm_data = {k: v.astype(np.float64)[:] for k, v in f['bpm'].items()}
            else:
                bpm_data = {}
            metadata = dict(f['metadata'].attrs)

            # state = dict(f['state'].attrs)
            def _parse(val):
                return val if not isinstance(val, h5py.Empty) else None

            state = {k: _parse(v) for k, v in f['state'].attrs.items()}

            return TBTData(uuid=f.attrs['uuid'],
                           timestamp=f.attrs['timestamp_save'],
                           metadata=metadata,
                           state=state,
                           data_type=f.attrs['data_type'],
                           bpm_data=bpm_data)


def get_bpm_data(bpm_ds: ArrayDeviceSet = None,
                 state_ds: DoubleDeviceSet = None,
                 mode: Literal['tbt', 'orbit'] = 'tbt',
                 read_beam_current: bool = True,
                 read_state: bool = True,
                 read_aux: bool = True,
                 metadata: dict = None,
                 check_sequence_id: bool = True,
                 last_sequence_id: int = None,
                 save: bool = False,
                 save_path: Path = None,
                 save_repeats: bool = False,
                 collection_seq_number: int = 0,
                 adapter: Adapter = None,
                 bpm_retries: int = 3,
                 ):
    """
    Acquire BPM data arrays from the bpm device set.
    If specified, accelerator state will also be saved.
    This routine does not perform any writes or destructive actions.

    :param bpm_ds: Device set of BPMs
    :param state_ds: Device set corresponding to 'state', if blank the default is used
    :param mode: 'tbt' or 'orbit', only affects the length of data to be saved
    :param read_beam_current: whether to read beam current
    :param read_state: Whether to read default or custom state
    :param read_aux: Whether to read auxiliary devices
    :param metadata: If provided, will be saved into 'metadata' key
    :param check_sequence_id: Check if BPM sequence has incremented
    :param last_sequence_id: ID of last BPM dataset
    :param save: Whether to save data
    :param save_path: Path to save data
    :param save_repeats: If true, save data that has same as ID as last dataset
    :param collection_seq_number: Deprecated, not used
    :param adapter: override which adapter to use for reading everything
    :return:
    """
    state = {}
    metadata = metadata if metadata is not None else {}
    if adapter is not None:
        pass
    else:
        adapter = state_ds.adapter if state_ds is not None else None
    if read_state:
        # TODO: FIX THIS GARBAGE NAMING
        if state_ds is None:
            knobs_to_save = iota.MASTER_STATE_CURRENTS  # all iota magnets + RF + kickers
            state_ds = DoubleDeviceSet(name='state',
                                       members=[DoubleDevice(d) for d in knobs_to_save])
        read_state = state_ds.read()
        n_state = len(read_state)
        state.update({d.name + '.SETTING': d.dump() for d in state_ds.devices.values()})

        state_setp = DoubleDeviceSet(members=[d for d in state_ds.devices], settings=True,
                                     adapter=state_ds.adapter)
        state_setp.read()
        n_state += len(read_state)
        state.update({d.name + '.SETP': d.dump() for d in state_setp.devices.values()})
    else:
        n_state = 0

    if read_beam_current and not read_aux:
        # Current also included in AUX
        state[iota.OTHER.BEAM_CURRENT_AVERAGE] = DoubleDevice(
                iota.OTHER.BEAM_CURRENT_AVERAGE).read()
        state[iota.OTHER.BEAM_CURRENT_SL] = DoubleDevice(
                iota.OTHER.BEAM_CURRENT_SL).read()

    if read_aux:
        aux = iota.OTHER.AUX_DEVICES

        ds1 = DoubleDeviceSet(members=[DoubleDevice(d) for d in aux], adapter=adapter)
        read_state1 = ds1.read()
        state.update({d.name + '.READING': d.dump() for d in ds1.devices.values()})
        n_state += len(read_state1)

        ds2 = DoubleDeviceSet(members=[DoubleDevice(d) for d in aux], settings=True,
                              adapter=adapter)
        read_state2 = ds2.read()
        state.update({d.name + '.SETTING': d.dump() for d in ds2.devices.values()})
        n_state += len(read_state2)

        statuses = iota.MASTER_STATUS_DEVICES
        status_ds = StatusDeviceSet(members=[StatusDevice(d) for d in statuses])
        read_state3 = status_ds.read()
        state.update({d.name + '.STATUS': d.dump() for d in status_ds.devices.values()})
        n_state += len(read_state3)

    if bpm_ds is None:
        bpms = [ArrayDevice(b) for b in iota.BPMS.ALLA]
        bpm_ds = ArrayDeviceSet(members=bpms,
                                adapter=DPM(),
                                enforce_array_length=7200)  # DPM limit lower, used to be 8192
    if mode == 'tbt':
        bpm_ds.array_length = 7200  # None
    else:
        bpm_ds.array_length = 2048

    bpm_ds.read()
    for i in range(bpm_retries):
        bad = False
        for d in bpm_ds.devices.values():
            if d.value is None:
                bad = True
                logger.warning(f'BPM ({d.name}) returned None as value ({d.error=}) -> retrying all'
                               f'({i}/{bpm_retries})')
        if not bad:
            break
        else:
            bpm_ds.read()
    bpm_data = {d.name: d.value for d in bpm_ds.devices.values()}

    repeat_data = False
    mixed_data = False
    val_last_ret = last_sequence_id
    if check_sequence_id:
        if len(bpm_data) == 0:
            val_last_ret = np.nan
        for i, (k, v) in enumerate(bpm_data.items()):
            if v is None:
                raise Exception(f'BPM ({k}) returned None as value')
            val_last_ret = int(v[0])
            if last_sequence_id is None:
                break
            else:
                if val_last_ret == last_sequence_id:
                    print(
                            f'Sequence number {last_sequence_id} did not change on BPM {k}(#{i})! Skipping!')
                    repeat_data = True
                    break

        val_seq = -2
        for i, (k, v) in enumerate(bpm_data.items()):
            if v is None:
                continue
            if val_seq == -2:
                val_seq = int(v[0])
            else:
                if val_seq != int(v[0]):
                    logger.error(
                            f'Sequence number not uniform - {val_seq} vs {int(v[0])} on BPM {k}(#{i}) (wont be saved)')
                    mixed_data = True
                    val_last_ret = last_sequence_id
                    # raise Exception
                    break
    else:
        val_seq = -1
        val_last_ret = -1

    kwargs = {'timestamp': datetime.datetime.utcnow().timestamp(),
              'uuid': str(uuid.uuid4()),
              # 'sequence_idx': collection_seq_number,
              'data_type': mode,
              'metadata': metadata,
              'state': state,
              'bpm_data': bpm_data,
              }
    data = TBTData.parse_obj(kwargs)

    # datadict = [{'idx': collection_seq_number, 'kickv': kickv, 'kickh': kickh, 'state': state,
    #              'custom': custom_state_parameters, **data
    #              }]
    # df = pd.DataFrame(data=datadict)

    if (save_repeats or not repeat_data) and save and not mixed_data:
        # savedate = datetime.datetime.now().strftime("%Y_%m_%d")
        save_path = Path(save_path)  # / f'{savedate}'
        assert not save_path.exists() or save_path.is_dir()
        save_path.mkdir(parents=True, exist_ok=True)
        name_format: str = "iota_kicks_%Y%m%d-%H%M%S.hdf5"
        file_name = datetime.datetime.now().strftime(name_format)
        fnamefull = save_path / file_name
        data.to_hdf5(fpath=fnamefull)
        saved_ok = True
    else:
        saved_ok = False

    return data, state, val_last_ret, saved_ok


def transition(final_state: 'Knob',
               steps: Union[int, list] = 5,
               equal_tolerance: float = 3e-4,
               extra_final_setting: bool = False,
               retry_endpoint: Optional[int] = 10,
               retry_tolerance: float = 1e-2,
               step_pause: float = 0.5,
               verbose: bool = False,
               trace: bool = False,
               delay_retry_endpoint: float = 0.1,
               split_settings: bool = False,
               split_readings: bool = False,
               equal_tol_exceptions: dict[str, float] = None,
               ):
    """
    Transition sequencing - moves towards knob state in uniform, smaller steps.

    :param final_state: Absolute knob representing final device state
    :param steps: Number of equal steps or list of step strengths. Changes will be normalized.
    :param equal_tolerance: Tolerance of setpoints to exclude from transition entirely
    :param retry_endpoint: Number of times to try final setting if too far away from goal
    :param retry_tolerance: Tolerance to consider setpoint match
    :param verbose: Print progress
    :param extra_final_setting: Do extra setting for all devices at the end
    :param step_pause: Delay between steps
    :param trace: Extra prints
    :param split_settings: Split setting commands into separate devices
    :param split_readings: Split readbacks into separate devices
    :param delay_retry_endpoint:
    :param equal_tol_exceptions: Devices with custom setpoint equal tolerances that override
    default of equal_tolerance. Used for RF with tiny 1e-8 changes.
    :return:
    """
    log = ConditionalLogger(verbose)
    logt = ConditionalLogger(trace)
    t0 = time.time()
    equal_tol_exceptions = equal_tol_exceptions or {'N:IRFLLF': 1e-9}
    if not final_state.absolute:
        raise NotImplementedError
        log(f'Add relative knob ({final_state}) in ({steps}) steps')
        initial_state = final_state.copy()
        initial_state.read_current_state(split=split_readings)
        if verbose:
            print(f'Current state read OK')
        delta_knob = final_state / steps
        for step in range(1, steps + 1):
            if verbose:
                print(f'{step}/{steps} ', end='')
            intermediate_state = initial_state + delta_knob * step
            intermediate_state.set(verbose=verbose)
            time.sleep(1.0)
        if extra_final_setting:
            time.sleep(0.5)
            (initial_state + final_state).set(verbose=verbose)
        if verbose:
            print(f'Done')
    else:
        log(f'Transitioning to ({final_state}) in ({steps}) steps')
        if isinstance(steps, list):
            assert sum(steps) > 0
            steps_relative = np.array(steps) / sum(steps)
        else:
            steps_relative = np.ones(steps) / steps
        initial_state = final_state.copy().read_current_state(split=split_readings)
        delta_knob = (final_state - initial_state).prune(tol=equal_tolerance,
                                                         exceptions=equal_tol_exceptions)
        logt(f'To change: {delta_knob.vars}')
        if delta_knob.is_empty():
            log(f'No changes necessary!')
            return -1
        initial_state_pruned = initial_state.copy().only_keep_shared(delta_knob)
        for step in range(1, steps + 1):
            print(f'{step}/{steps}...', end='')
            if steps_relative[step - 1] > 0:
                partial_sum = steps_relative[:step].sum()
                delta_knob_l = delta_knob * partial_sum
                intermediate_state = initial_state_pruned + delta_knob_l
                intermediate_state.set(verbose=verbose, split_types=False, split=split_settings)
            time.sleep(step_pause)
        print('done!')

        if extra_final_setting:
            time.sleep(0.1)
            final_state.set(verbose=verbose, split=split_settings)

        if retry_endpoint is not None:
            time.sleep(delay_retry_endpoint)
            for i in range(retry_endpoint):
                st = final_state.copy().read_current_state(split=split_readings, settings=True)
                logt(f'\nExtra state read:\n {[(k.acnet_var, k.value) for k in st.vars.values()]}')
                mismatches = (final_state - st).prune(tol=retry_tolerance)
                if i == retry_endpoint - 1:
                    raise SequencerError(
                            f'Transition FAILED - deviations: {mismatches.vars}')
                if mismatches.is_empty():
                    logt(f'Retry {i}/{retry_endpoint} - all settings satisfied!')
                    break
                to_set = final_state.copy().only_keep_shared(mismatches)
                logt(f'Endpoint deviations: {mismatches.vars}')
                to_set.set(verbose=verbose, split=split_settings)
                log(f'Retry {i}/{retry_endpoint} - set {len(to_set.vars)} devices')
                time.sleep(1.1)

        log(f'Knob {final_state.name} set in {time.time() - t0:.5f}s')
        return time.time() - t0


def arm_bpms(mode: str = 'orbit', wait_for_trigger=False, timeout: float = 3):
    """
    Arm bpms for acquisition. DOES NOT do any subsequent triggering.
    For injection, this trigger will be the next kick.
    For orbit, trigger will be next 1s pulse (event A6).

    :param mode: Mode - 'tbt' or 'orbit'
    :param wait_for_trigger: If mode is orbit, waits for next A6 event before returning
    :param timeout: If waiting for trigger, timeout after this many seconds
    """
    if mode == 'orbit':
        bpmorb = StatusDevice(iota.CONTROLS.BPM_ORB_TRIGGER)
        bpmorb.reset()
        if wait_for_trigger:
            assert isinstance(timeout, float) and timeout >= 0
            logger.debug('>>Waiting for next $A6')
            a6cnt = DoubleDevice(iota.CONTROLS.TRIGGER_A6)
            t1 = time.time()
            a6_initial = a6cnt.read()
            while a6cnt.read() == a6_initial:
                if time.time() - t1 > timeout:
                    raise ValueError(f'Waiting for orbit trigger $A6 timed out')
                time.sleep(100)
            logger.debug('>>Got $A6, returning')
    elif mode == 'tbt':
        bpminj = StatusDevice(iota.CONTROLS.BPM_INJ_TRIGGER)
        bpminj.reset()
    else:
        raise ValueError(f'Mode {mode} is not recognized');


def inject_until_current(current: float = 1.0,
                         arm_bpms: bool = False,
                         debug: bool = False,
                         limit: int = 10,
                         read_delay: float = 1.5
                         ):
    """
    Inject until current threshold reached

    :param current: Desired current in mA
    :param arm_bpms: Whether to arm BPMs as part of injection
    :param debug: Print more
    :param limit: Retry limit
    :param read_delay: Delay after injection event before reading current
    """
    ibeam = DoubleDevice(iota.run4.OTHER.BEAM_CURRENT)
    for i in range(limit):
        inject(arm_bpms=arm_bpms, debug=debug)
        time.sleep(read_delay)
        i = -ibeam.read()
        i = i if i > 0 else 0.0
        if i >= current:
            print(f'Injection loop - got {i:.5f} mA (>{current:.3f}) - goal met')
            break
        else:
            print(f'Injection loop - got {i:.5f} mA (<{current:.3f}) - retrying')


def inject(arm_bpms: bool = False, debug: bool = False):
    plc = StatusDevice(iota.CONTROLS.CHIP_PLC)
    vkicker_status = StatusDevice(iota.CONTROLS.VKICKER)
    hkicker_status = StatusDevice(iota.CONTROLS.HKICKER)
    vkicker = DoubleDevice(iota.CONTROLS.VKICKER)
    vkicker_setpoint = DoubleDevice(iota.CONTROLS.VKICKER)
    vkicker_setpoint.drf2.property = DRF_PROPERTY.SETTING
    hkicker = DoubleDevice(iota.CONTROLS.HKICKER)
    hkicker_setpoint = DoubleDevice(iota.CONTROLS.VKICKER)
    hkicker_setpoint.drf2.property = DRF_PROPERTY.SETTING
    a5 = StatusDevice(iota.CONTROLS.TRIGGER_A5)
    a5cnt = DoubleDevice(iota.CONTROLS.TRIGGER_A5)
    a6 = StatusDevice(iota.CONTROLS.TRIGGER_A6)
    a6cnt = DoubleDevice(iota.CONTROLS.TRIGGER_A6)
    vres = StatusDevice(iota.CONTROLS.VKICKER_RESCHARGE)
    vtrig = StatusDevice(iota.CONTROLS.VKICKER_TRIG)
    hres = StatusDevice(iota.CONTROLS.HKICKER_RESCHARGE)
    htrig = StatusDevice(iota.CONTROLS.HKICKER_TRIG)
    # instr = StatusDevice(iota.CONTROLS.INJ_INSTR)
    bpminj = StatusDevice(iota.CONTROLS.BPM_INJ_TRIGGER)
    # bpm_control = StatusDevice(iota.CONTROLS.BPM_CONFIG_DEVICE)
    shutter = StatusDevice(iota.CONTROLS.FAST_LASER_SHUTTER)
    # inj = DoubleDevice(iota.CONTROLS.FAST_LASER_INJECTOR)
    inj_setpoint = DoubleDevice(iota.CONTROLS.FAST_LASER_INJECTOR)
    inj_setpoint.drf2.property = DRF_PROPERTY.SETTING
    log = ConditionalLogger(debug)
    t_start = time.time()

    StatusDeviceSet.read_many([plc, vres, vtrig, vkicker_status,
                               hres, htrig, hkicker_status, a5, a6, shutter])
    DoubleDeviceSet.read_many([a5cnt, a6cnt])

    if debug:
        logger.debug('>>Setting up Chip PLC')
    plc.read()
    if not plc.on:
        plc.set_on_and_verify(retries=6, delay=0.5)
    if not plc.remote:
        plc.set("Trig:Inj")
        time.sleep(2)
        plc.read()
        if not plc.on:
            raise SequencerError(f'CHIP PLC IS NOT ON?')
        if not plc.remote:
            raise SequencerError(f'Chip PLC {iota.CONTROLS.CHIP_PLC} is not remote')

    if debug:
        logger.debug('>>Checking $A6')
    a6.read()
    if not a6.on:
        a6.set_on()

    if debug:
        logger.debug('>>Turning off HKICKER aux devices')
    hres.set_off()
    htrig.set_off()

    if debug:
        logger.debug('>>Turning on VKICKER aux devices')
    vres.set_on()
    vtrig.set_on()

    if debug:
        logger.debug('>>Enabling vertical kicker')
    # vkicker_status.read()
    if not vkicker_status.on:
        vkicker_status.set_on()
    if not vkicker_status.ready:
        vkicker_status.reset()

    vertical_kv = 4.15
    if debug:
        logger.debug('>>Setting vertical kicker')
    # Quick check for settings
    setting = vkicker_setpoint.read()
    if np.abs(setting - vertical_kv) > 1e-3:
        vkicker_setpoint.set(vertical_kv)
        time.sleep(0.1)
        for i in range(100):
            delta_set = np.abs(vkicker_setpoint.read() - vertical_kv)
            delta = np.abs(vkicker.read() - vertical_kv)
            if delta > 0.15 or delta_set > 1e-3:
                if i < 100:
                    time.sleep(0.1)
                    continue
                else:
                    raise SequencerError(
                            f'>>Failed to set kicker - final deltas: {delta} | {delta_set}')
            else:
                if debug:
                    logger.debug(f'>>Kicker setting ok - deltas {delta} | {delta_set}')
                break

    if not shutter.on:
        raise SequencerError("SHUTTER IS CLOSED - ABORT")
    if not shutter.ready:
        raise SequencerError("MPS FAULT - ABORT")

    a6_initial_val = a5cnt.read()
    log(f'>>Initial $A5 cnt: {a6_initial_val}')

    if debug:
        logger.debug('>>Awaiting A8')
    acl = ACL(fallback=True)
    acl._raw_command("wait/nml/event/timeout=2 A8")

    if debug:
        logger.debug('>>Firing')

    if arm_bpms:
        bpminj.reset()
    a6.reset()
    inj_setpoint.set(1)

    t0 = time.time()
    for i in range(20):
        a6_new_val = a6cnt.read()
        if a6_new_val > a6_initial_val:
            if debug:
                log(f'>>$A5 received ({a6_initial_val}->{a6_new_val}) - kick complete'
                    f' in {time.time() - t_start:.2f}s')
            # time.sleep(0.3)
            # vkicker_status.read()
            # if not vkicker_status.on: vkicker_status.set_on_and_verify(retries=10, delay=0.05)
            # if not vkicker_status.ready: vkicker_status.reset_and_verify(retries=10, delay=0.05)
            return
        else:
            if i > 5:
                print(f'>>Awaiting $A6 {i}/{20} ({time.time() - t0:.3f}s)')
            time.sleep(0.05)


def kick(vertical_kv: float = 0.0,
         horizontal_kv: float = 0.0,
         restore: bool = False,
         debug: bool = False,
         silent: bool = False,
         recover_no_verify: bool = False,
         ):
    """
     Perform IOTA kick

     :param vertical_kv: Vertical kicker in acnet units
     :param horizontal_kv: Horizontal kicker in acnet units
     :param restore: Whether to restore PREVIOUS state after kick
     :param debug: Print debug messages
     :param silent: Suppress all messages
     :param recover_no_verify: If true, send on/reset after kick to recover to state of CURRENT KICK

    """
    log = ConditionalLogger(debug)
    assert 4.21 > vertical_kv >= 0.0  # 1.1
    assert 4.26 > horizontal_kv >= 0.0
    t_start = time.time()

    plc = StatusDevice(iota.CONTROLS.CHIP_PLC)
    vkicker_status = StatusDevice(iota.CONTROLS.VKICKER)
    hkicker_status = StatusDevice(iota.CONTROLS.HKICKER)
    vkicker = DoubleDevice(iota.CONTROLS.VKICKER)
    vkicker_setpoint = DoubleDevice(iota.CONTROLS.VKICKER)
    vkicker_setpoint.drf2.property = DRF_PROPERTY.SETTING
    hkicker = DoubleDevice(iota.CONTROLS.HKICKER)
    hkicker_setpoint = DoubleDevice(iota.CONTROLS.HKICKER)
    hkicker_setpoint.drf2.property = DRF_PROPERTY.SETTING
    a5 = StatusDevice(iota.CONTROLS.TRIGGER_A5)
    a5cnt = DoubleDevice(iota.CONTROLS.TRIGGER_A5)
    vres = StatusDevice(iota.CONTROLS.VKICKER_RESCHARGE)
    vtrig = StatusDevice(iota.CONTROLS.VKICKER_TRIG)
    hres = StatusDevice(iota.CONTROLS.HKICKER_RESCHARGE)
    htrig = StatusDevice(iota.CONTROLS.HKICKER_TRIG)
    # instr = StatusDevice(iota.CONTROLS.INJ_INSTR)
    # inj = DoubleDevice(iota.CONTROLS.FAST_LASER_INJECTOR)
    inj_setpoint = DoubleDevice(iota.CONTROLS.FAST_LASER_INJECTOR)
    inj_setpoint.drf2.property = DRF_PROPERTY.SETTING
    # bpminj = StatusDevice(iota.CONTROLS.BPM_INJ_TRIGGER)
    bpm_control = StatusDevice(iota.CONTROLS.BPM_CONFIG_DEVICE)

    readback_tolerance_V = 0.04
    readback_tolerance_H = 0.15

    StatusDeviceSet.read_many([plc, vres, vtrig, vkicker_status,
                               hres, htrig, hkicker_status, a5])
    DoubleDeviceSet.read_many([vkicker, hkicker])
    DoubleDeviceSet.read_many([vkicker_setpoint, hkicker_setpoint], setting=True)

    do_h = horizontal_kv > 0.0
    do_v = vertical_kv > 0.0

    h_initial = hkicker_setpoint.value
    initial_hkicker_status_on = hkicker_status.on
    v_initial = vkicker_setpoint.value
    initial_vkicker_status_on = vkicker_status.on

    log('>>Setting up Chip PLC')
    if not plc.on:
        plc.set_on_and_verify(retries=6, delay=0.5)
    if plc.remote:
        plc.set("Trig:Circ")
        time.sleep(1)
        plc.read()
        if not plc.local:
            time.sleep(2)
            plc.read()
            assert plc.local, f'Chip PLC {iota.CONTROLS.CHIP_PLC} did not transition to LOCAL mode'
            assert plc.ready, f'Chip PLC {iota.CONTROLS.CHIP_PLC} is not ready'

    if vertical_kv > 0.0:
        delta_setpt = np.abs(vkicker_setpoint.value - vertical_kv)
        if delta_setpt > 1.0e-3:
            vkicker_setpoint.set(vertical_kv)

    if horizontal_kv > 0.0:
        delta_setpt = np.abs(hkicker_setpoint.value - horizontal_kv)
        if delta_setpt > 1.0e-3:
            hkicker_setpoint.set(horizontal_kv)

    if vertical_kv > 0.0:
        log('>>Enabling vertical kicker')
        vres_flag = not vres.on
        vtrig_flag = not vtrig.on
        vkrdy_flag = not vkicker_status.ready
        #
        # if vres_flag:
        #     vres.set_on()
        # if vtrig_flag:
        #     vtrig.set_on()
        # if vkrdy_flag:
        #     vkicker_status.reset()
        # if vres_flag:
        #     vres.set_on_and_verify(retries=60, delay=0.10, no_initial_set=True,
        #                            check_first=True)
        # if vtrig_flag:
        #     vtrig.set_on_and_verify(retries=60, delay=0.10, no_initial_set=True,
        #                             check_first=True)
        # if vkrdy_flag:
        #     vkicker_status.reset_and_verify(retries=60, delay=0.10, no_initial_set=True,
        #                                     check_first=True)

        if vres_flag:
            vres.set_on_and_verify(retries=60, delay=0.15)
        if vtrig_flag:
            vtrig.set_on_and_verify(retries=60, delay=0.15)
        if vkrdy_flag:
            vkicker_status.reset_and_verify(retries=60, delay=0.15)
        time.sleep(0.05)
        vkicker_status.read()
        if not vkicker_status.on:
            vkicker_status.set_on_and_verify(retries=60, delay=0.15)
    else:
        log('>>Turning off VKICKER aux devices')
        if vres.on:
            vres.set_off_and_verify(retries=100, delay=0.2)
        if vtrig.on:
            vtrig.set_off_and_verify(retries=100, delay=0.2)

    if horizontal_kv > 0.0:
        log('>>Enabling horizontal kicker')
        if not hres.on:
            hres.set_on_and_verify(retries=60, delay=0.15)
        if not htrig.on:
            htrig.set_on_and_verify(retries=60, delay=0.15)
        if not hkicker_status.ready:
            hkicker_status.reset_and_verify(retries=60, delay=0.15)
        time.sleep(0.05)
        hkicker_status.read()
        if not hkicker_status.on:
            hkicker_status.set_on_and_verify(retries=60, delay=0.15)
    else:
        log('>>Turning off HKICKER aux devices')
        if hres.on:
            hres.set_off_and_verify(retries=100, delay=0.2)
        if htrig.on:
            htrig.set_off_and_verify(retries=100, delay=0.2)

    log('>>Checking $A5')
    if not a5.on:
        a5.set_on_and_verify(retries=40, delay=0.10)

    if do_v:
        log('>>Setting vertical kicker')
        setpt_ok = rb_ok = False
        delta_setpt = delta = None
        for i in range(200):
            if not setpt_ok:
                delta_setpt = np.abs(vkicker_setpoint.read() - vertical_kv)
                if delta_setpt > 1.0e-3:
                    if i < 50:
                        time.sleep(0.05)
                    else:
                        vkicker_setpoint.set(vertical_kv)
                        time.sleep(0.2)
                else:
                    setpt_ok = True

            if not rb_ok:
                delta = np.abs(vkicker.read() - vertical_kv)
                if delta > readback_tolerance_V:
                    if i < 50:
                        time.sleep(0.05)
                    else:
                        vkicker_status.read()
                        if not vkicker_status.ready:
                            vkicker_status.reset_and_verify(retries=60, delay=0.15)
                        if not vkicker_status.on:
                            vkicker_status.set_on_and_verify(retries=60, delay=0.15)
                        vkicker_setpoint.set(vertical_kv)
                        time.sleep(0.2)
                else:
                    rb_ok = True

            if setpt_ok and rb_ok:
                log(f'>>VKicker set ok - {vertical_kv=}|{delta=}|{delta_setpt}')
                break

            if i == 199:
                raise SequencerError(f'>>Failed to set {vertical_kv=} - difference:'
                                     f' {delta=} | {delta_setpt=} | {setpt_ok=} | {rb_ok=}')

    # if horizontal_kv > 0.0:
    #     log('>>Setting horizontal kicker')
    #     for i in range(100):
    #         delta = np.abs(hkicker.read() - horizontal_kv)
    #         delta_setpt = np.abs(hkicker_setpoint.read() - horizontal_kv)
    #         if delta > readback_tolerance_H or delta_setpt > 1e-3:
    #             hkicker_setpoint.set(horizontal_kv)
    #         else:
    #             log(f'>>Kicker set ok - {horizontal_kv}|{delta}|{delta_setpt}')
    #             break
    #         if i < 30:
    #             if i > 20:
    #                 hkicker_setpoint.set(horizontal_kv)
    #                 time.sleep(0.1)
    #             else:
    #                 time.sleep(0.05)
    #             continue
    #         else:
    #             raise SequencerError(f'>>Failed to set kicker - {delta=} | {delta_setpt=}')

    if do_h:
        log('>>Setting horizontal kicker')
        setpt_ok = rb_ok = False
        delta_setpt = delta = None
        for i in range(200):
            if not setpt_ok:
                delta_setpt = np.abs(hkicker_setpoint.read() - horizontal_kv)
                if delta_setpt > 1.0e-3:
                    if i < 50:
                        time.sleep(0.05)
                    else:
                        hkicker_setpoint.set(horizontal_kv)
                        time.sleep(0.2)
                else:
                    setpt_ok = True

            if not rb_ok:
                delta = np.abs(hkicker.read() - horizontal_kv)
                if delta > readback_tolerance_H:
                    if i < 50:
                        time.sleep(0.05)
                    else:
                        hkicker_status.read()
                        if not hkicker_status.ready:
                            hkicker_status.reset_and_verify(retries=60, delay=0.15)
                        if not hkicker_status.on:
                            hkicker_status.set_on_and_verify(retries=60, delay=0.15)
                        hkicker_setpoint.set(horizontal_kv)
                        time.sleep(0.2)
                else:
                    rb_ok = True

            if setpt_ok and rb_ok:
                log(f'>>HKicker set ok - {horizontal_kv=}|{delta=}|{delta_setpt}')
                break

            if i == 199:
                raise SequencerError(f'>>Failed to set {horizontal_kv=} - difference:'
                                     f' {delta=} | {delta_setpt=} | {setpt_ok=} | {rb_ok=}')

    # Extra checks since things are weirdly unreliable!!!
    read_set = []
    if do_v:
        read_set.extend([vkicker_status, vres, vtrig])
    if do_h:
        read_set.extend([hkicker_status, hres, htrig])
    if len(read_set) > 0:
        StatusDeviceSet.read_many(read_set)
    if do_v:
        log('>>Checking vertical kicker')
        if not vkicker_status.on or not vkicker_status.ready or not vres.on or not vtrig.on:
            raise SequencerError("VKICK not ready")
    else:
        if vres.on or vtrig.on:
            raise SequencerError("VKICK is still enabled")
    if do_h:
        log('>>Checking horizontal kicker')
        if not hkicker_status.on or not hkicker_status.ready or not hres.on or not htrig.on:
            raise SequencerError("HKICK not ready")
    else:
        if hres.on or htrig.on:
            raise SequencerError("VKICK is still enabled")

    time.sleep(0.1)

    a5_initial_val = a5cnt.read()
    log(f'>>Initial $A5 cnt: {a5_initial_val}')

    plc.read()
    if plc.remote:
        raise SequencerError("PLCCCCCCCCCC CHIP FIX PLZ")

    log('>>Awaiting A8')
    acl = ACL(fallback=True)
    acl._raw_command("wait/nml/event/timeout=3 A8")

    log('>>Arming')
    bpm_control.set('Arm Injection')

    # bpminj.reset()
    time.sleep(0.01)
    log('>>Firing')
    a5.reset()
    inj_setpoint.set(1)

    # Await actual fire event
    t0 = time.time()
    good = False
    for i in range(20):
        a5_new_val = a5cnt.read()
        if a5_new_val > a5_initial_val:
            if not silent:
                logger.info(f'>>$A5 received ({a5_initial_val}->{a5_new_val}) - kick complete'
                            f' in {time.time() - t_start:.2f}s')
            # time.sleep(0.3)
            # vkicker_status.read()
            # if not vkicker_status.on:
            #     vkicker_status.set_on_and_verify(retries=10, delay=0.05)
            # if not vkicker_status.ready:
            #     vkicker_status.reset_and_verify(retries=10, delay=0.05)
            good = True
            break
        else:
            if i > 5:
                logger.info(f'>>Awaiting $A5 {i}/{20} ({time.time() - t0:.3f}s)')
            time.sleep(0.02)

    if not good:
        raise SequencerError(f'$A5 never received (cnt: {a5_initial_val}) - something went wrong!')

    if restore:
        if do_v:
            if initial_vkicker_status_on:
                vkicker_status.set_on_and_verify()
            else:
                vkicker_status.set_off_and_verify()
            vkicker_setpoint.set(v_initial)
            logger.info(f'Restoring VK to {v_initial=}|{initial_vkicker_status_on=}')
        if do_h:
            if initial_hkicker_status_on:
                hkicker_status.set_on_and_verify()
            else:
                hkicker_status.set_off_and_verify()
            hkicker_setpoint.set(h_initial)
            logger.info(f'Restoring HK to {h_initial=}|{initial_hkicker_status_on=}')

    if recover_no_verify and not restore:
        if do_v:
            vkicker_status.reset()
            # vres.set_on()
            # vtrig.set_on()
            time.sleep(0.01)
            # vkicker_status.set_on()
            log(f'Restoring noverify VK')
        if do_h:
            # H doesnt trip on kick
            # hkicker_status.set_on()
            # hres.set_on()
            # htrig.set_on()
            log(f'Restoring noverify HK')


def test_sequence():
    log = ConditionalLogger(True)
    plc = StatusDevice(iota.CONTROLS.CHIP_PLC)
    plc.read()
    log(f'Test sequence ok')

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
