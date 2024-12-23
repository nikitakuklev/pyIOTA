import collections
import copy
import json
import logging
import time
import uuid
from abc import ABC
from typing import Any, Callable, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd

from .data import ACNETResponse, ArrayDataResponse, DoubleDataResponse, StatusDataResponse, \
    ACNETErrorResponse
from .drf2 import DRF_EXTRA, DRF_PROPERTY, DRF_RANGE, DiscreteRequest, get_default_field, \
    get_qualified_device, \
    parse_request
from .errors import ACNETConfigError

if TYPE_CHECKING:
    from .adapters import Adapter

logger = logging.getLogger(__name__)

data_entry = collections.namedtuple('DataEntry', ['id', 'value', 'ts', 'src'])

__all__ = ['DoubleDevice', 'DoubleDeviceSet',
           'ArrayDevice', 'ArrayDeviceSet']


class Device(ABC):
    """
    Basic building block of the ACNET system. All channels are devices, and inherit from
    base Device class. It should not be used directly.
    """

    def __init__(self, name: str, drf2: DiscreteRequest, adapter: 'Adapter' = None,
                 history: int = 10
                 ):
        self._name: str = name
        self._drf2: DiscreteRequest = drf2
        self.adapter = adapter
        self.history_len: int = history
        self.history = collections.deque(maxlen=history)
        self.update_cnt: int = 0
        self.access_method: Optional[str] = None
        self.last_update: Optional[float] = None
        self.last_response = None
        self.debug: bool = False
        self._device_set: Optional[DeviceSet] = None
        # self._device_set_internal: Optional[DeviceSet] = None
        # self.log = pd.DataFrame({'time': pd.Series([], dtype='datetime64[ns]'),
        #                          'value': pd.Series([], dtype=np.float64),
        #                          'unit': pd.Series([], dtype=str)
        #                          })

    @property
    def name(self):
        return self._name

    @property
    def drf2(self):
        return self._drf2

    @property
    def device_set(self):
        return self._device_set

    def get_history(self) -> list:
        return list(self.history)

    def update(self, v: ACNETResponse, source: str):
        self.last_update = v.timestamp
        self.last_response = v
        self.history.append(v)
        self.update_cnt += 1

    def copy(self):
        c = copy.deepcopy(self)
        c._device_set = None
        return c

    @property
    def value_string(self):
        return str(self.last_response)

    def __str__(self) -> str:
        return f'Device [{self.name}]: {self.value_string=}'

    def __repr__(self) -> str:
        return f'Device [{self.name}]: {self.value_string=}'

class RawDevice(Device):
    def __init__(self, name: str, adapter=None, history=10):
        self.setpoint = None
        self.readback = None
        self.error = None
        drf2 = parse_request(name)
        assert drf2.property in [DRF_PROPERTY.READING, DRF_PROPERTY.SETTING]
        super().__init__(name, drf2, adapter, history)

    @property
    def value(self):
        if self.drf2.is_reading:
            return self.readback
        else:
            return self.setpoint

    def dump(self):
        return self.value

    def update(self, v: DoubleDataResponse, source: str):
        value = None
        if isinstance(v, DoubleDataResponse):
            if isinstance(v.data, list):
                value = v.data
            else:
                value = v.data
            self.error = None
        elif isinstance(v, ACNETErrorResponse):
            self.error = v
        if self.drf2.is_reading:
            self.readback = value
        else:
            self.setpoint = value

        super().update(v, source)

    def update_log(self, df_log):
        try:
            self.log = pd.concat([self.log, df_log], axis=0, ignore_index=True)
            duplicates = self.log.duplicated('time', keep=False)
            duplicates_all = self.log.duplicated(keep=False)
            assert duplicates.sum() == duplicates_all.sum()
            self.log.drop_duplicates(inplace=True)
            assert self.log.loc[:, 'time'].is_unique
        except Exception as e:
            raise ValueError(f'Device {self.name} - log {df_log} add failed - {e}')

    def read(self, adapter=None, **kwargs):
        adapter = adapter or self.adapter
        ds = DoubleDeviceSet(members=[self], adapter=adapter)
        return ds.read(**kwargs)[0]

    def read_readback(self, adapter=None, **kwargs):
        is_reading = self.drf2.is_reading
        if not is_reading:
            d = self.copy()
            d.drf2.property = DRF_PROPERTY.READING
        else:
            d = self
        ds = DoubleDeviceSet(members=[d], adapter=adapter, settings=False)
        # if is_reading:
        #     self.drf2.property = DRF_PROPERTY.READING
        # else:
        #     self.drf2.property = DRF_PROPERTY.SETTING
        return ds.read(**kwargs)[0]

    def read_setpoint(self, adapter=None, **kwargs):
        is_reading = self.drf2.is_reading
        if is_reading:
            d = self.copy()
            d.drf2.property = DRF_PROPERTY.READING
        else:
            d = self
        ds = DoubleDeviceSet(members=[d], adapter=adapter, settings=True)
        # if is_reading:
        #     self.drf2.property = DRF_PROPERTY.READING
        # else:
        #     self.drf2.property = DRF_PROPERTY.SETTING
        return ds.read(**kwargs)[0]

    def set(self, value: Any, adapter=None, **kwargs):
        ds = DoubleDeviceSet(members=[self], adapter=adapter)
        r = ds.set([value], **kwargs)
        return r

class DoubleDevice(Device):
    def __init__(self, name: str, adapter=None, history=10):
        self.setpoint = None
        self.readback = None
        self.error = None
        drf2 = parse_request(name)
        assert drf2.property in [DRF_PROPERTY.READING, DRF_PROPERTY.SETTING]
        super().__init__(name, drf2, adapter, history)

    @property
    def value(self):
        if self.drf2.is_reading:
            return self.readback
        else:
            return self.setpoint

    def dump(self):
        return self.value

    def update(self, v: DoubleDataResponse, source: str):
        value = None
        if isinstance(v, DoubleDataResponse):
            if isinstance(v.data, list):
                value = v.data
            else:
                value = float(v.data)
            self.error = None
        elif isinstance(v, ACNETErrorResponse):
            self.error = v
        # elif isinstance(v, AcnetStatusResponse):
        #     value = None
        if self.drf2.is_reading:
            self.readback = value
        else:
            self.setpoint = value
            # if self.drf2.extra is not None and self.drf2.extra == DRF_EXTRA.FTP:
            #     if isinstance(v.data, list):
            #         self.value = v.data
            #     else:
            #         self.value = float(v.data)
            # else:
            #     self.value = float(v.data)

        super().update(v, source)

    def update_log(self, df_log):
        try:
            self.log = pd.concat([self.log, df_log], axis=0, ignore_index=True)
            duplicates = self.log.duplicated('time', keep=False)
            duplicates_all = self.log.duplicated(keep=False)
            assert duplicates.sum() == duplicates_all.sum()
            self.log.drop_duplicates(inplace=True)
            assert self.log.loc[:, 'time'].is_unique
        except Exception as e:
            raise ValueError(f'Device {self.name} - log {df_log} add failed - {e}')

    def read(self, adapter=None, **kwargs):
        adapter = adapter or self.adapter
        ds = DoubleDeviceSet(members=[self], adapter=adapter)
        return ds.read(**kwargs)[0]

    def read_readback(self, adapter=None, **kwargs):
        is_reading = self.drf2.is_reading
        if not is_reading:
            d = self.copy()
            d.drf2.property = DRF_PROPERTY.READING
        else:
            d = self
        ds = DoubleDeviceSet(members=[d], adapter=adapter, settings=False)
        # if is_reading:
        #     self.drf2.property = DRF_PROPERTY.READING
        # else:
        #     self.drf2.property = DRF_PROPERTY.SETTING
        return ds.read(**kwargs)[0]

    def read_setpoint(self, adapter=None, **kwargs):
        is_reading = self.drf2.is_reading
        if is_reading:
            d = self.copy()
            d.drf2.property = DRF_PROPERTY.READING
        else:
            d = self
        ds = DoubleDeviceSet(members=[d], adapter=adapter, settings=True)
        # if is_reading:
        #     self.drf2.property = DRF_PROPERTY.READING
        # else:
        #     self.drf2.property = DRF_PROPERTY.SETTING
        return ds.read(**kwargs)[0]

    def set(self, value: Any, adapter=None, **kwargs):
        ds = DoubleDeviceSet(members=[self], adapter=adapter)
        r = ds.set([value], **kwargs)
        return r


class StatusDevice(Device):
    def __init__(self, name, adapter=None, history: int = 10, auto_drf: bool = True):
        self.value = None
        self.on: Optional[bool] = None
        self.ready: Optional[bool] = None
        self.remote: Optional[bool] = None
        self.positive: Optional[bool] = None
        self.ramp: Optional[bool] = None
        drf2 = parse_request(name)
        if auto_drf:
            drf2.property = DRF_PROPERTY.STATUS
            drf2.field = get_default_field(drf2.property)
            drf2.event = None
        else:
            assert drf2.property in [DRF_PROPERTY.STATUS, DRF_PROPERTY.CONTROL]
        super().__init__(name, drf2, adapter, history)

    @property
    def off(self):
        return not self.on

    @property
    def tripped(self):
        return not self.ready

    @property
    def local(self):
        return not self.remote

    def dump(self):
        return json.dumps({k: v for k, v in self.value.items() if k in ['on', 'ready', 'remote',
                                                                        'positive', 'ramp']})

    def update(self, v: StatusDataResponse, source: str):
        if isinstance(v, StatusDataResponse):
            self.value = v.data
            self.on = v.data.get('on', None)
            self.ready = v.data.get('ready', None)  # DPM can be missing
            self.remote = v.data.get('remote', None)  # DPM doesnt provide it
            self.positive = v.data.get('positive', None)  # DPM can be missing
            self.ramp = v.data.get('ramp', None)  # DPM doesnt provide it
            if self.debug:
                print(f'Status device: {self.on=} | {self.ready=} | {self.remote=}| '
                      f'{self.positive=} | {self.ramp=}')
        elif isinstance(v, ACNETErrorResponse):
            self.value = self.on = self.ready = self.remote = self.positive = self.ramp = None
        super().update(v, source)

    def check_value(self):
        if self.value is None:
            raise ValueError(f'Device {self.name} has no value')

    def is_on(self) -> bool:
        self.check_value()
        return self.on

    def is_off(self) -> bool:
        self.check_value()
        return not self.on

    def is_ready(self) -> bool:
        self.check_value()
        return self.ready

    def read(self, adapter=None, **kwargs):
        if not self._device_set or (adapter and self._device_set.adapter != adapter):
            self._device_set = StatusDeviceSet(members=[self], adapter=adapter)
        return self._device_set.read(**kwargs)[0]

    def set(self, value, adapter=None, **kwargs):
        # We have special ones that take custom strings
        # assert value in ['RESET', 'ON', 'OFF', 'POS', 'NEG', 'RAMP', 'DC',
        #                  'LOCAL', 'REMOTE', 'TRIP']
        if not self._device_set or (adapter and self._device_set.adapter != adapter) or True:
            self._device_set = StatusDeviceSet(members=[self], adapter=adapter)
        r = self._device_set.set([value], **kwargs)
        return r[0]

    def set_on(self):
        self.set('ON')

    def set_on_and_verify(self, retries=3, delay=0.02, check_first=False,
                          no_initial_set=False, idelay=None
                          ):
        idelay = idelay or delay
        if check_first:
            self.read()
            if self.is_on():
                return
        if not no_initial_set:
            self.set('ON')
        time.sleep(idelay)
        for i in range(retries):
            self.read()
            if self.is_on():
                return
            else:
                if i > retries / 2:
                    self.set('ON')
            time.sleep(delay)
        raise Exception(f'{self.name} - ON verification failure, value {self.value_string}')

    def set_off(self):
        self.set('OFF')

    def set_off_and_verify(self, retries=3, delay=0.02, check_first=False,
                           no_initial_set=False, idelay=None,
                           resend_setpoint_after: Optional[int] = None
                           ):
        idelay = idelay or delay
        if check_first:
            self.read()
            if self.is_off():
                return
        if not no_initial_set:
            self.set('OFF')
        time.sleep(idelay)
        resend_setpoint_after = resend_setpoint_after or retries / 4
        for i in range(retries):
            self.read()
            if self.is_off():
                return
            else:
                if i > resend_setpoint_after:
                    self.set('OFF')
            time.sleep(delay)
        raise Exception(f'{self.name} - OFF verification failure, value {self.value_string}')

    def reset(self):
        self.set('RESET')

    def reset_and_verify(self, retries=3, delay=0.02, check_first=False,
                         no_initial_set=False, idelay=None
                         ):
        idelay = idelay or delay
        if check_first:
            self.read()
            if self.is_ready():
                return
        if not no_initial_set:
            self.set('RESET')
        time.sleep(idelay)
        for i in range(retries):
            self.read()
            if self.is_ready():
                return
            else:
                if i > retries / 2:
                    self.set('RESET')
            time.sleep(delay)
        raise Exception(f'{self.name} - OFF verification failure, value {self.value_string}')

    def as_control(self):
        name = get_qualified_device(self.drf2.device, DRF_PROPERTY.CONTROL)
        return StatusDevice(name)

    def as_status(self):
        name = get_qualified_device(self.drf2.device, DRF_PROPERTY.CONTROL)
        return StatusDevice(name)


class ArrayDevice(Device):
    def __init__(self, name, adapter=None, history=10, auto_drf: bool = True):
        self.error = None
        self.value = None
        drf2 = parse_request(name)
        assert drf2.property in [DRF_PROPERTY.READING, DRF_PROPERTY.SETTING]
        # Fix lack of range specifier
        if auto_drf and drf2.range is None:
            drf2.range = DRF_RANGE('full')
        super().__init__(name, drf2, adapter, history)

    def read(self, adapter=None, **kwargs):
        if not self._device_set:
            ds = ArrayDeviceSet(members=[self], adapter=adapter)
        else:
            ds = self._device_set
        return ds.read(**kwargs)[0]

    def set(self, value, **kwargs):
        raise NotImplementedError(f'Arrays can only be read for now (since I never needed to set '
                                  f'them...poke me if needed)')

    def dump(self):
        return self.value

    def update(self, v: ArrayDataResponse, source: str):
        if isinstance(v, ArrayDataResponse):
            self.value = v.data
            self.error = None
        elif isinstance(v, ACNETErrorResponse):
            self.value = None
            self.error = v
        super().update(v, source)

    def __str__(self) -> str:
        if self.value_string is None:
            return f'{self.__class__.__name__} {self.name}:<NO DATA>'
        else:
            return f'{self.__class__.__name__} {self.name}:{self.value_string[:4]}...'

    def __repr__(self) -> str:
        return self.__str__()


class RawArrayDevice(ArrayDevice):
    pass


#####


class DeviceSet:
    """
    A container of devices and other device sets, which is subclassed by more specialized sets. Implemented as
    an unbalanced tree, with 'nodes' containing only other device sets and 'leafs' containing only devices.
    Supports iterable and dictionary access modes.
    """

    def __init__(self, members: list, name: str, adapter: 'Adapter' = None, method="oneshot"):
        from .adapters import AdapterManager
        self.adapter = adapter or AdapterManager.get_default()
        node = all([isinstance(ds, DeviceSet) for ds in members])
        leaf = all([isinstance(ds, Device) for ds in members])
        assert node or leaf
        name = name or uuid.uuid4().hex[:10]
        self.devices: Optional[dict[str, Device]] = None

        if node:
            self.leaf = False
            self.children = {ds.name: ds for ds in members}
            assert len(self.children) == len(members)
            self.devices = None
            for c in self.children:
                c.parent = self
        else:
            self.leaf = True
            # for d in members:
            #    if d._device_set is not None:
            #        raise ACNETConfigError(f'Device {d} already bound to adapter {d._device_set}')
            self.devices = {d.name: d for d in members}
            if not len(self.devices) == len(members):
                un, uc = np.unique(list([d.name for d in members]), return_counts=True)
                raise ACNETConfigError(
                        f'Non-unique devices in the list: {un}, {un[uc > 1]}, {uc},'
                        f' {len(self.devices)}, {len(members)}')
            # for d in members:
            #    d._device_set = self
            self.children = None
            self.parent = None
        self.name = name
        self.method = method

    def __iter__(self):
        self.iter_node = self
        return self

    def __next__(self) -> Device:
        """
        Depth-first pre-order traversal
        :return:
        """
        if not self.leaf:
            for c in self.children:
                yield next(c)
            raise StopIteration
        else:
            for d in self.devices.values():
                print(d)
                time.sleep(1)
                yield d
            raise StopIteration

    def __getitem__(self, item):
        return self.devices[item]

    def add(self, device: Device) -> None:
        if isinstance(device, Device):
            if device.name not in self.devices.keys():
                self.devices[device.name] = device
            else:
                raise AttributeError('This device is already in the set')
        else:
            raise AttributeError('You must add a device object!')

    def remove(self, name: str) -> Device:
        """
        Removes and returns the desired device or device set.
        Raises exception if not found
        :param name: Unique name of the element
        :return: The removed member
        """
        if self.leaf:
            if name in self.devices:
                return self.devices.pop(name)
            else:
                raise ValueError(f'No such device in set: {name}')
        else:
            if name in self.children:
                return self.children.pop(name)
            else:
                raise ValueError(f'No such device set: {name}')

    def check_acquisition_available(self, method: str) -> bool:
        assert method
        if not self.adapter:
            raise AttributeError('Adapter is not set!')
        if self.leaf:
            if len(self.devices) == 0:
                raise Exception("Add devices before starting acquisition!")
            return self.adapter.check_available(self.devices, method)
        else:
            return all([c.check_acquisition_available(method) for c in self.children])

    def check_acquisition_supported(self, method: str) -> bool:
        assert method
        if self.adapter is None:
            raise AttributeError('Adapter is not set!')
        if self.leaf:
            return self.adapter.check_supported(self.devices, method)
        else:
            return all([c.check_acquisition_supported(method) for c in self.children])

    def set(self, values: list, adapter: 'Adapter' = None,
            full: bool = False, verbose: bool = False, split: bool = False
            ) -> list:
        if not self.adapter.check_setting_supported():
            raise Exception(f'Setting is not support for adapter {self.adapter.__class__}')
        if not self.devices:
            raise AttributeError("Cannot start acquisition on an empty set")
        if len(values) != len(self.devices):
            raise AttributeError(f"Value list length {len(values)} does not match number of "
                                 f"devices in set {len(self.devices)}")
        if self.leaf:
            a = adapter or self.adapter
            values = a.set(self, values, full=full, verbose=verbose, split=split)
        else:
            raise ValueError("Tree traversing writes are not supported")
        return values

    def read(self, adapter: 'Adapter' = None, full: bool = False, verbose: bool = False,
             split: bool = False, accept_null: bool = True
             ) -> list:
        if not self.check_acquisition_supported(method='oneshot'):
            raise Exception(f'Acquisition not supported - have {self.adapter.supported}')
        if not self.devices:
            raise AttributeError("Cannot start acquisition on an empty set")
        if self.leaf:
            a = adapter or self.adapter
            values = a.read(self, full=full, verbose=verbose, split=split, accept_null=accept_null)
        else:
            values = []
            for c in self.children:
                values += c.update(c.adapter.oneshot())
        return values

    def monitor(self, callback: Callable = None):
        """
        Start monitoring the devices in the set. Depending on the DRF specification, either one or more
        updates will arrive in response to each specification.
        :param callback: callback for when new data arrives
        """
        if not self.check_acquisition_supported('monitoring'):
            raise Exception('Acquisition is not supported for method: {}'.format('polling'))
        if not self.check_acquisition_available('monitoring'):
            return False
        if not self.devices:
            raise AttributeError("Cannot start acquisition on an empty set")
        if self.leaf:
            return self.adapter.subscribe(self, callback)
        else:
            raise NotImplementedError

    def start_polling(self) -> bool:
        if not self.check_acquisition_supported('polling'):
            raise Exception('Acquisition is not supported for method: {}'.format('polling'))
        if not self.check_acquisition_available('polling'):
            return False
        if not self.devices:
            raise AttributeError("Cannot start acquisition on an empty set")
        if self.adapter.running:
            print('Acquisition is already running!')
            return False
        if self.leaf:
            self.adapter.start()
        else:
            for c in self.children:
                c.adapter.start()
        return True

    def stop_polling(self) -> bool:
        if not self.adapter.running:
            print('Acquisition is not running!')
            return False
        if self.leaf:
            return self.adapter.stop()
        else:
            success = True
            for c in self.children:
                success &= c.adapter.stop()
            return success


class DoubleDeviceSet(DeviceSet):
    def __init__(self, members: list,
                 name: str = None,
                 adapter: 'Adapter' = None,
                 settings: Optional[bool] = None
                 ):
        assert isinstance(members, list), 'a list of devices must be passed'
        if all([isinstance(d, str) for d in members]):
            members = [DoubleDevice(d) for d in members]
        elif all([isinstance(ds, DoubleDevice) for ds in members]):
            pass
        else:
            raise AttributeError(
                    f'{self.__class__.__name__} can only contain devices - use DeviceSet')
        if settings is not None:
            if settings:
                for d in members:
                    d.drf2.property = DRF_PROPERTY.SETTING
            else:
                for d in members:
                    if d.drf2.property == DRF_PROPERTY.SETTING:
                        logger.warning(f'Device {d.name} is a SETTING, but read_many will force a READING unless'
                                       f' settings=True is passed')
                    d.drf2.property = DRF_PROPERTY.READING
        super().__init__(members, name, adapter)

    def add(self, device: DoubleDevice):
        if isinstance(device, DoubleDevice):
            super().add(device)
        else:
            raise Exception("Device is not a double!")

    @staticmethod
    def read_many(devices: list[Device], setting=False):
        """ Static method to read many devices as a single shot without creating a set """
        ds = DoubleDeviceSet(members=devices, settings=setting)
        return ds.read()

    @staticmethod
    def read_many_strings(device_strings: list[str], setting=False):
        """ Static method to read many devices as a single shot without creating a set """
        devices = []
        for s in device_strings:
            devices.append(DoubleDevice(s))
        ds = DoubleDeviceSet(members=devices, settings=setting)
        return ds.read()


class ArrayDeviceSet(DeviceSet):
    """
    Leaf container of BPM and other array devices.
    """

    def __init__(self, members: list, name: str = None, adapter=None,
                 enforce_array_length: int = None
                 ):
        if all([isinstance(d, str) for d in members]):
            members = [ArrayDevice(d) for d in members]
        elif all([isinstance(ds, ArrayDevice) for ds in members]):
            pass
        else:
            raise AttributeError(
                    f'{self.__class__.__name__} can only contain devices - use DeviceSet for coalescing')
        super().__init__(members, name, adapter)
        self.array_length = enforce_array_length

    def add(self, device: ArrayDevice):
        if isinstance(device, ArrayDevice):
            super().add(device)
        else:
            raise Exception("Device is not an array!")


class StatusDeviceSet(DeviceSet):
    def __init__(self, members: list, name: str = None, adapter=None):
        if all([isinstance(d, str) for d in members]):
            members = [StatusDevice(d) for d in members]
        elif all([isinstance(d, StatusDevice) for d in members]):
            pass
        else:
            print([isinstance(m, StatusDevice) for m in members])
            print(members, [type(m) for m in members])
            raise AttributeError(
                    f'{self.__class__.__name__} can only contain devices - use DeviceSet for coalescing')
        super().__init__(members, name, adapter)

    def add(self, device: Device):
        if isinstance(device, StatusDevice):
            super().add(device)
        else:
            raise Exception("Incorrect device type!")

    @staticmethod
    def read_many(devices: list[Device]):
        ds = StatusDeviceSet(members=devices)
        return ds.read()


DEVICE_TO_SET = {DoubleDevice: DoubleDeviceSet}
