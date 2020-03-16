import asyncio
import base64
import collections
import sys
import datetime
import time
from typing import Optional, Tuple

import numpy as np

data_entry = collections.namedtuple('DataEntry', ['id', 'value', 'ts', 'src'])

default_adapter = None


# class CommManager:
#     default_adapter = None


class Device:

    def __init__(self, name: str, drf2: str, history: int = 10):
        self.name: str = name
        self.drf2: str = drf2
        self.history_len: int = history
        self.history = collections.deque(maxlen=history)
        self.update_cnt: int = 0
        self.access_method: Optional[str] = None
        self.value_string: Optional[str] = None
        self.value_tuple: Optional[Tuple] = None
        self.last_update: Optional[float] = None
        self.value = None
        self.debug: bool = False
        self._device_set: Optional[DeviceSet] = None

    def get_history(self) -> list:
        return list(self.history)

    def update(self, data: str, timestamp: float, source: str):
        self.value_string = data
        self.value_tuple = data_entry(self.update_cnt, self.value_string, timestamp, source)
        self.last_update = timestamp
        self.history.append(self.value_tuple)
        self.update_cnt += 1

    def __str__(self) -> str:
        return f'Device {self.name}: {self.value_string}'

    def __repr__(self) -> str:
        return f'Device {self.name}: {self.value_string}'


class DoubleDevice(Device):
    def __init__(self, name: str, drf2: str = None, history=10):
        """
        Floating point device
        """
        self.value = None
        super().__init__(name, drf2, history)

    def update(self, data, timestamp, source: str):
        try:
            self.value = float(data)
        except:
            raise
        super().update(data, timestamp, source)

    def read(self, adapter=None, verbose: bool = False, setting: bool = False):
        if not self._device_set:
            adapter = adapter or default_adapter or ACL(fallback=True)
            DoubleDeviceSet(self.name, members=[self], adapter=adapter).readonce(settings=setting)
        else:
            self._device_set.readonce()
        return self.value

    def set(self, value, adapter=None):
        if not self._device_set:
            adapter = adapter or default_adapter or ACL(fallback=True)
            resp = DoubleDeviceSet(self.name, members=[self], adapter=adapter).set([value])
        else:
            resp = self._device_set.set([value])
        return resp


class StatusDevice(Device):
    def __init__(self, name, drf2=None, history=10):
        """
        Basic status device
        """
        self.on = None
        self.ready = None
        self.remote = None
        super().__init__(name, drf2, history)

    def update(self, data, timestamp: float, source: str):
        v = data
        if self.debug:
            print(f'Status device {self.name} update {data}')
        self.value = v
        assert ('ON' in v) != ('OFF' in v)
        self.on = 'ON' in v
        assert ('READY' in v) != ('TRIPPED' in v)
        self.ready = 'READY' in v
        assert ('LOCAL' in v) != ('REMOTE' in v)
        self.remote = 'REMOTE' in v
        super().update(data, timestamp, source)
        if self.debug:
            print(f'Status device on|ready|rem: {self.on} | {self.ready} | {self.remote}')

    def is_on(self) -> bool:
        if self.value_string is None:
            raise Exception("No measurement available")
        return self.on

    def is_ready(self) -> bool:
        if self.value_string is None:
            raise Exception("No measurement available")
        return self.ready

    def read(self, adapter=None, verbose: bool = False):
        if not self._device_set:
            adapter = adapter or default_adapter or ACL(fallback=True)
            StatusDeviceSet(self.name, members=[self], adapter=adapter).readonce(verbose)
        else:
            self._device_set.readonce()
        return self.value_string

    def set(self, value, adapter=None):
        if not self._device_set:
            adapter = adapter or default_adapter or ACL(fallback=True)
            self._device_set = StatusDeviceSet(self.name, members=[self], adapter=adapter)
        return self._device_set.set([value])

    def set_on(self):
        self.set('ON')

    def set_off(self):
        self.set('OFF')

    def reset(self):
        self.set('RESET')


class BPMDevice(Device):
    def __init__(self, name, drf2=None, history=10):
        """
        BPM device with a circular history buffer
        """
        super().__init__(name, drf2, history)

    def read(self, adapter=None, verbose: bool = False):
        if not self._device_set:
            adapter = adapter or default_adapter or ACL(fallback=True)
            BPMDeviceSet(self.name, members=[self], adapter=adapter).readonce()
        else:
            self._device_set.readonce()
        return self.value_string

    def set(self, value):
        ds = StatusDeviceSet(self.name, members=[self], adapter=ACL(fallback=True))
        return ds.set([value])

    def update(self, data, timestamp, source: str):
        try:
            self.value = np.array(data)
        except:
            raise
        super().update(data, timestamp, source)

    def __str__(self) -> str:
        return f'{self.__class__.__name__} {self.name}:{self.value_string[:4]}...'

    def __repr__(self) -> str:
        return f'{self.__class__.__name__} {self.name}:{self.value_string[:4]}...'


#####


class DeviceSet:
    """
    A container of devices and other device sets, which is subclassed by more specialized sets. Implemented as
    an unbalanced tree, with 'nodes' containing only other device sets and 'leafs' containing only devices.
    Supports iterable and dictionary access modes.
    """

    def __init__(self, name: str, members: list, adapter=None, method="oneshot"):
        adapter = adapter or default_adapter or ACL(fallback=True)
        node = all([isinstance(ds, DeviceSet) for ds in members])
        leaf = all([isinstance(ds, Device) for ds in members])
        assert node or leaf

        if node:
            self.leaf = False
            self.children = {ds.name: ds for ds in members}
            assert len(self.children) == len(members)
            self.devices = None
            for c in self.children:
                c.parent = self
        else:
            self.leaf = True
            self.devices = {d.name: d for d in members}
            if not len(self.devices) == len(members):
                un, uc = np.unique(list([d.name for d in members]), return_counts=True)
                raise Exception(
                    f'There are non-unique devices in the list: {un}, {un[uc > 1]}, {uc}, {len(self.devices)}, {len(members)}')
            self.children = None
            self.parent = None
        self.adapter = adapter
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
        if not self.adapter:
            raise AttributeError('Adapter is not set!')
        if self.leaf:
            return self.adapter.check_supported(self.devices, method)
        else:
            return all([c.check_acquisition_supported(method) for c in self.children])

    def set(self, values: list, verbose: bool = False) -> int:
        if not self.adapter.check_setting_supported():
            raise Exception(f'Setting is not support for adapter {self.adapter.__class__}')
        if not self.check_acquisition_available(method='oneshot'):
            return 0
        if not self.devices:
            raise AttributeError("Cannot start acquisition on an empty set")
        if len(values) != len(self.devices):
            raise AttributeError("Value list length does not match number of devices in set")
        if self.leaf:
            cnt = self.adapter.set(self, values, verbose)
        else:
            raise ValueError("Tree traversing writes are not supported")
        return cnt

    def readonce(self, settings: bool = False, verbose: bool = False) -> int:
        if not self.check_acquisition_supported(method='oneshot'):
            raise Exception(f'Acquisition not supported - have {self.adapter.supported}')
        if not self.check_acquisition_available(method='oneshot'):
            return 0
        if not self.devices:
            raise AttributeError("Cannot start acquisition on an empty set")
        if self.leaf:
            cnt = self.adapter.readonce(self, settings=settings, verbose=verbose)
        else:
            cnt = 0
            for c in self.children:
                cnt += c.update(c.adapter.oneshot())
        return cnt

    def start_streaming(self):
        raise Exception('Not yet supported')

    def stop_streaming(self):
        raise Exception('Not yet supported')

    def start_polling(self) -> bool:
        if not self.check_acquisition_supported():
            raise Exception('Acquisition is not supported for method: {}'.format('polling'))
        if not self.check_acquisition_available():
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
    def __init__(self, name: str, members: list, adapter: 'Adapter' = None):
        if all([isinstance(d, str) for d in members]):
            members = [DoubleDevice(d) for d in members]
        elif all([isinstance(ds, DoubleDevice) for ds in members]):
            pass
        else:
            raise AttributeError(f'{self.__class__.__name__} can only contain devices - use DeviceSet for coalescing')
        super().__init__(name, members, adapter)

    def add(self, device: DoubleDevice):
        if isinstance(device, DoubleDevice):
            super().add(device)
        else:
            raise Exception("Device is not a BPM!")


class BPMDeviceSet(DeviceSet):
    """
    Leaf container of BPM and other array devices. Exposes methods for fast batch readout, and enforces data uniformity.
    """

    def __init__(self, name: str, members: list, adapter=None, enforce_array_length: int = None):
        if all([isinstance(d, str) for d in members]):
            members = [BPMDevice(d) for d in members]
        elif all([isinstance(ds, BPMDevice) for ds in members]):
            pass
        else:
            raise AttributeError(f'{self.__class__.__name__} can only contain devices - use DeviceSet for coalescing')
        super().__init__(name, members, adapter)
        self.array_length = enforce_array_length

    def add(self, device: Device):
        if isinstance(device, BPMDevice):
            # if self.array_length is not None and self.array_length != device.array_length:
            #     print('Device array length does not match set - using set value {}'.format(self.array_length))
            super().add(device)
        else:
            raise Exception("Device is not a BPM!")


class StatusDeviceSet(DeviceSet):
    """"
    Status device set
    """

    def __init__(self, name: str, members: list, adapter=None):
        if all([isinstance(d, str) for d in members]):
            members = [StatusDevice(d) for d in members]
        elif all([isinstance(ds, StatusDevice) for ds in members]):
            pass
        else:
            print([isinstance(m, StatusDevice) for m in members])
            print(members, [type(m) for m in members])
            raise AttributeError(f'{self.__class__.__name__} can only contain devices - use DeviceSet for coalescing')
        super().__init__(name, members, adapter)

    def add(self, device: Device):
        if isinstance(device, StatusDevice):
            super().add(device)
        else:
            raise Exception("Incorrect device type!")


class Adapter:
    supported = []
    can_set = False
    running = False

    def __init__(self):
        pass

    def check_supported(self, devices: dict, method: str = None) -> bool:
        if not method or method not in self.supported:
            raise AttributeError(f'Acquisition method {method} not supported - have {self.supported}')
        else:
            self.method = method
            return True

    def check_available(self, devices: dict, method: str = None) -> bool:
        return len(devices) < 500

    def check_setting_supported(self) -> bool:
        return self.can_set


class FakeAdapter(Adapter):
    """
    Fake adapter for internal testing
    """

    def __init__(self):
        self.name = 'Fake'
        self.supported = ['oneshot', 'polling', 'streaming']
        self.rate_limit = [-1, -1, -1]
        self.can_set = True
        self.state = {'status': 'OFF', 'double': 0.0, 'ready': 'TRIPPED'}
        super().__init__()

    def readonce(self, ds: DeviceSet, settings: bool = False, verbose: bool = False) -> int:
        assert ds.leaf
        data = {}
        for k, d in ds.devices.items():
            assert k == d.name
            if isinstance(d, BPMDevice):
                data[d.name] = np.random.rand(100)
                print(f'Fake array {data[d.name][0:5]}... returned')
            elif isinstance(d, StatusDevice):
                data[
                    d.name] = f"[{self.state['status']},{self.state['ready']},LOCAL,NEGATIVE,DC] 2020-02-23T13:41:44.153-0600"
                print(f'Fake status {data[d.name]} returned')
            elif isinstance(d, DoubleDevice):
                if settings:
                    data[d.name] = self.state['double']
                    print(f'Fake double state setting of {data[d.name]} returned')
                else:
                    data[d.name] = self.state['double'] + np.random.rand(1)
                    print(f'Fake double state reading of {data[d.name]} returned')
            else:
                raise Exception
        t1 = datetime.datetime.utcnow().timestamp()
        for k, v in ds.devices.items():
            print(f'Fake updating device {k} with {data[k]}')
            v.update(data[k], t1, self.name)
        print(f'Fake updated {len(data)} devices - returning')
        return len(data)

    def set(self, ds: DeviceSet, values: list) -> int:
        assert ds.leaf
        assert len(values) == len(ds.devices) and np.ndim(values) == 1
        data = {}
        for d, v in zip(ds.devices.values(), values):
            if isinstance(d, BPMDevice):
                raise Exception
            elif isinstance(d, StatusDevice):
                if v in ['ON', 'OFF']:
                    self.state['status'] = v
                elif v in ['READY', 'TRIPPED']:
                    self.state['ready'] = v
                elif v == 'RESET':
                    self.state['ready'] = 'READY'
                else:
                    raise
                print(f'Fake status state updated to {self.state["status"]} {self.state["ready"]}')
                data[d.name] = f"ok"
            elif isinstance(d, DoubleDevice):
                print(f'Fake double state updated to {v}')
                self.state['double'] = v
                data[d.name] = f"ok2"
            else:
                raise Exception
        return len(data)


class ACL(Adapter):
    """
    ACL web adapter. Works by fetching from the ACL web proxy via HTTP. Supports single-shot with many parallel
    requests using AsyncIO framework, with fallback to single thread via urllib,
    and quasi-polling mode that loops over single-shot commands.
    """

    def __init__(self, fallback: bool = False):
        self.name = 'ACL'
        self.supported = ['oneshot']
        self.rate_limit = [-1, -1]
        self.can_set = False
        self.fallback = fallback
        try:
            import httpx
        except ImportError:
            raise Exception('HTTP functionality requires httpx library')
        tm = httpx.Timeout(timeout=20, connect_timeout=5)
        pool = httpx.PoolLimits(soft_limit=25, hard_limit=25)
        # self.aclient = httpx.AsyncClient()
        # self.client = httpx.Client()
        if fallback:
            self.client = httpx.Client(timeout=tm, pool_limits=pool)
        else:
            # define async client in separate variable to not forget
            import nest_asyncio
            nest_asyncio.apply()
            self.aclient = httpx.AsyncClient(timeout=tm)
        super().__init__()

    def __del__(self):
        try:
            if self.fallback:
                self.client.close()
            else:
                async def terminate():
                    if self.aclient:
                        await self.aclient.aclose()

                asyncio.run(terminate())
        except Exception as e:
            print(f'Issue with disposing adapter {self.name}')
            print(e)
            # raise

    def _raw_command(self, cmd):
        """
        Executes specified ACL directly, with no modification, on the blocking client. Can be useful to wait for
        events and similar weird calls.
        :param cmd:
        :return:
        """
        acl = f'http://www-ad.fnal.gov/cgi-bin/acl.pl?acl={cmd}'
        resp = self.client.get(acl)
        return resp.text

    def readonce(self, ds: DeviceSet, settings: bool = False, verbose: bool = False) -> int:
        assert ds.leaf
        device_dict = ds.devices
        if settings:
            dev_names = [d.name + '.SETTING' for d in device_dict.values()]
        else:
            dev_names = [d.name for d in device_dict.values()]
        if self.fallback:
            c = self.client
            url = self._generate_url(ds, ','.join(dev_names))
            print('Url: ', url)
            try:
                resp = c.get(url)
                text = resp.text
                t1 = datetime.datetime.utcnow().timestamp()
                data = self._process_string(ds, text)
                for k, v in device_dict.items():
                    v.update(data[k], t1, self.name)
                return len(data)
            except Exception as e:
                print(e, sys.exc_info())
                raise
        else:
            c = self.aclient
            # Split tasks into sets
            num_lists = min(len(ds.devices), 10)
            dev_lists = [list(l) for l in np.array_split(dev_names, num_lists)]
            # print(dev_lists)
            urls = [self._generate_url(ds, ','.join(dl)) for dl in dev_lists]
            # print('Urls: ', urls)
            try:
                data = {}

                async def get(urls_inner):
                    tasks = [c.get(u) for u in urls_inner]
                    rs = await asyncio.gather(*tasks)
                    return rs

                responses = asyncio.run(get(urls))
                t1 = datetime.datetime.utcnow().timestamp()
                for r in responses:
                    data.update(self._process_string(ds, r.text))
                if len(data) != len(device_dict):
                    print('Issue with acquisition - devices missing:')
                    print([dn for dn in device_dict if dn not in data.keys()])
                    raise Exception
                for k, v in device_dict.items():
                    v.update(data[k], t1, self.name)
                return len(data)
            except Exception as e:
                print(f'Acquisition failed, urls: {urls}')
                print(e, sys.exc_info())
                raise

    def _generate_url(self, ds: DeviceSet, devstring: str):
        acl_root = 'http://www-ad.fnal.gov/cgi-bin/acl.pl?acl='
        if isinstance(ds, BPMDeviceSet):
            if ds.array_length:
                url_string = 'read/row/pendWait=0.5+devices="{}"+/num_elements={}'
                url = acl_root + url_string.format(devstring, ds.array_length)
            else:
                url_string = 'read/row/pendWait=0.5+devices="{}"+/all_elements'
                url = acl_root + url_string.format(devstring)
        elif isinstance(ds, StatusDeviceSet):
            url_string = 'read/row/pendWait=0.5+devices="{}"'
            url = acl_root + url_string.format(devstring + '.STATUS')
        elif isinstance(ds, DeviceSet):
            url_string = 'read/row/pendWait=0.5+devices="{}"'
            url = acl_root + url_string.format(devstring)
        else:
            raise
        return url

    def _process_string(self, ds: DeviceSet, text: str):
        text = text.strip()
        if isinstance(ds, BPMDeviceSet):
            if not (text.endswith('mm') or text.endswith('raw')):
                # print(f'URL: {url}')
                print(f'Text: {text}')
                raise Exception("Bad ACL output")
        if "DPM_PEND" in text:
            print('Bad ACL output - DPM_PEND')
        split_text = text.split('\n')
        data = {}
        for line in filter(None, split_text):
            spl = line.strip().split()
            # print(spl)
            devname = spl[0].split('@')[0].split('[')[0].replace('_', ':')
            if '[' in spl[0]:
                # Array mode
                try:
                    _ = float(spl[-2])
                    data[devname] = np.array([float(v) for v in spl[2:-1]])
                except ValueError:
                    # For weirdos that have stuff like 'db (low)'
                    data[devname] = np.array([float(v) for v in spl[2:-2]])
            else:
                if len(spl) < 3:
                    print(spl)
                    raise Exception
                data[devname] = float(spl[2])
        return data


class ACNETRelay(Adapter):
    """
    Adapter based on Java commmand proxy. One of quickest, but can be error-prone. Requires async libraries.
    """

    def __init__(self, address: str = "http://127.0.0.1:8080/", comm_method=1, set_multi=False, verbose: bool = False):
        self.name = 'ACNETRelay'
        self.supported = ['oneshot', 'polling']
        self.rate_limit = [-1, -1]
        self.can_set = True
        self.address = address
        # method 0 = cached values, method 1 = always read fresh
        if comm_method not in [0, 1]:
            raise Exception(f'unsupported method: {comm_method}')
        self.comm_method = 1
        self.method = 'oneshot'
        self.set_method_multi = set_multi
        self.verbose = verbose
        super().__init__()

        try:
            import httpx
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            raise Exception('Relay functionality requires certain libraries')
        tm = httpx.Timeout(timeout=20, connect_timeout=2)
        pool = httpx.PoolLimits(soft_limit=5, hard_limit=5)
        # self.aclient = httpx.AsyncClient()
        self.aclient = httpx.AsyncClient(timeout=tm)
        # self.client = httpx.Client()
        self.client = httpx.Client(timeout=tm, pool_limits=pool)

    def check_available(self, devices: dict, method=None):
        return len(devices) < 500

    def readonce(self, ds: DeviceSet, settings: bool = False, verbose: bool = False, retries: int = 1) -> int:
        if verbose or self.verbose:
            verbose = True

        assert ds.leaf
        c = self.aclient
        postfix = '.SETTING' if settings else '.READING'
        if isinstance(ds, BPMDeviceSet):
            # BPMS are slow to read out
            num_lists = len(ds.devices)
        else:
            num_lists = min(len(ds.devices), 5)
        device_lists = [list(ll) for ll in np.array_split(list(ds.devices.values()), num_lists)]
        params = []
        for dl in device_lists:
            request_items = []
            for device in dl:
                if isinstance(device, BPMDevice):
                    if ds.array_length:
                        request_items.append(f'{device.name}{postfix}' + f'[:{ds.array_length}]@I')
                    else:
                        request_items.append(f'{device.name}{postfix}' + '[]@I')
                elif isinstance(device, StatusDevice):
                    request_items.append(f'{device.name}.STATUS')
                elif isinstance(device, DoubleDevice):
                    request_items.append(f'{device.name + postfix}@I')
                else:
                    raise Exception(f'Unrecognized device type {device.__class__.__name__}')
            request_string = ';'.join(request_items)

            if self.comm_method == 0:
                params.append({'requestType': 'V1_DRF2_READ_MULTI_CACHED',
                               'request': request_string})
            elif self.comm_method == 1:
                params.append({'requestType': 'V1_DRF2_READ_SINGLE',
                               'request': request_string})
            else:
                print(self.comm_method)
                raise Exception
        if verbose: print(f'{self.name} : params {params}')

        try_cnt = 0
        while try_cnt < retries + 1:
            try:
                # async def get(json_lists):
                #     results = [await c.post(self.address, json=p) for p in json_lists]
                #     print(results)
                #     return results
                # results = asyncio.run(get(params))
                # def get(json_lists):
                #     results = [self.client.post(self.address, json=p) for p in json_lists]
                #     #print(results)
                #     return results
                async def get(json_lists):
                    tasks = [c.post(self.address, json=p) for p in json_lists]
                    results = await asyncio.gather(*tasks)
                    return results

                responses = asyncio.run(get(params))
                t1 = datetime.datetime.utcnow().timestamp()
                data = {}
                for r in responses:
                    # if not isinstance(ds, BPMDeviceSet):
                    if verbose: print(f'{self.name} : result {r._content}')
                    if r.status_code == 200:
                        data.update(self._process(ds, r.json()))
                    else:
                        return -1
                assert len(data) == len(ds.devices)
                for k, v in ds.devices.items():
                    v.update(data[k], t1, self.name)
                return len(data)
            except Exception as e:
                try_cnt += 1
                if try_cnt >= retries + 1:
                    print(f'{self.name} : FINAL EXCEPTION IN READONCE (try {try_cnt - 1}) - {e}')
                    # print(e, sys.exc_info())
                    raise
                else:
                    print(f'{self.name} : EXCEPTION IN READONCE (try {try_cnt - 1}) - {e}')

    def _process(self, ds: DeviceSet, r):
        responses = r['responseJson']
        # print(responses)
        data = {}
        if isinstance(ds, BPMDeviceSet):
            # Working in array base 64 mode
            for (k, v) in responses.items():
                # print(v)
                v_decoded = base64.b64decode(v)
                # print(v_decoded)
                data[k.split('.')[0]] = np.frombuffer(v_decoded, dtype='>f8')[:ds.array_length]
        elif isinstance(ds, StatusDeviceSet):
            # Status fields
            for (k, v) in responses.items():
                data[k.split('.')[0]] = v
        else:
            # Double values
            for (k, v) in responses.items():
                data[k.split('.')[0]] = float(v)
        return data

    def _process_settings(self, ds: DeviceSet, r):
        responses = r['responseJson']
        # print(responses)
        data = {}
        if isinstance(ds, BPMDeviceSet):
            # Working in array base 64 mode
            for (k, v) in responses.items():
                # print(v)
                v_decoded = base64.b64decode(v)
                # print(v_decoded)
                data[k.split('.')[0]] = np.frombuffer(v_decoded, dtype='>f8')[:ds.array_length]
        elif isinstance(ds, StatusDeviceSet):
            # Status fields
            for (k, v) in responses.items():
                data[k.split('.')[0]] = v
        else:
            # Double values
            for (k, v) in responses.items():
                data[k.split('.')[0]] = v
        return data

    def set(self, ds: DeviceSet, values: list, verbose: bool = False) -> int:
        if verbose or self.verbose:
            print(f'{self.name} : SETTING : {verbose} : {self.verbose}')
            verbose = True

        assert ds.leaf
        assert len(values) == len(ds.devices) and np.ndim(values) == 1
        c = self.aclient
        try:
            if self.set_method_multi:

                async def __submit(json_lists):
                    rs = [await c.post(self.address, json=p) for p in json_lists]
                    if verbose: print(f'{self.name} : result {rs}')
                    return rs

                params = []
                num_lists = min(len(ds.devices), 5) if len(ds.devices) > 20 else 1
                device_lists = [list(ll) for ll in np.array_split(list(ds.devices.values()), num_lists)]
                val_lists = [list(ll) for ll in np.array_split(values, num_lists)]
                for dl, vl in zip(device_lists, val_lists):
                    if isinstance(ds, StatusDeviceSet):
                        params.append({'requestType': 'V1_DRF2_SET_MULTI',
                                       'request': ";".join([device.name + '.CONTROL' for device in dl]),
                                       'requestValue': ";".join(vl)})
                    elif isinstance(ds, DoubleDeviceSet):
                        params.append({'requestType': 'V1_DRF2_SET_MULTI',
                                       'request': ";".join([device.name + '.SETTING' for device in dl]),
                                       'requestValue': ";".join([str(v) for v in vl])})
                    elif isinstance(ds, BPMDeviceSet):
                        raise Exception(f'Not allowed to set BPM sets {ds.name}')
                    else:
                        raise Exception
                if verbose: print(f'{self.name} : params {params}')
                responses = asyncio.run(__submit(params))
                t1 = datetime.datetime.utcnow().timestamp()
                ok_cnt = 0
                data = {}
                for r in responses:
                    if verbose: print(f'{self.name} : result {r._content}')
                    if r.status_code == 200:
                        data.update(self._process_settings(ds, r.json()))
                    else:
                        raise Exception
                for (k, v) in data.items():
                    if "0 0" in v or "72 1" in v:
                        ok_cnt += 1
                        continue
                    try:
                        if isinstance(self, DoubleDeviceSet):
                            float(v)
                            ok_cnt += 1
                        else:
                            if 'Error' not in v:
                                ok_cnt += 1
                            else:
                                raise Exception
                    except Exception as e:
                        print(f'Setting failure: {v}')
                        # raise Exception
                if ok_cnt != len(ds.devices):
                    print(f'Only {ok_cnt}/{len(ds.devices)} settings succeeded!')
                    raise Exception
                return ok_cnt
            else:
                async def __submit(json_lists):
                    rs = [await c.post(self.address, json=p) for p in json_lists]
                    if verbose: print(f'{self.name} : result {rs}')
                    return rs

                params = []
                for device, value in zip(ds.devices.values(), values):
                    if isinstance(device, BPMDevice):
                        raise Exception(f'Not allowed to set BPM device {device.name}')
                    elif isinstance(device, StatusDevice):
                        params.append({'requestType': 'V1_DRF2_SET_SINGLE',
                                       'request': device.name + '.CONTROL',
                                       'requestValue': value})
                    elif isinstance(device, DoubleDevice):
                        params.append({'requestType': 'V1_DRF2_SET_SINGLE',
                                       'request': device.name + '.SETTING',
                                       'requestValue': str(value)})
                    else:
                        raise Exception
                if verbose: print(f'{self.name} : params {params}')
                responses = asyncio.run(__submit(params))
                t1 = datetime.datetime.utcnow().timestamp()
                ok_cnt = 0
                for r in responses:
                    if verbose: print(f'{self.name} : result {r._content}')
                    # if r['status_code'] == 200:
                    if r.status_code == 200:
                        js = r.json()
                        if "0 0" not in js['responseJson'][js['request']]:
                            try:
                                if isinstance(self, DoubleDevice):
                                    float(js['responseJson'][js['request']])
                                    ok_cnt += 1
                                else:
                                    ok_cnt += 1
                                    pass
                            except Exception as e:
                                print(f'Setting failure: {js}')
                                # raise Exception
                                return -1
                        else:
                            ok_cnt += 1
                    else:
                        raise Exception
                        return -1
                assert ok_cnt == len(ds.devices)
                return ok_cnt
        except Exception as e:
            print(e, sys.exc_info())
            raise
# class DPM(Adapter):
#     def __init__(self):
#        import DPM, acnet
#         self.name = 'DPM'
#         self.supported = ['oneshot', 'polling']
#
#
#     def start_polling(self):
#         self.acnet = acnet.Connection()
#         self.DPM = dpm = DPM.Polling(self.acnet)
#
#         for i, name in enumerate(self.names):
#             dpm.add_entry(i, device)
#             self.device_tags[i] = self.name
#
#         dpm.start()
#     elif method == 'dpm_polling':
#         try:
#             self.DPM.stop()
#             self.DPM = None
#             self.acnet._close()


# def getHistory(self):
#     bpmlist = [dev.name for dev in self.devices]
#     if self.method == 0:
#         bpmstring = ''
#         try:
#             for b in bpmlist:
#                 bpmstring += '{{{}}},'.format(b)
#             txt = ("http://www-ad.fnal.gov/cgi-bin/acl.pl?acl="
#                    "read/row/pendWait=0.5+devices=\'" + bpmstring[:-1] + '\'+/num_elements=1000')
#             # print(txt)
#             resp = urlopen(txt)
#             r = resp.read().decode("ascii")
#             t1 = datetime.datetime.utcnow().timestamp()
#             if "DPM_PEND" in r:
#                 print('Bad ACL output - DPM_PEND')
#             if 'mm' not in r or 'raw' not in r:
#                 raise Exception("Bad ACL output")
#             split_text = r.strip().split('\n')
#             data = {}
#             for line in filter(None, split_text):
#                 spl = line.strip().split(' ')
#                 data[spl[0][:-3]] = np.array([float(v) for v in spl[2:-1]])
#             return (data, t1, 'unknown')
#         except Exception as e:
#             print(e, sys.exc_info())
#             return (None, datetime.datetime.utcnow().timestamp(), 'error')
#     elif self.method == 1:
#         return getBPMdataACLAsync()

# async def _fetch_task(session, url):
#     async with session.get(url) as response:
#         return await response.read().decode('ascii')

# async def _http_fetch_async(loop, urls):
#     connector = aiohttp.TCPConnector(limit=None, ssl=False)
#     tasks = []
#
#     # Fetch all responses within one Client session
#     # keep connection alive for all requests.
#     async with aiohttp.ClientSession(loop=loop, connector=connector) as session:
#         # async with aiohttp.ClientSession(loop=loop) as session:
#         for url in urls:
#             url_enc = yarl.URL(url, encoded=True)
#             task = asyncio.ensure_future(fetch_task(session, url_enc))
#             tasks.append(task)
#             # await asyncio.sleep(0.01)
#         responses = await asyncio.gather(*tasks, return_exceptions=True)
#         # you now have all response bodies in this variable
#         # print(responses)
#         return responses
#
# def _getBPMdataACLAsync(bpmlist):
#     # Split tasks into sets
#     num_lists = min(len(bpmlist), 5)
#     bpm_lists = [list(l) for l in np.array_split(bpmlist_initial, numlists)]
#     bpm_strings = []
#     for bpm_list in bpm_lists:
#         bpm_string = ''
#         for b in bpm_list:
#             bpm_string += '{{{}}},'.format(b)
#         bpm_string = ("http://www-ad.fnal.gov/cgi-bin/acl.pl?acl=read/row/pendWait=0.5+devices=\'"
#                       + bpm_string[:-1] + '\'+/num_elements=1000')
#         bpm_strings.append(bpmstring)
#     print(bpm_strings)
#     try:
#         data = {}
#         loop = asyncio.get_event_loop()
#         responses = loop.run_until_complete(_http_fetch_async(loop, bpm_strings))
#         t1 = datetime.datetime.utcnow().timestamp()
#         for r in responses:
#             if 'mm' not in r or 'raw' not in r:
#                 raise Exception('ACL async error - bad response')
#             split_text = r.strip().split('\n')
#             for line in filter(None, split_text):
#                 spl = line.strip().split(' ')
#                 data[spl[0][:-3]] = np.array([float(v) for v in spl[2:-1]])
#         return (data, t1, 'unknown')
#     except Exception as e:
#         print(e, sys.exc_info())
#         return (None, datetime.datetime.utcnow().timestamp(), 'error')
