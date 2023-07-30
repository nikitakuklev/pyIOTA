import asyncio
import base64
import datetime
import functools
import logging
import random
import sys
import threading
import time
import traceback
from abc import ABC, abstractmethod
from asyncio import AbstractEventLoop
from enum import Enum
from typing import Callable, Optional

import httpx
import numpy as np
import pandas as pd

from .acsys.dpm import ItemData, ItemStatus
from .data import ACNETErrorResponse, ArrayDataResponse, DoubleDataResponse, StatusDataResponse
from .devices import ArrayDevice, ArrayDeviceSet, Device, DeviceSet, DoubleDevice, DoubleDeviceSet, \
    StatusDevice, StatusDeviceSet
from .drf2 import DRF_PROPERTY, DRF_RANGE, parse_request
from .drf2.event import ImmediateEvent
from .errors import ACNETError, ACNETProxyError, ACNETTimeoutError

logger = logging.getLogger(__name__)


class READ_METHOD(Enum):
    CACHED = 0
    FRESH = 1


class READ_TYPE(Enum):
    ONCE = 0
    PERIODIC = 1


class WRITE_TYPE(Enum):
    ONCE = 0


def _convert_devices_to_immediate(device_list: list[Device], array_length=None):
    reqs = []
    for device in device_list:
        if isinstance(device, ArrayDevice):
            if array_length is not None:
                ds = device.drf2.to_canonical(range=DRF_RANGE('std', None,
                                                              array_length),
                                              event=ImmediateEvent())
            else:
                ds = device.drf2.to_canonical(event=ImmediateEvent())
            reqs.append(ds)
        elif isinstance(device, StatusDevice):
            reqs.append(device.drf2.to_canonical(property=DRF_PROPERTY.STATUS,
                                                 event=ImmediateEvent()))
        elif isinstance(device, DoubleDevice):
            reqs.append(device.drf2.to_canonical(event=ImmediateEvent()))
        else:
            raise Exception(f'Unrecognized device type {device.__class__.__name__}')
    return reqs


def _convert_devices_to_settings(device_list: list[Device]):
    reqs = []
    for device in device_list:
        if isinstance(device, ArrayDevice):
            ds = device.drf2.to_canonical(range=DRF_RANGE('full'), event=None)
            reqs.append(ds)
        elif isinstance(device, StatusDevice):
            reqs.append(device.drf2.to_canonical(property=DRF_PROPERTY.CONTROL, event=None))
        elif isinstance(device, DoubleDevice):
            reqs.append(device.drf2.to_canonical(property=DRF_PROPERTY.SETTING, event=None))
        else:
            raise Exception(f'Unrecognized device type {device.__class__.__name__}')
    return reqs


class ACNETSettings:
    n_connect = 2
    n_read = 2
    n_set = 2
    set_timeout = 5.0
    read_timeout = 5.5
    connect_timeout = 1.0
    devices_chunk_size = 25

    @staticmethod
    def acnet_retry(func, connect: int = 3, read: int = 3):
        initial_c = ACNETSettings.n_connect
        initial_r = ACNETSettings.n_read

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            ACNETSettings.n_connect = connect
            ACNETSettings.n_read = read
            value = func(*args, **kwargs)
            ACNETSettings.n_connect = initial_c
            ACNETSettings.n_read = initial_r
            return value

        return wrapper

    class Context:
        def __init__(self, n_read: int = 2, n_set: int = 2,
                     read_timeout: float = 5.5, set_timeout: float = 5.0):
            assert n_read >= 1
            assert n_set >= 1
            assert read_timeout >= 0.0
            # self.initial_c = ACNETSettings.n_connect
            self.initial_r = ACNETSettings.n_read
            self.initial_s = ACNETSettings.n_set
            self.initial_rt = ACNETSettings.read_timeout
            self.initial_st = ACNETSettings.set_timeout
            # self.connect = connect
            self.read = n_read
            self.set = n_set
            self.rt = read_timeout
            self.st = set_timeout

        def __enter__(self):
            # ACNETSettings.n_connect = self.connect
            ACNETSettings.n_read = self.read
            ACNETSettings.n_set = self.set
            ACNETSettings.read_timeout = self.rt
            ACNETSettings.set_timeout = self.st

        def __exit__(self, exc_type, exc_val, exc_tb):
            # ACNETSettings.n_connect = self.initial_c
            ACNETSettings.n_read = self.initial_r
            ACNETSettings.n_set = self.initial_s
            ACNETSettings.read_timeout = self.initial_rt
            ACNETSettings.set_timeout = self.initial_st
            return False


class AdapterManager:
    default_adapter = None

    @staticmethod
    def sort_drf_strings(devices: list[str]):
        """ Return Double/Status/Array device sets """
        double = []
        status = []
        array = []
        for d in devices:
            drf2 = parse_request(d)
            if drf2.property in [DRF_PROPERTY.READING, DRF_PROPERTY.SETTING]:
                if drf2.range is None:
                    double.append(d)
                else:
                    array.append(d)
            elif drf2.property in [DRF_PROPERTY.STATUS, DRF_PROPERTY.CONTROL]:
                if drf2.range is None:
                    status.append(d)
                else:
                    raise Exception(f'Status device with range is not supported')
            else:
                raise Exception(f'Device {d} not supported or invalid')
        return double, status, array

    @staticmethod
    def get_default():
        assert AdapterManager.default_adapter is not None, f'Default adapter not defined'
        return AdapterManager.default_adapter

    @staticmethod
    def read(devices: list[str]):
        """
        EasyAPI wrapper for all acnet read operations
        """

    @abstractmethod
    def subscribe(self, ds: DeviceSet, **kwargs):
        pass


# default_adapter = None


class Adapter(ABC):
    supported = []
    can_set = False

    def __init__(self):
        pass

    def check_supported(self, devices: dict, method: str = None) -> bool:
        if not method or method not in self.supported:
            raise AttributeError(
                    f'Acquisition method {method} not supported - have {self.supported}')
        else:
            self.method = method
            return True

    def check_available(self, devices: dict, method: str = None) -> bool:
        return len(devices) < 1000

    def check_setting_supported(self) -> bool:
        return self.can_set

    @abstractmethod
    def set(self, ds: DeviceSet, values: list, **kwargs) -> list[ACNETErrorResponse]:
        pass

    @abstractmethod
    def read(self, ds: DeviceSet, **kwargs) -> list:
        pass


class FakeAdapter(Adapter):
    """
    Fake adapter for internal testing
    """
    name = 'Fake'
    supported = ['oneshot']
    can_set = True

    def __init__(self, acnet_fails: list[str] = None,
                 failed_partial: dict[str, float] = None
                 ):
        self.acnet_fails = acnet_fails or []
        self.failed_partial = failed_partial or {}
        self.state = {}
        super().__init__()

    def status(self, fc, err, t_read):
        return ACNETErrorResponse(facility_code=fc,
                                  error_number=err,
                                  message='',
                                  timestamp=None,
                                  t_read=t_read)

    def _init_device(self, dn: str, array_len=10):
        data = {'value_r': 0.1,
                'value_w': 0.1,
                'array_write': np.random.rand(array_len),
                'array_read': np.random.rand(array_len),
                'status': {'on': True, 'ready': True, 'remote': False, 'positive': False,
                           'ramp': True
                           }
                }
        self.state[dn] = data

    def read(self, ds: DeviceSet, full: bool = False, **kwargs):
        assert ds.leaf
        data = {}
        devices = list(ds.devices.values())
        for k, d in ds.devices.items():
            assert k == d.name
            dn = d.drf2.device
            if dn not in self.state:
                self._init_device(dn)

        # reqs = _convert_devices_to_immediate(devices, al)
        for k, d in ds.devices.items():
            dn = d.drf2.device
            device_state = self.state[dn]
            ts = datetime.datetime.now().timestamp()
            t_read = 1e-3
            kwargs = dict(timestamp=ts, t_read=t_read)
            if dn in self.acnet_fails:
                data[k] = self.status(99, -99, t_read)
                continue
            if dn in self.failed_partial:
                is_fail = random.random() < self.failed_partial[dn]
                if is_fail:
                    raise ACNETTimeoutError(f'Nope')
                    # continue
            if isinstance(d, ArrayDevice):
                r = ArrayDataResponse(data=device_state['array_read'], **kwargs)
                data[k] = r
                # logger.debug(f'Fake array {data[k]}... returned')
            elif isinstance(d, StatusDevice):
                data[k] = StatusDataResponse(data=device_state['status'], **kwargs)
                logger.debug(f'Fake status {data[k]} returned')
            elif isinstance(d, DoubleDevice):
                if d.drf2.is_setting:
                    data[k] = DoubleDataResponse(data=device_state['value_w'], **kwargs)
                    logger.debug(f'Fake double state setting of {data[k]} returned')
                elif d.drf2.is_reading:
                    data[k] = DoubleDataResponse(data=device_state['value_r'], **kwargs)
                    logger.debug(f'Fake double state reading of {data[k]} returned')
                else:
                    raise Exception(f'Invalid device state {d.drf2}')
            else:
                raise Exception
        for i, (k, v) in enumerate(ds.devices.items()):
            # logger.debug(f'Fake updating device {k} with {data[k]}')
            v.update(data[k], source=self.name)
        if full:
            return [d.last_response for d in devices]
        else:
            return [d.value for d in devices]

    def read_raw(self, dn):
        if dn not in self.state:
            self._init_device(dn)
        device_state = self.state[dn]
        if dn in self.failed_partial:
            is_fail = random.random() < self.failed_partial[dn]
            if is_fail:
                raise ACNETTimeoutError(f'Nope')
        if '[' in dn:
            data = device_state['array_read']
        elif '|' in dn or '&' in dn:
            data = device_state['status']
        else:
            if 'SETTING' in dn:
                data = device_state['value_w']
            else:
                data = device_state['value_r']
        return data

    def set(self, ds: DeviceSet, values: list, **kwargs) -> list[ACNETErrorResponse]:
        assert ds.leaf
        assert len(values) == len(
                ds.devices), f'Got {len(values)} values for {len(ds.devices)} devices'
        # assert np.ndim(values) == 1
        data = {}
        for d, v in zip(ds.devices.values(), values):
            dn = d.drf2.device
            if isinstance(d, ArrayDevice):
                raise ACNETError(f'Array setting not supported')
            elif isinstance(d, StatusDevice):
                s = self.state[dn]
                if v in ['ON', 'OFF']:
                    s['status']['on'] = True if v == 'ON' else False
                elif v in ['READY', 'TRIPPED']:
                    s['ready'] = v
                elif v == 'RESET':
                    s['ready'] = 'READY'
                else:
                    raise
                print(f'Fake status state updated to {s["status"]}')
                data[dn] = f"ok"
            elif isinstance(d, DoubleDevice):
                # if d.drf2.is_reading:
                #    raise ACNETError(f'{d} - cant set a readback')
                print(f'Fake double state updated to {v}')
                self.state[dn]['value_w'] = v
                self.state[dn]['value_r'] = v + random.random() / 1.0e4
                data[dn] = f"ok2"
            else:
                raise Exception
        return [ACNETErrorResponse(facility_code=0,
                                   error_number=0,
                                   message='',
                                   timestamp=None) for d in ds.devices]


class ACL(Adapter):
    """
    ACL web adapter. Works by fetching from the ACL web proxy via HTTP. Supports single-shot with many parallel
    requests using AsyncIO framework, with fallback to single thread via urllib,
    and quasi-polling mode that loops over single-shot commands.
    """
    name = 'ACL'
    can_set = False
    supported = 'oneshot'

    def __init__(self, fallback: bool = False, **kwargs):
        self.fallback = fallback
        try:
            import httpx
        except ImportError:
            raise Exception('HTTP functionality requires httpx library')
        tm = httpx.Timeout(timeout=20, connect=1)
        # pool = httpx.PoolLimits(soft_limit=25, hard_limit=25)
        # self.aclient = httpx.AsyncClient()
        # self.client = httpx.Client()
        # if fallback:
        self.client = httpx.Client(timeout=tm)
        if not fallback:
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

    @staticmethod
    def _format_raw_command(cmd):
        return f'https://www-ad.fnal.gov/cgi-bin/acl.pl?acl={cmd}'

    def _raw_command(self, cmd):
        """
        Executes specified ACL directly, with no modification, on the blocking client. Can be useful to wait for
        events and similar weird calls.
        :param cmd:
        :return:
        """
        acl = f'https://www-ad.fnal.gov/cgi-bin/acl.pl?acl={cmd}'
        resp = self.client.get(acl)
        return resp.text

    def read_history(self, ds: DeviceSet, start: datetime, end: datetime, node='Backup'):
        assert ds.leaf
        device_dict = ds.devices
        assert len(device_dict) == 1
        dev_names = [d.name for d in device_dict.values()]
        s_start = start.strftime('%d-%b-%Y %H:%M:%S')
        s_end = end.strftime('%d-%b-%Y %H:%M:%S')
        cmd = f'logger_get/start="{s_start}"/end="{s_end}"/node={node}/double/units {dev_names[0]}'
        # /verbose
        acl = f'https://www-ad.fnal.gov/cgi-bin/acl.pl?acl={cmd}'
        resp = self.client.get(acl)
        text = resp.text
        t1 = datetime.datetime.utcnow().timestamp()

        text = text.strip()
        split_text = text.split('\n')
        rows = []
        for line in filter(None, split_text):
            spl = line.strip().split()
            assert len(spl) == 3
            stamp = datetime.datetime.strptime(spl[0], "%d-%b-%Y %H:%M:%S.%f")
            val = float(spl[1])
            units = spl[2]
            row = {'time': stamp, 'value': val, 'unit': units}
            rows.append(row)
        df = pd.DataFrame(data=rows)
        for k, v in device_dict.items():
            v.update_log(df)
        return len(df)

    def read(self, ds: DeviceSet, full: bool = False,
             verbose: bool = False, split: bool = False
             ) -> list:
        assert ds.leaf
        device_dict = ds.devices
        dev_names = [d.name for d in device_dict.values()]
        if self.fallback:
            c = self.client
            url = self._generate_url(ds, ','.join(dev_names))
            if verbose:
                logger.info(f'Fetching URL {url}')
            # print('Url: ', url)
            try:
                resp = c.get(url)
                text = resp.text
                t1 = datetime.datetime.utcnow().timestamp()
                result = self._process_string(ds, text)
                values = []
                for k, v in device_dict.items():
                    v.update(result, self.name)
                    values.append(v.value)
                return values
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
                async def get(urls_inner):
                    tasks = [c.get(u) for u in urls_inner]
                    rs = await asyncio.gather(*tasks)
                    return rs

                responses = asyncio.run(get(urls))
                data = {}
                for r in responses:
                    data.update(self._process_string(ds, r.text))
                if len(data) != len(device_dict):
                    print('Issue with acquisition - devices missing:')
                    print([dn for dn in device_dict if dn not in data.keys()])
                    raise Exception
                print(data)
                values = []
                for k, v in device_dict.items():
                    v.update(data[k], self.name)
                    values.append(v.value)
                return values
            except Exception as e:
                print(f'Acquisition failed, urls: {urls}')
                print(e, sys.exc_info())
                raise

    def set(self, ds: DeviceSet, values: list, **kwargs) -> list[ACNETErrorResponse]:
        raise NotImplementedError

    def _generate_url(self, ds: DeviceSet, devstring: str):
        acl_root = 'https://www-ad.fnal.gov/cgi-bin/acl.pl?acl='
        if isinstance(ds, ArrayDeviceSet):
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
        if isinstance(ds, ArrayDeviceSet):
            if not (text.endswith('mm') or text.endswith('raw')):
                # print(f'URL: {url}')
                print(f'Text: {text}')
                raise Exception("Bad ACL output")
        if "DPM_PEND" in text:
            print('Bad ACL output - DPM_PEND')
        split_text = text.split('\n')
        data = {}
        ts = time.time()
        for line in filter(None, split_text):
            spl = line.strip().split()
            # print(spl)
            devname = spl[0].split('@')[0].split('[')[0].replace('_', ':')
            if '[' in spl[0]:
                # Array mode
                try:
                    _ = float(spl[-2])
                    arr = np.array([float(v) for v in spl[2:-1]])
                    if len(arr) == 1:
                        assert isinstance(ds, DoubleDeviceSet)
                        resp = DoubleDataResponse(data=arr[0], timestamp=ts)
                    else:
                        assert isinstance(ds, ArrayDeviceSet)
                        resp = ArrayDataResponse(data=arr, timestamp=ts)
                    data[devname] = resp
                except ValueError:
                    # For weirdos that have stuff like 'db (low)'
                    arr = np.array([float(v) for v in spl[2:-2]])
                    if len(arr) == 1:
                        assert isinstance(ds, DoubleDeviceSet)
                        resp = DoubleDataResponse(data=arr[0], timestamp=ts)
                    else:
                        assert isinstance(ds, ArrayDeviceSet)
                        resp = ArrayDataResponse(data=arr, timestamp=ts)
                    data[devname] = resp
            else:
                if len(spl) < 3:
                    raise Exception(f'Bad ACL data: {spl}')
                if isinstance(ds, DoubleDeviceSet):
                    resp = DoubleDataResponse(data=float(spl[2]), timestamp=ts)
                    data[devname] = resp
                elif isinstance(ds, StatusDeviceSet):
                    v = spl[2]
                    datal = {}
                    assert ('ON' in v) != ('OFF' in v)
                    datal['on'] = 'ON' in v
                    assert ('READY' in v) != ('TRIPPED' in v)
                    datal['ready'] = 'READY' in v
                    assert ('LOCAL' in v) != ('REMOTE' in v)
                    datal['remote'] = 'REMOTE' in v
                    assert ('NEGATIVE' in v) != ('POSITIVE' in v)
                    datal['positive'] = 'POSITIVE' in v
                    assert ('DC' in v) != ('RAMP' in v)
                    datal['ramp'] = 'RAMP' in v
                    data[devname] = StatusDataResponse(data=datal, timestamp=ts)
        return data


class ACNETRelay(Adapter):
    """
    Adapter based on Java commmand proxy. One of quickest, but can be error-prone.
    Requires async libraries.
    """
    name = 'ACNETRelay'
    supported = ['oneshot', 'periodic']
    can_set = True

    def __init__(self,
                 address: str = "http://127.0.0.1:8080/",
                 read_method: READ_METHOD = READ_METHOD.FRESH,
                 set_multi: bool = True,
                 verbose: bool = False,
                 verify_connection: bool = False,
                 mock: FakeAdapter = None
                 ):
        super().__init__()
        self.address = address
        assert isinstance(read_method, READ_METHOD)
        self.read_method = read_method
        self.set_method_multi = set_multi
        self.verbose = verbose
        self.mock = mock

        try:
            import httpx
            import nest_asyncio
            nest_asyncio.apply()
        except ImportError:
            raise Exception('Relay functionality requires certain libraries')
        tm = httpx.Timeout(timeout=ACNETSettings.read_timeout,
                           connect=ACNETSettings.connect_timeout)
        limits = httpx.Limits(max_keepalive_connections=0,
                              max_connections=1000,
                              keepalive_expiry=None)
        self.aclient = httpx.AsyncClient(timeout=tm, limits=limits)
        self.client = httpx.Client(timeout=tm, limits=limits)

        if verify_connection:
            try:
                self.ping()
            except httpx.TimeoutException:
                logger.warning(f'Proxy ping failed - is it launched?')

    # def start_proxy(self, java_path=None, jar_path=None):
    #     import subprocess
    #
    #     java_path = java_path or r"I:\Data\Run_4\NIO\nkuklev\Binaries\amazon-corretto-8.352.08.1" \
    #                              r"-windows-x64-jdk\jdk1.8.0_352\bin\java.exe"
    #     jar_path = jar_path or r"I:\Data\Run_4\NIO\nkuklev\Software\ACNET-Proxy\target\ACNET" \
    #                            r"-Proxy-1.1-SNAPSHOT-jar-with-dependencies.jar"
    #     logger.info(f"Executing {[java_path, '-jar', jar_path]}")
    #     p = subprocess.Popen([java_path, '-jar', jar_path])
    #     return p

    def ping(self):
        import httpx
        t1 = time.perf_counter()
        try:
            response = self.client.get(self.address + 'status', timeout=1.0)
        except httpx.TimeoutException:
            raise ACNETTimeoutError(f"Request {self.address + 'status'} timeout out") from None
        assert response.status_code == 200
        assert response.text == 'OK'
        return {'ping_time': time.perf_counter() - t1, 'address': self.address}

    def check_available(self, devices: dict, method=None):
        import httpx
        try:
            self.ping()
            return True
        except httpx.TimeoutException:
            return False

    def _process(self, ds: DeviceSet, r: dict, t_read: float):
        responses = r['responseJson']
        timestamps = r['responseTimestampJson']
        data = {}

        for (k, v) in responses.items():
            if v['error'] != '':
                msg = v['error_message']
                if msg == 'null':
                    msg = None
                resp = ACNETErrorResponse(facility_code=int(v['facility_code']),
                                          error_number=int(v['error_number']),
                                          message=msg,
                                          timestamp=timestamps[k],
                                          t_read=t_read)
            elif isinstance(ds, DoubleDeviceSet):
                # Double values
                resp = DoubleDataResponse(data=float(v['data']), timestamp=timestamps[k])
            elif isinstance(ds, ArrayDeviceSet):
                # Working in array base 64 mode
                v_decoded = base64.b64decode(v['data'])
                arr = np.frombuffer(v_decoded, dtype='>f8')  # [:ds.array_length]
                resp = ArrayDataResponse(data=arr, timestamp=timestamps[k])
            elif isinstance(ds, StatusDeviceSet):
                # Status fields
                vmap = v.copy()
                for status_key in ['ready', 'ramp', 'positive', 'remote', 'on']:
                    if v[status_key].lower() == 'true':
                        vmap[status_key] = True
                    elif v[status_key].lower() == 'false':
                        vmap[status_key] = False
                    else:
                        raise Exception(f'Bad status key in {vmap}')
                resp = StatusDataResponse(data=vmap, timestamp=timestamps[k])
            else:
                raise Exception
            data[k] = resp
        return data

    def read(self,
             ds: DeviceSet,
             full: bool = False,
             verbose: bool = False,
             split: bool = True
             ) -> list:
        if verbose or self.verbose:
            verbose = True
        retries = ACNETSettings.n_read
        cs = ACNETSettings.devices_chunk_size

        assert ds.leaf
        c = self.aclient
        devices = list(ds.devices.values())
        if isinstance(ds, ArrayDeviceSet) or not split:
            num_lists = 1
        else:
            num_lists = min(len(ds.devices) // cs + 1, 5) if len(ds.devices) > cs else 1
        device_lists = [list(ll) for ll in np.array_split(devices, num_lists)]

        params = []
        name_to_device_map = {}
        al = ds.array_length if isinstance(ds, ArrayDeviceSet) else None
        for i, dl in enumerate(device_lists):
            reqs = _convert_devices_to_immediate(dl, al)
            for j, req in enumerate(reqs):
                name_to_device_map[req] = dl[j].name
            request_string = ';'.join(reqs)
            #if verbose:
            #    logger.debug(f'List {i}: {request_string}')
            if self.read_method == READ_METHOD.CACHED:
                params.append({'requestType': 'READ_CACHED', 'request': request_string})
            elif self.read_method == READ_METHOD.FRESH:
                params.append({'requestType': 'READ', 'request': request_string})
            else:
                raise ACNETError(f'{self.name}: method {self.read_method} is not recognized!')
        if verbose:
            logger.debug(f'{self.name}: params {params}')

        try_cnt = 0
        while True:
            try:
                async def get(json_lists):
                    try:
                        tasks = [c.post(self.address, json=p,
                                        timeout=ACNETSettings.read_timeout) for p in json_lists]
                        results = await asyncio.gather(*tasks)
                        return results
                    except httpx.ReadTimeout as ex:
                        raise ACNETTimeoutError(f'Proxy read timed out') from None
                    except Exception as ex:
                        raise ACNETError(f'Unexpected failure in proxy adapter')

                t1 = time.perf_counter()
                responses = asyncio.run(get(params))
                t_read = time.perf_counter() - t1

                data = {}
                for r in responses:
                    if r.status_code == 200:
                        processed = self._process(ds, r.json(), t_read)
                        if verbose:
                            for ip, p in enumerate(processed):
                                logger.debug(f'{self.name}: {ip} {p}')
                        data.update(processed)
                    else:
                        if verbose:
                            logger.debug(f'{self.name}: result {r.content.decode()[:1000]}...')
                        raise ACNETProxyError(f'Bad status code {r.status_code}: {r.text=}')

                assert len(data) == len(
                        ds.devices), f'Length mismatch {len(data)} {len(ds.devices)}'

                for i, (k, v) in enumerate(data.items()):
                    dev_name = name_to_device_map[k]
                    dev = ds.devices[dev_name]
                    dev.update(v, source=self.name)

                values = []
                if full:
                    for k, v in ds.devices.items():
                        values.append(v.last_response)
                    return values
                else:
                    for k, v in ds.devices.items():
                        values.append(v.value)
                    return values
            except Exception as ex:
                try_cnt += 1
                logger.warning(f'{self.name}: read failed ({try_cnt} of {retries}) | {ex=}')
                if try_cnt >= retries:
                    logger.warning(f'{self.name}: out of retries - aborting read')
                    raise ex

    def _process_settings(self, ds: DeviceSet, r):
        responses = r['responseJson']
        timestamps = r['responseTimestampJson']
        data = {}
        if isinstance(ds, ArrayDeviceSet):
            # Working in array base 64 mode
            for (k, v) in responses.items():
                if v['error'] != '':
                    resp = ACNETErrorResponse(facility_code=int(v['facility_code']),
                                              error_number=int(v['error_number']),
                                              message=v['error_message'],
                                              timestamp=timestamps[k])
                else:
                    v_decoded = base64.b64decode(v['data'])
                    arr = np.frombuffer(v_decoded, dtype='>f8')  # [:ds.array_length]
                    resp = ArrayDataResponse(data=arr, timestamp=timestamps[k])
                data[k] = resp
        elif isinstance(ds, StatusDeviceSet):
            # Status fields
            for (k, v) in responses.items():
                if v['error'] != '':
                    resp = ACNETErrorResponse(facility_code=int(v['facility_code']),
                                              error_number=int(v['error_number']),
                                              message=v['error_message'],
                                              timestamp=timestamps[k])
                else:
                    vmap = v.copy()
                    for status_key in ['ready', 'ramp', 'positive', 'remote', 'on']:
                        vmap[status_key] = bool(v[status_key])
                    resp = StatusDataResponse(data=vmap, timestamp=timestamps[k])
                data[k] = resp
        else:
            # Double values
            for (k, v) in responses.items():
                if v['error'] != '':
                    print(v)
                    resp = ACNETErrorResponse(facility_code=int(v['facility_code']),
                                              error_number=int(v['error_number']),
                                              message=v['error_message'],
                                              timestamp=timestamps[k])
                else:
                    resp = DoubleDataResponse(data=float(v['data']), timestamp=timestamps[k])
                data[k] = resp
        return data

    def set(self,
            ds: DeviceSet,
            values: list,
            full: bool = False,
            verbose: bool = False,
            split: bool = True
            ) -> list:
        if verbose or self.verbose:
            logger.debug(f'{self.name}: SETTING : {verbose} : {self.verbose}')
            verbose = True
        retries = ACNETSettings.n_set

        assert ds.leaf
        assert len(values) == len(ds.devices) and np.ndim(values) == 1
        c = self.aclient
        try:
            if self.set_method_multi:
                async def __submit(json_lists):
                    rs = []
                    for p in json_lists:
                        try_cnt = 0
                        while True:
                            try:
                                # if self.mock:
                                #     logger.debug(f'Mocking proxy set')
                                #     devices = p['request'].split(';')
                                #     values = p['requestValue'].split(';')
                                #     for i, d in enumerate(devices):
                                #         try:
                                #             v = self.mock.read_raw(d)
                                #             output_data[i] = ItemData(i, time.time(), v)
                                #             logger.debug(f'{d=} {i=} {output_data[i]=}')
                                #         except ACNETTimeoutError:
                                #             # output_data[i] = ItemStatus(i, 5080)
                                #             raise
                                resp = await c.post(self.address, json=p,
                                                    timeout=ACNETSettings.set_timeout)
                                rs.append(resp)
                                break
                            except httpx.TimeoutException as exi:
                                try_cnt += 1
                                logger.warning(
                                        f'{self.name}: set failed ({try_cnt} of {retries}) | {exi=}')
                                if try_cnt >= retries:
                                    logger.warning(f'{self.name}: out of retries - aborting read')
                                    raise exi
                            except Exception as exi:
                                try_cnt += 1
                                logger.error(f'{self.name}: Unexpected proxy read error {exi}')
                                raise exi
                    if verbose:
                        logger.debug(f'{self.name} : result {rs}')
                    return rs

                devices = list(ds.devices.values())
                params = []
                if split:
                    num_lists = min(len(ds.devices) // 20 + 1, 5) if len(ds.devices) > 20 else 1
                else:
                    num_lists = 1
                device_lists = [list(ll) for ll in np.array_split(devices, num_lists)]
                val_lists = [list(ll) for ll in np.array_split(values, num_lists)]

                name_to_device_map = {}
                for dl, vl in zip(device_lists, val_lists):
                    reqs = _convert_devices_to_settings(dl)
                    for j, req in enumerate(reqs):
                        name_to_device_map[req] = dl[j].name
                    request_string = ';'.join(reqs)
                    values_string = ";".join([str(val) for val in vl])
                    if isinstance(ds, StatusDeviceSet) or isinstance(ds, DoubleDeviceSet):
                        params.append({'requestType': 'SET_MULTI',
                                       'request': request_string,
                                       'requestValue': values_string
                                       })
                    elif isinstance(ds, ArrayDeviceSet):
                        raise NotImplemented(f'Not allowed to set arrays')
                    else:
                        raise Exception
                if verbose:
                    logger.debug(f'{self.name}: params {params}')

                t1 = time.perf_counter()
                try:
                    responses = asyncio.run(__submit(params))
                except httpx.ReadTimeout as ex:
                    logger.warning(f'Async setting communication timed out')
                    raise ACNETTimeoutError(f'Proxy SET timeout error') from None
                t_read = time.perf_counter() - t1

                data = {}
                for r in responses:
                    if verbose:
                        logger.debug(f'{self.name}: result {r._content}')
                    if r.status_code == 200:
                        data.update(self._process(ds, r.json(), t_read))
                    else:
                        raise ACNETProxyError(f'Bad status code {r.status_code=} {r=}')

                assert len(data) == len(
                        ds.devices), f'Length mismatch {len(data)} {len(ds.devices)}'

                values: list[Optional[ACNETErrorResponse]] = [None] * len(devices)
                for i, (k, v) in enumerate(data.items()):
                    dev_name = name_to_device_map[k]
                    dev = ds.devices[dev_name]
                    dev_idx = devices.index(dev)
                    values[dev_idx] = v
                    # dev.update(v, source=self.name)

                if full:
                    return values
                else:
                    return [v.is_success for v in values]

                # for (k, v) in data.items():
                #     if "0 0" in v or "72 1" in v:
                #         ok_cnt += 1
                #         continue
                #     try:
                #         if isinstance(self, DoubleDeviceSet):
                #             float(v)
                #             ok_cnt += 1
                #         else:
                #             if 'Error' not in v:
                #                 ok_cnt += 1
                #             else:
                #                 raise Exception
                #     except Exception as e:
                #         print(f'Setting failure: {v}')
                #         # raise Exception
                # if ok_cnt != len(ds.devices):
                #     print(f'Only {ok_cnt}/{len(ds.devices)} settings succeeded!')
                #     raise Exception
                # return ok_cnt
            else:
                async def __submit(json_lists):
                    rs = [await c.post(self.address, json=p) for p in json_lists]
                    if verbose:
                        print(f'{self.name} : result {rs}')
                    return rs

                params = []
                for device, value in zip(ds.devices.values(), values):
                    if isinstance(device, ArrayDevice):
                        raise Exception(f'Not allowed to set BPM device {device.name}')
                    elif isinstance(device, StatusDevice):
                        params.append({'requestType': 'SET_SINGLE',
                                       'request': device.name + '.CONTROL',
                                       'requestValue': value
                                       })
                    elif isinstance(device, DoubleDevice):
                        params.append({'requestType': 'SET_SINGLE',
                                       'request': device.name + '.SETTING',
                                       'requestValue': str(value)
                                       })
                    else:
                        raise Exception
                if verbose:
                    print(f'{self.name} : params {params}')
                responses = asyncio.run(__submit(params))
                t1 = datetime.datetime.utcnow().timestamp()
                ok_cnt = 0
                for r in responses:
                    if verbose:
                        print(f'{self.name} : result {r._content}')
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
                        raise Exception(f'{self.name} : bad return - {r.status_code} {r.text}')
                assert ok_cnt == len(ds.devices)
                final_data = []
                for k, v in ds.devices.items():
                    resp = ACNETErrorResponse(facility_code=v.status.facility,
                                              error_number=v.status.err_code,
                                              message='',
                                              timestamp=None)
                    final_data.append(resp)
                return final_data
        except Exception as e:
            logger.error(f'Proxy set failed unexpectedly with {e=}')
            logger.error(sys.exc_info())
            logger.error(traceback.format_exc())
            raise e


class AsyncioQueue:
    def __init__(self, loop, maxsize=0, ):
        self._queue = asyncio.Queue(maxsize)
        self._loop = loop

    async def async_get(self):
        return await self._queue.get()

    async def async_put(self, value):
        return await self._queue.put(value)

    def get(self):
        future = asyncio.run_coroutine_threadsafe(
                self._queue.get(), self._loop)

        return future.result()

    def put(self, value):
        asyncio.run_coroutine_threadsafe(
                self._queue.put(value), self._loop)


class Subscription:
    def __init__(self):
        self.subs = []
        self.running = True
        self.stop_requested = False
        self.thread = None

    def register_callback(self, cb):
        assert callable(cb)
        self.subs.append(cb)

    def process_callback(self, device, data):
        for cb in self.subs:
            try:
                cb(device, data)
            except Exception as ex:
                logger.warning(f'Callback {cb} failed with {ex=}')
                logger.warning(f'{traceback.format_exc()}')

    def request_stop(self):
        self.stop_requested = True


class DPM(Adapter):
    """
    DPM access using acsys library
    """
    name = 'DPM'
    supported = ['oneshot', 'monitoring']
    supported_reads = [READ_TYPE.ONCE]
    supported_writes = [WRITE_TYPE.ONCE]
    default_role = 'testing'

    def __init__(self,
                 default_role: str = None,
                 verbose: bool = False,
                 mock: FakeAdapter = None
                 ):
        super().__init__()
        self.can_set = True
        self.verbose = verbose
        self.mock = mock
        self.loop: AbstractEventLoop = None
        if default_role is not None:
            self.default_role = default_role

    def ping(self):
        from .acsys.dpm import DPMContext, find_dpm
        from .acsys import run_client
        task = None
        task_name = None
        tping = None

        async def app(con):
            nonlocal task, task_name, tping
            async with DPMContext(con) as dpm:
                task = dpm.dpm_task
                t1 = time.perf_counter()
                task_name = await find_dpm(dpm.con, node=dpm.desired_node)
                tping = time.perf_counter() - t1

        run_client(app)
        return {'ping_time': tping, 'acnet_id': task, 'dpm': task_name}

        # def _start_acsys_thread(self, requests):

    #     from .acsys.dpm import DPMContext
    #     from .acsys import run_client
    #     async def my_app(con):
    #         # Setup context
    #         async with DPMContext(con) as dpm:
    #             await dpm.add_entry(0, 'Z:CUBE_X.SETTING')
    #             await dpm.add_entry(1, 'Z:CUBE_Y.SETTING')
    #             await dpm.start()
    #             async for evt_res in dpm:
    #                 if evt_res.is_reading_for(0):
    #                     if evt_res.is_reading_for(0):
    #                         print(evt_res)
    #                     else:
    #                         pass
    #                 else:
    #                     pass
    #
    #     run_client(my_app)

    def _settings_thread(self, input_data: dict):
        from .acsys.dpm import DPMContext
        from .acsys import run_client
        output_data = {}
        tag_map = {}

        async def app(con):
            async with DPMContext(con) as dpm:
                for i, d in enumerate(input_data.keys()):
                    await dpm.add_entry(i, d)
                    tag_map[i] = d
                await dpm.enable_settings(role=self.default_role)
                await dpm.start()
                for i, (k, v) in enumerate(input_data.items()):
                    await dpm.apply_settings([(i, v)])

                async for ev in dpm.replies():
                    logger.info(f'DPM new data: {ev}')
                    if ev.isStatus:
                        if ev.status.isFatal:
                            logger.warning(f'cannot set {ev.tag}: %s', ev.status)
                        output_data[ev.tag] = ev
                    else:
                        raise Exception
                    if len(output_data) == len(input_data):
                        break

        run_client(app)
        final_data = {v: output_data[k] for k, v in tag_map.items()}
        return final_data

    def set(self,
            ds: DeviceSet,
            values: list,
            full: bool = False,
            verbose: bool = False,
            split: bool = True
            ):
        if verbose or self.verbose:
            logger.debug(f'{self.name} : SETTING : {verbose} : {self.verbose}')
            verbose = True
        assert ds.leaf
        assert len(values) == len(ds.devices) and np.ndim(values) == 1
        devices = list(ds.devices.values())
        reqs = []
        req_to_device_map = {}
        postfix = '.SETTING'
        for device in devices:
            if isinstance(device, ArrayDevice):
                raise
            elif isinstance(device, StatusDevice):
                req = f'{device.name}.STATUS@N'
            elif isinstance(device, DoubleDevice):
                req = f'{device.name + postfix}@N'

            else:
                raise Exception(f'Unrecognized device type {device.__class__.__name__}')
            reqs.append(req)
            req_to_device_map[req] = device
        data = self._settings_thread({k: v for k, v in zip(reqs, values)})
        final_data = []
        for k, v in data.items():
            dev = req_to_device_map[k]
            resp = ACNETErrorResponse(facility_code=v.status.facility,
                                      error_number=v.status.err_code,
                                      message='',
                                      timestamp=None)
            final_data.append(resp)
        return final_data

    def _format_ev(self, ev):
        from .acsys.dpm import ItemData
        if isinstance(ev, ItemData):
            if not isinstance(ev.data, list):
                return f'{ev.meta["name"]} at {ev.stamp} = {ev.data}'
            else:
                return f'{ev.meta["name"]} at {ev.stamp} = {ev.data[:3]}... (len={len(ev.data)})'
        else:
            return f'{ev.tag} = STATUS {ev.status}'

    def process_dpm_response(self, ds, k, v, t_read=None):
        from .acsys.dpm import ItemStatus
        def status(k, v):
            return ACNETErrorResponse(facility_code=v.status.facility,
                                      error_number=v.status.err_code,
                                      message='',
                                      timestamp=None,
                                      t_read=t_read)

        if isinstance(ds, ArrayDeviceSet):
            if isinstance(v, ItemStatus):
                return status(k, v)
            else:
                return ArrayDataResponse(data=np.array(v.data),
                                         timestamp=v.stamp,
                                         t_read=t_read)
        elif isinstance(ds, DoubleDeviceSet):
            if isinstance(v, ItemStatus):
                return status(k, v)
            else:
                return DoubleDataResponse(data=v.data,
                                          timestamp=v.stamp,
                                          t_read=t_read)
        elif isinstance(ds, StatusDeviceSet):
            if isinstance(v, ItemStatus):
                return status(k, v)
            else:
                return StatusDataResponse(data=v.data,
                                          timestamp=v.stamp,
                                          t_read=t_read)
        else:
            raise Exception

    def _monitor_thread(self, ds: DeviceSet, sub: Subscription, full: bool):
        from .acsys.dpm import DPMContext
        from .acsys import run_client

        tag_map = {}
        device_map = {}
        for device in ds.devices.values():
            cn = device.drf2.to_canonical()
            device_map[cn] = device

        def put_result(tag, data):
            response = self.process_dpm_response(ds, tag, data)
            if isinstance(response.data, float):
                response.data = [response.data]
            cn = tag_map[tag]
            device_map[cn].update(response, source=self.name)
            sub.process_callback(device_map[cn], response)

        def run_monitor_thread():
            async def reader(con):
                sub.running = True
                async with DPMContext(con) as dpm:
                    for i, d in enumerate(device_map.keys()):
                        await dpm.add_entry(i, d)
                        tag_map[i] = d
                    await dpm.start()

                    async for ev in dpm:
                        logger.debug(f'DPM mon {ev.tag}-{tag_map[ev.tag]}: {self._format_ev(ev)}')
                        if ev.isReading:
                            put_result(ev.tag, ev)
                        if sub.stop_requested:
                            break
                sub.running = False

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            run_client(reader)

        t = threading.Thread(target=run_monitor_thread, daemon=True)
        t.start()
        sub.thread = t
        # logger.debug(f'DPM read finished in {(time.perf_counter() - t1)}')
        # final_data = {v: output_data[k] for k, v in tag_map.items()}
        # return sub
        return t

    def subscribe(self, ds: DeviceSet, callback: Callable = None, full=True):
        sub = Subscription()
        if callback is not None:
            sub.register_callback(callback)
        logger.debug(f'DPM sub start at {datetime.datetime.now()}')
        self._monitor_thread(ds, sub, full)
        return sub

    def _readonce_thread(self, requests):
        from .acsys.dpm import DPMContext
        from .acsys import run_client
        output_data = {}
        tag_map = {}

        async def reader(con):
            async with DPMContext(con) as dpm:
                for ii, dd in enumerate(requests):
                    await dpm.add_entry(ii, dd)
                    tag_map[ii] = dd
                await dpm.start()
                while True:
                    try:
                        ev = await asyncio.shield(asyncio.wait_for(dpm.__anext__(),
                                                                   ACNETSettings.read_timeout))
                        logger.debug(f'DPM {ev.tag}-{tag_map[ev.tag]}: {self._format_ev(ev)}')
                        if ev.isReading:
                            output_data[ev.tag] = ev
                        else:
                            output_data[ev.tag] = ev
                        if len(output_data) == len(requests):
                            break
                    except StopAsyncIteration:
                        break
                    except TimeoutError:
                        remaining = [x for x in requests if x not in output_data]
                        n_t = len(requests)
                        n_r = len(remaining)
                        raise ACNETTimeoutError(f'DPM timeout waiting for ({remaining}) ({n_r} of'
                                                f' {n_t})')
                # await process_obj(obj1))
                # async for ev in dpm:
                #     logger.debug(f'DPM {ev.tag}-{tag_map[ev.tag]}: {self._format_ev(ev)}')
                #     if ev.isReading:
                #         output_data[ev.tag] = ev
                #     else:
                #         output_data[ev.tag] = ev
                #     if len(output_data) == len(requests):
                #         break

        if self.mock:
            logger.debug(f'Mocking DPM reads')
            for i, d in enumerate(requests):
                tag_map[i] = d
                try:
                    v = self.mock.read_raw(d)
                    output_data[i] = ItemData(i, time.time(), v)
                    logger.debug(f'{d=} {i=} {output_data[i]=}')
                except ACNETTimeoutError:
                    # output_data[i] = ItemStatus(i, 5080)
                    raise
        else:
            run_client(reader)
        # logger.debug(f'DPM read finished in {(time.perf_counter() - t1)}')
        final_data = {v: output_data[k] for k, v in tag_map.items()}
        return final_data

    def read(self, ds: DeviceSet, full=False, verbose=False, split: bool = False) -> list:
        from .acsys.dpm import ItemStatus
        if verbose or self.verbose:
            verbose = True
        retries = ACNETSettings.n_read
        devices = list(ds.devices.values())
        al = ds.array_length if isinstance(ds, ArrayDeviceSet) else None
        reqs = _convert_devices_to_immediate(devices, al)

        t1 = time.perf_counter()
        try_cnt = 0
        data = {}
        while True:
            try:
                data = self._readonce_thread(reqs)
                break
            except ACNETTimeoutError as ex:
                try_cnt += 1
                logger.warning(f'{self.name}: read failed ({try_cnt} of {retries}) | {ex=}')
                if try_cnt >= retries:
                    logger.warning(f'{self.name}: out of retries - aborting read')
                    raise ex
            except Exception as ex:
                try_cnt += 1
                logger.error(f'Unexpected DPM read error {ex}')
                raise ex

        t_read = time.perf_counter() - t1

        def status(k, v):
            return ACNETErrorResponse(facility_code=v.status.facility,
                                      error_number=v.status.err_code,
                                      message='',
                                      timestamp=None,
                                      t_read=t_read)

        data_resps = {}
        if isinstance(ds, ArrayDeviceSet):
            for k, v in data.items():
                if isinstance(v, ItemStatus):
                    data_resps[k] = status(k, v)
                else:
                    data_resps[k] = ArrayDataResponse(data=np.array(v.data),
                                                      timestamp=v.stamp,
                                                      t_read=t_read)
        elif isinstance(ds, DoubleDeviceSet):
            for k, v in data.items():
                if isinstance(v, ItemStatus):
                    data_resps[k] = status(k, v)
                else:
                    data_resps[k] = DoubleDataResponse(data=v.data,
                                                       timestamp=v.stamp,
                                                       t_read=t_read)
        elif isinstance(ds, StatusDeviceSet):
            for k, v in data.items():
                if isinstance(v, ItemStatus):
                    data_resps[k] = status(k, v)
                else:
                    data_resps[k] = StatusDataResponse(data=v.data,
                                                       timestamp=v.stamp,
                                                       t_read=t_read)
        else:
            raise Exception

        for i, (k, v) in enumerate(ds.devices.items()):
            req = reqs[i]
            v.update(data_resps[req], source=self.name)

        if verbose:
            for d in devices:
                v = d.last_response
                if isinstance(v, ACNETErrorResponse):
                    if not v.is_warning:
                        logger.debug(f'Read of {d.name} produced error {d.last_response}')
        if full:
            return [d.last_response for d in devices]
        else:
            return [d.value for d in devices]

    def get_auth(self):
        import gssapi
        gssapi.creds.Credentials.acquire()
        creds = gssapi.creds.Credentials(usage='both')
        return creds

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
