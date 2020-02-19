import asyncio
import collections
import sys
import datetime
import numpy as np

data_entry = collections.namedtuple('DataEntry',['id','value','ts','src'])

class Device:

    def __init__(self, name, drf2, history=10):
        self.name = name
        self.drf2 = drf2
        self.history_len = history
        self.history = collections.deque(maxlen=history)
        self.update_cnt = 0
        self.access_method = None
        self.value = None
        self.value_tuple = None
        self.last_update = None

    def get_history(self) -> list:
        return list(self.history)

    def update(self, data, timestamp, source:str):
        self.value = data.copy()
        self.value_tuple = data_entry(self.update_cnt, self.value, timestamp, source)
        self.last_update = timestamp
        self.history.append(self.value_tuple)
        self.update_cnt += 1

class DoubleDevice(Device):
    def __init__(self, name, drf2, history=10):
        """
        Floating point device with a circular history buffer
        """
        super().__init__(name, drf2, history)

class BPMDevice(Device):
    def __init__(self, name, drf2, array_length=1000, history=10):
        """
        BPM device with a circular history buffer
        """
        self.array_length = array_length
        super().__init__(name, drf2, history)

    # def get_value(self):
    #     if not self.access_method:
    #         raise ValueError('This device does not have an associated adapter')
    #     print('This method is discouraged - please consider using device sets')
    #     ds = devicesets.BPMDeviceSet(method)
    #     ds.add(self)
    #     values = ds.getValues()
    #     return values[self.name]

    def _update_oneshot(self, data, timestamp, source):
        self.value = data.copy()
        self.last_update = timestamp
        self.history.append((self.cnt, self.last_update, source, self.value))
        self.cnt += 1

#####

class DeviceSet:
    """
    A container of devices and other device sets, which is subclassed by more specialized sets. Implemented as
    an unbalanced tree, with 'nodes' containing only other device sets and 'leafs' containing only devices.
    Supports iterable and dictionary access modes.
    """
    def __init__(self, name:str, members:list):
        # This is classical tree structure
        node =  all([isinstance(ds,DeviceSet) for ds in members])
        leaf =  all([isinstance(ds,Device) for ds in members])
        assert node or leaf

        if node:
            self.leaf = False
            self.children = {ds.name:ds for ds in members}
            assert len(self.children) == len(members)
            self.devices = None
            #self.device_names = None
            for c in self.children:
                c.parent = self
        else:
            self.leaf = True
            self.devices = {d.name:d for d in members}
            assert len(self.devices) == len(members)
            #self.device_names = [d.name for d in args]
            self.children = None
            self.parent = None
        self.adapter = None
        self.name = name

    def __iter__(self):
        self.iter_node = self
        return self

    def __next__(self):
        """
        Depth-first pre-order traversal
        :return:
        """
        if not self.iter_node.leaf:
            for c in self.children:
                yield next(c)
        else:
            for d in self.devices:
                yield d

    def add(self, device):
        if isinstance(device, Device):
            if device.name not in self.devices.keys():
                self.devices[device.name] = device
            else:
                raise AttributeError('This device is already in the set')
        else:
            raise AttributeError('You must add a device object!')

    def remove(self, name:str):
        """
        Removes and returns the desired device or device set, or None if not found
        :param name: Unique name of the element
        :return: The removed member
        """
        if self.leaf:
            if name in self.devices:
                return self.devices.pop(name)
        else:
            if name in self.children:
                return self.children.pop(name)

    def gather(self):
        return

    def check_acquisition_available(self, method:str) -> bool:
        if not self.adapter:
            raise AttributeError('Adapter is not set!')
        if self.leaf:
            if len(self.devices) == 0:
                raise Exception("Add devices before starting acquisition!")
            return self.adapter.check_availability(self.devices, method)
        else:
            return all([c.check_acquisition_available(method) for c in self.children])

    def check_acquisition_supported(self, method:str) -> bool:
        if not self.adapter:
            raise AttributeError('Adapter is not set!')
        if self.leaf:
            return self.adapter.check_supported(self.devices, method)
        else:
            return all([c.check_acquisition_supported(method) for c in self.children])

    def start_oneshot(self) -> bool:
        if not self.check_acquisition_supported():
            raise Exception('Acquisition is not supported for method: {}'.format('oneshot'))
        if not self.check_acquisition_available('oneshot'):
            return False
        if self.leaf:
            self.adapter.oneshot()
        else:
            for c in self.children:
                c.adapter.oneshot()

    def start_polling(self) -> bool:
        if not self.check_acquisition_supported():
            raise Exception('Acquisition is not supported for method: {}'.format('polling'))
        if not self.check_acquisition_available('polling'):
            return False
        if self.leaf:
            self.adapter.start()
        else:
            for c in self.children:
                c.adapter.start()

    # def stopAcquisition(self):
    #     if not running:
    #         print('Acquisition is not running!')
    #     if method == 'aclasync':
    #         continue
    #     elif method == 'dpm_polling':
    #         try:
    #             self.DPM.stop()
    #             self.DPM = None
    #             self.acnet._close()
    #         except Exception e:
    #             print(e)
    #             raise
    #     else:
    #         raise Exception('Continuous acquisition is not supported for method: {}'.format(self.method))


class BPMDeviceSet(DeviceSet):
    """
    Leaf container of BPM and other array devices. Exposes methods for fast batch readout, and enforces data uniformity.
    """
    def __init__(self, name:str, members:list, enforce_array_length:int=1000):
        leaf = all([isinstance(ds,Device) for ds in members])
        if not leaf:
            raise AttributeError('BPM set can only contain devices - use generic DeviceSet for coalescing')
        assert enforce_array_length is None or isinstance(enforce_array_length,int)
        super().__init__(name, members)
        self.array_length = enforce_array_length

    def add(self, device:Device):
        if isinstance(device, BPMDevice):
            if self.array_length is not None and self.array_length != device.array_length:
                print('Device array length does not match set - using set value {}'.format(self.array_length))
            super().add(device)
        else:
            raise Exception("Device is not a BPM!")


class Adapter:
    supported = []

    def __init__(self):
        pass

    def check_supported(self, devices, method):
        return method in self.supported

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


class ACLAsync(Adapter):
    """
    ACL web adapter. Works by fetching from the ACL web proxy via HTTP. Supports single-shot with many parallel
    requests using AsyncIO framework, with fallback to single thread via urllib,
    and quasi-polling mode that loops over single-shot commands.
    """
    def __init__(self, fallback:bool=False):
        super().__init__()
        self.name = 'ACL Async'
        self.supported = ['oneshot', 'polling']
        self.rate_limit = [-1, -1]
        self.fallback = fallback
        try:
            import httpx
            #import yarl
        except ImportError:
            print('HTTP functionality requires httpx library')
            raise
        if fallback:
            self.client = httpx.Client()
        else:
            self.client = httpx.AsyncClient()

    def __del__(self):
        try:
            if self.fallback:
                self.client.close()
            else:
                async def terminate():
                    await self.client.aclose()
                asyncio.run(terminate())
        except Exception as e:
            print(f'Issue with disposing adapter {self.name}')
            print(e)

    def start_oneshot(self, ds:DeviceSet) -> dict:
        assert ds.leaf
        devices = ds.devices
        dev_drf2 = [d.drf2 for d in devices.values()]
        bpmlist = [d.name for d in devices]
        bpmstring = ','.join(bpmlist)
        if self.fallback:
            c = self.client
            try:
                txt = ('http://www-ad.fnal.gov/cgi-bin/acl.pl?acl=' +
                       f'read/row/pendWait=0.5+devices="{bpmstring[:-1]}"+/num_elements=1000')
                print(txt)
                resp = c.get(txt)
                t1 = datetime.datetime.utcnow().timestamp()
                text = resp.text
                if "DPM_PEND" in text:
                    print('Bad ACL output - DPM_PEND')
                if 'mm' not in text or 'raw' not in text:
                    raise Exception("Bad ACL output")
                split_text = text.strip().split('\n')
                data = {}
                for line in filter(None, split_text):
                    spl = line.strip().split(' ')
                    devname = spl[0].split('@')[0].split('[')[0]
                    data[devname] = np.array([float(v) for v in spl[2:-1]])
                for d in ds.devices:
                    d.update(data[d.name],t1,'unknown')
                return data
            except Exception as e:
                print(e, sys.exc_info())
                # (None, datetime.datetime.utcnow().timestamp(), 'error')
                return None
        # else:
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