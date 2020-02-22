import pyIOTA.acnet as acnet
import pyIOTA.iota as iota

# devs = [acnet.frontends.BPMDevice(b) for b in iota.BPMS.HA]
# ds = acnet.frontends.BPMDeviceSet(name='bpms', members=devs, enforce_array_length=1000)
# ds.adapter = acnet.frontends.ACL(fallback=False)
# ds.start_oneshot()

# devices = [acnet.frontends.BPMDevice(b) for b in iota.BPMS.HA][0:1]
# ds = acnet.frontends.BPMDeviceSet(name='bpms', members=devices, enforce_array_length=1000)
# ds.adapter = acnet.frontends.ACNETRelay()
# ds.start_oneshot()
# print(ds)

# devices = [acnet.frontends.DoubleDevice(b) for b in iota.QUADS.ALL_CURRENTS][0:1]
# ds = acnet.frontends.DeviceSet(name='bpms', members=devices)
# ds.adapter = acnet.frontends.ACNETRelay()
# ds.start_oneshot()

devices = [acnet.frontends.DoubleDevice(b) for b in iota.QUADS.ALL_CURRENTS]
ds = acnet.frontends.DeviceSet(name='bpms', members=devices)
ds.adapter = acnet.frontends.ACNETRelay()
ds.start_oneshot()

print([[d.name, d.value] for d in devices])
