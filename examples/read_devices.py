import pyIOTA.acnet as acnet
import pyIOTA.iota as iota

# devs = [acnet.frontends.BPMDevice(b) for b in iota.BPMS.HA]
# ds = acnet.frontends.BPMDeviceSet(name='bpms', members=devs, enforce_array_length=1000)
# ds.adapter = acnet.frontends.ACL(fallback=False)
# ds.readonce()

# devices = [acnet.frontends.BPMDevice(b) for b in iota.BPMS.HA][0:1]
# ds = acnet.frontends.BPMDeviceSet(name='bpms', members=devices, enforce_array_length=1000)
# ds.adapter = acnet.frontends.ACNETRelay()
# ds.readonce()
# print(ds)

# devices = [acnet.frontends.DoubleDevice(b) for b in iota.QUADS.ALL_CURRENTS][0:1]
# ds = acnet.frontends.DeviceSet(name='bpms', members=devices)
# ds.adapter = acnet.frontends.ACNETRelay()
# ds.readonce()

devices = [acnet.frontends.DoubleDevice(b) for b in iota.QUADS.ALL_CURRENTS]
ds = acnet.frontends.DeviceSet(name='bpms', members=devices)
ds.adapter = acnet.frontends.ACNETRelay()
ds.readonce()

print([[d.name, d.value_string] for d in devices])
