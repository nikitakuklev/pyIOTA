import pyIOTA.acnet as acnet
import pyIOTA.iota as iota

devices = [acnet.frontends.DoubleDevice(b) for b in iota.QUADS.ALL_CURRENTS]
ds = acnet.frontends.DeviceSet(name='bpms', members=devices)
ds.adapter = acnet.frontends.ACNETRelay()
ds.read()

print([[d.name, d.value_string] for d in devices])
