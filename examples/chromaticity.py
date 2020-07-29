"""
Find IOTA lattice chromaticity by tracking
Intended for profiling
"""
import time
from pathlib import Path

import ocelot
import pyIOTA.lattice as lat
import pyIOTA.sixdsim as sixdsim
from ocelot.cpbd.elements import *

lattice, correctors, monitors, info, variables = sixdsim.parse_lattice(Path('IOTA_1NL_100MeV_v8.6.1.4.6ds'),
                                                                       verbose=False)
method = ocelot.MethodTM()
method.global_method = ocelot.SecondTM
box = lat.LatticeContainer(name='IOTA', lattice=lattice, correctors=correctors,
                           monitors=monitors, info=info, variables=variables, silent=False, method=method)
box.remove_elements(box.get_all(CouplerKick))
box.transmute_elements(box.get_all(Cavity), Drift)
box.transmute_elements(box.filter_elements('oq.*', Quadrupole), Drift)
box.transmute_elements(box.filter_elements('nl.*', Quadrupole), Drift)
box.merge_drifts()
box.update_element_positions()
box.lattice.update_transfer_maps()

tws = box.update_twiss()

print('Starting tracking...')
start = time.time()
chroma = lat.chromaticity(box.lattice, tws[0], method='track', method_kwargs={'n_turns': 2048})
print(f'Perf: ({time.time()-start : .4f}) - ({(time.time()-start)/1024 :.4f}) per turn')
print(chroma)
