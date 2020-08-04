__all__ = ['compute_floor']

import numpy as np
from pyIOTA.math import pol2cart, cart2pol, Vector3D
from ocelot import SBend, MagneticLattice

"""
Calculate 2D floor map. Does not take into account element tilts, but uses 3D vectors,
so should be easy to add if you have such an interesting machine.

See pyIOTA.gui.plots for floor plot

Angle convention is theta=0 along positive x-axis, with counter-clockwise positive theta.
+y -> pi/2
-x -> pi
-y -> 3pi/2
"""


def add_local_vectors(el):
    """
    Propagate in frame of element - assume we start at x=y=angle=0. Then, any element without a bend is along +x.
    """
    el.xl_ent = el.yl_ent = 0.0
    if isinstance(el, SBend):
        # Switch to polar, move on arc
        r = np.abs(el.l / el.angle)
        t = np.pi / 2 if el.angle > 0 else 3 * np.pi / 2
        t -= el.angle
        # Back to cartesian
        x, y = pol2cart(r, t)
        # Shift to have relative deltas
        el.xl_ext = x - el.xl_ent
        el.yl_ext = y - (el.yl_ent + r)
    else:
        # Straight element
        el.xl_ext = el.xl_ent + el.l
        el.yl_ext = el.yl_ent
    el.vl_ent_to_ext = np.array([el.xl_ext-el.xl_ent, el.yl_ext-el.yl_ent, 0.0])
    el.vl_ent = Vector3D.normalize(np.array([el.l, 0.0, 0.0]))


def compute_floor(lattice: MagneticLattice, monitors=None, correctors=None):
    """
    Compute floor coordinates in 2D.
    :param lattice: OCELOT lattice
    :param monitors: BPMs, which are plotted in addition to those in lattice
    :return:
    """
    loc = np.array([0.0, 0.0, 0.0])
    angle = 0.0
    for el in lattice.sequence:
        add_local_vectors(el)
        # Global normal at entrance
        el.v_ent = Vector3D.rotate_z(el.vl_ent, angle)
        # Global vector to next element
        el.v_ent_to_ext = Vector3D.rotate_z(el.vl_ent_to_ext, angle)
        # Global rotation at entrance
        _, el.rot_v_ent = cart2pol(el.v_ent[0], el.v_ent[1])
        # Origin coordinate
        el.o = loc.copy()
        # Update current global location
        loc += el.v_ent_to_ext
        # print(f'{el.__class__.__name__:<12s} {el.l:.03f} {el.angle:.03f} {angle:.03f} {el.o} {el.v_local} {el.v}')
        if isinstance(el, SBend):
            angle -= el.angle


def plot_floor(*args, **kwargs):
    import pyIOTA.gui as gui
    gui.plot_floor_map(*args, **kwargs)

