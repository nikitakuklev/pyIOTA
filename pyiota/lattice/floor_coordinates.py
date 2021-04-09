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


def local_vectors(el, l=None):
    """
    Propagate in frame of element - assume we start at x=y=z=angle=0. Then, any element without a bend is along +x.
    """
    xl_ent = yl_ent = 0.0
    if l:
        assert 0 <= l <= el.l
    vl_ent = Vector3D.normalize(np.array([el.l, 0.0, 0.0]))
    if isinstance(el, SBend):
        # Switch to polar, move on arc
        r = np.abs(el.l / el.angle)
        t = np.pi / 2 if el.angle > 0 else 3 * np.pi / 2
        if l:
            dt = el.angle * l/el.l
        else:
            dt = el.angle
        t -= dt
        # Back to cartesian
        x, y = pol2cart(r, t)
        # Shift to have relative deltas
        xl_ext = x - xl_ent
        yl_ext = y - (yl_ent + r)
        vl_ext = Vector3D.rotate_z(vl_ent, -dt)
    else:
        # Straight element
        if l:
            xl_ext = xl_ent + l
        else:
            xl_ext = xl_ent + el.l
        yl_ext = yl_ent
        vl_ext = vl_ent.copy()
    vl_ent_to_ext = np.array([xl_ext-xl_ent, yl_ext-yl_ent, 0.0])
    return xl_ent, yl_ent, xl_ext, yl_ext, vl_ent, vl_ent_to_ext, vl_ext


def global_vectors(vl_ent, vl_ent_to_ext, vl_ext, angle: float):
    """ Add global vectors from local ones based on entrance angle """
    # Global normal at entrance
    v_ent = Vector3D.rotate_z(vl_ent, angle)
    # Global vector to next element
    v_ent_to_ext = Vector3D.rotate_z(vl_ent_to_ext, angle)
    # Global normal at exit
    v_ext = Vector3D.rotate_z(vl_ext, angle)
    # Global rotation at entrance
    _, rot_v_ent = cart2pol(v_ent[0], v_ent[1])
    return v_ent, v_ent_to_ext, v_ext, rot_v_ent


def vector_at(lattice, s:float):
    """ Find direction at position s """
    el = element_at(lattice, s)
    if el.l == 0.0:
        # Copy vector
        return el.v_ent.copy()
    else:
        # Need to partially propagate
        dl = s - el.s_start
        xl_ent, yl_ent, xl_ext, yl_ext, vl_ent, vl_ent_to_ext, vl_ext = local_vectors(el, dl)
        return vl_ext


def element_at(lattice, s:float):
    """ Return element at position s (if many, first one) """
    #lengths = [0.0] + [el.l for el in lattice.sequence]
    #clengths = np.cumsum(lengths)
    starts = np.array([el.s_start for el in lattice.sequence])
    #return lattice.sequence[np.argmax(clengths >= s)-1]
    return lattice.sequence[np.argmin(starts <= s)-1]


def compute_floor(lattice: MagneticLattice, monitors=None, correctors=None):
    """
    Compute floor coordinates in 2D.
    :param lattice: OCELOT lattice
    :param monitors: BPMs, which are plotted in addition to those in lattice
    :return:
    """
    monitors = monitors or []
    loc = np.array([0.0, 0.0, 0.0])
    angle = 0.0
    for el in lattice.sequence:
        xl_ent, yl_ent, xl_ext, yl_ext, vl_ent, vl_ent_to_ext, vl_ext = local_vectors(el)
        # Set element properties
        el.vl_ent = vl_ent
        el.vl_ent_to_ext = vl_ent_to_ext
        el.vl_ext = vl_ext
        # # Global normal at entrance
        # el.v_ent = Vector3D.rotate_z(vl_ent, angle)
        # # Global vector to next element
        # el.v_ent_to_ext = Vector3D.rotate_z(vl_ent_to_ext, angle)
        # # Global rotation at entrance
        # _, el.rot_v_ent = cart2pol(el.v_ent[0], el.v_ent[1])
        # # Origin coordinate
        # el.o = loc.copy()

        v_ent, v_ent_to_ext, v_ext, rot_v_ent = global_vectors(vl_ent, vl_ent_to_ext, vl_ext, angle)
        # Set element properties
        el.o = loc.copy()
        el.v_ent = v_ent
        el.v_ent_to_ext = v_ent_to_ext
        el.v_ext = v_ext
        el.rot_v_ent = rot_v_ent

        # Update current global location
        loc += el.v_ent_to_ext
        # print(f'{el.__class__.__name__:<12s} {el.l:.03f} {el.angle:.03f} {angle:.03f} {el.o} {el.v_local} {el.v}')
        if isinstance(el, SBend):
            angle -= el.angle

    lengths = [0.0] + [el.l for el in lattice.sequence]
    s = np.cumsum(lengths)
    s_dict = {k: v for k, v in zip(lattice.sequence, s)}
    s_inverse_dict = {v: k for k, v in zip(lattice.sequence, s)}
    for m in monitors:
        #print(m.id, m.ref_el.id)
        m.s_mid = s_dict[m.ref_el] + m.shift
        if m.s_mid > lattice.totalLen:
            raise Exception(f'Monitor {m} is outside lattice length at {m.s_mid}')

        loc_el = element_at(lattice, m.s_mid)
        dl = m.s_mid - loc_el.s_start
        print(f'Resolved {m.id} (ref {m.ref_el.id} + {m.shift}) = {m.s_mid:.4f} (at ({loc_el.id}) +{dl:.4f})')
        print(f'({loc_el.id}) - {loc_el.s_start:.4f}:{loc_el.l:.4f}:{loc_el.s_end:.4f}')
        #p_el = lattice.sequence[lattice.sequence.index(loc_el)-1]
        #print(f'({p_el.id}) - {p_el.s_start:.4f}:{p_el.l:.4f}:{p_el.s_end:.4f}')


        xl_ent, yl_ent, xl_ext, yl_ext, vl_ent, vl_ent_to_ext, vl_ext = local_vectors(loc_el, dl)
        v_ent, v_ent_to_ext, v_ext, rot_v_ent = global_vectors(vl_ent, vl_ent_to_ext, vl_ext, loc_el.rot_v_ent)
        m.o = loc_el.o + v_ent_to_ext
        #m.v_ent_to_ext = v_ext.copy()
        # For thin monitor, same vectors
        m.v_ent = v_ext.copy()
        m.v_ext = v_ext.copy()
        _, rot_v_ext = cart2pol(v_ext[0], v_ext[1])
        print(vl_ent, vl_ext)
        print(rot_v_ent, rot_v_ext, loc_el.rot_v_ent, v_ent, v_ext)
        m.rot_v_ent = m.rot_v_ext = rot_v_ext


def plot_floor(*args, **kwargs):
    import pyIOTA.gui as gui
    gui.plot_floor_map(*args, **kwargs)

