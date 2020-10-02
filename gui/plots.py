import copy
import logging

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict, Union

from pyIOTA.lattice.elements import ILMatrix
from matplotlib import font_manager
import matplotlib.patches as patches
import matplotlib as mpl
from ocelot import Quadrupole, Bend, SBend, RBend, Vcor, Hcor, Sextupole, Undulator, \
    Cavity, Multipole, Marker, Edge, Octupole, Matrix, Monitor, Drift, Solenoid, UnknownElement, TDCavity, TWCavity, \
    MagneticLattice, Element

logger = logging.getLogger(__name__)


def plot(*args, **kwargs):
    return plot_simple(*args, **kwargs)


def plot_grid(*args, **kwargs):
    return plot_simple_grid(*args, **kwargs)


def plot_simple(*args,
                x: Union[List, np.ndarray] = None,
                fmt: str = None,
                xlabel: str = None,
                ylabel: str = None,
                sizes: Tuple = (9, 3),
                demean: bool = False,
                normalize: bool = False,
                twiny_args: Union[List, Dict, np.ndarray] = None,
                **kwargs):
    """
    Routine to plot data sharing a single plot. Each tuple is treated as (x,y) pair, and any number of singletons,
    lists or dicts containing these tuples can be supplied. Plotting on right y axis is supported with respective arguments.
    :param x: Shared x axis
    :param fmt: Matplotlib format
    :param xlabel: X label, same as matplotlib
    :param ylabel: Y label, same as matplotlib
    :param sizes: Plot size
    :param demean: Demean each data series
    :param normalize: Minmax normalize each data series
    :param twiny_args: Right y axis arguments
    :return:
    """
    arrays = []
    arrays_twiny = []
    twiny_args = twiny_args or []
    fmt = fmt or '-'
    if len(twiny_args) == 1:
        twiny_args = [twiny_args]
    for inputs, arrs in zip([args, twiny_args], [arrays, arrays_twiny]):
        for i, data_dict in enumerate(inputs):
            if isinstance(data_dict, list):
                if all(isinstance(n, np.number) for n in data_dict):
                    arrs.append((str(i), data_dict))
                else:
                    for v in data_dict:
                        arrs.append((str(i), v))
            elif isinstance(data_dict, tuple):
                # Put x,y tuple
                arrs.append((str(i), data_dict))
            elif isinstance(data_dict, np.ndarray):
                # Put y only
                arrs.append((str(i), data_dict))
            elif isinstance(data_dict, dict):
                for k, v in data_dict.items():
                    arrs.append((k, v))
            else:
                raise Exception(f'Supplied argument is not supported: {data_dict.__class__.__name__}')

    fig, ax = plt.subplots(1, 1, figsize=(sizes[0], sizes[1]))
    for z, (i, v) in enumerate(arrays):
        if isinstance(v, tuple):
            x_local = v[0]
            v = v[1]
        else:
            x_local = None
        if demean:
            v = v - np.mean(v)
        if normalize:
            from sklearn.preprocessing import minmax_scale
            v = minmax_scale(v)
        if x_local is not None:
            ax.plot(x_local, v, fmt, label=i, zorder=z + 100, **kwargs)
        elif x is not None:
            ax.plot(x, v, fmt, label=i, zorder=z + 100, **kwargs)
        else:
            ax.plot(v, fmt, label=i, zorder=z + 100, **kwargs)
        # ax.set_title(f"{i}|{k}")
    ax.legend(loc=1)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    if arrays_twiny:
        axy = ax.twinx()
        axy.grid(False)
        axy._get_lines.prop_cycler = ax._get_lines.prop_cycler
        for z, (i, v) in enumerate(arrays_twiny):
            if demean:
                v = v - np.mean(v)
            if normalize:
                from sklearn.preprocessing import minmax_scale
                v = minmax_scale(v)
            if x:
                axy.plot(x, v, fmt, label=i, zorder=z, **kwargs)
            else:
                axy.plot(v, fmt, label=i, zorder=z, **kwargs)
            # ax.set_title(f"{i}|{k}")
        axy.legend()
        ax.set_zorder(axy.get_zorder() + 1)
    ax.patch.set_visible(False)
    fig.tight_layout()
    return fig, ax


def plot_simple_grid(*args,
                     nperrow: int = 11,
                     sizes: Tuple[int, int] = (2, 2),
                     demean: bool = False,
                     normalize: bool = False,
                     paired_bpm_mode: bool = False,
                     fontsize: int = 10,
                     scatter: bool = False,
                     colorbar: bool = False,
                     no_title_idx=True,
                     no_title=False,
                     sharey=True,
                     sharex=True,
                     **kwargs):
    """
    Plot a grid of data sharing x/y axes. Each top level argument creates a new plot grid.
    Each argument must be a dict or list of dicts or list of arrays or list of tuples.
    One multi plot per each argument (after flattening) is produced.
    All lowest level entries must either be arrays (Y), or 2/3-tuples of arrays (X,Y,{Z})
    Example: {'1': [(x1,y1),(x2,y2)],'2':[(x1,y1),(x2,y2)]} makes 2x1 grid with 2 lines each
    :param scatter: Toggle scatter plot - requires Z
    :param normalize:
    :param demean: Whether to demean Y data
    :param paired_bpm_mode: IOTA specific - special mode to plot left-right BPM pairs in same columns
    :param nperrow: Plots per row
    :param sizes: Figure size per each row/column
    :return: fig, ax
    """
    fig_l = []
    ax_l = []
    for entry in args:
        if isinstance(entry, list):
            if all(isinstance(v, (np.ndarray, pd.Series)) for v in entry):
                # Have list of arrays
                entry = {i: v for i, v in enumerate(entry)}
            elif all(isinstance(v, tuple) for v in entry):
                # List of x,y tuples
                entry = {i: v for i, v in enumerate(entry)}
            else:
                # Have list of dicts
                entry = {k: v for d in entry for k, v in d.items()}
        elif isinstance(entry, tuple):
            entry = {1: entry}
        if not isinstance(entry, dict):
            raise Exception(f'Supplied argument is not a dict: {entry}')
        if len(entry) == 0:
            logger.warning('Skipping empty plot')
            continue
        n_rows = math.ceil(len(entry) / nperrow)
        width = nperrow if n_rows > 1 else len(entry)
        fig, ax = plt.subplots(n_rows, width, figsize=(sizes[0] * width, sizes[1] * n_rows),
                               sharex=sharex, sharey=sharey, squeeze=False)
        fig_l.append(fig)
        # ax_l.append(ax.flatten())
        if n_rows == width == 1:
            ax = np.array([[ax]])
        ax_l.append(ax)
        # print([ax])
        if paired_bpm_mode:
            keys = [k for k in entry.keys()]
            roots = [k[:-2] for k in keys]
            seen = set()
            roots = [x for x in roots if not (x in seen or seen.add(x))]
            if not nperrow >= len(roots):
                raise Exception(f'Not enough columns ({nperrow}) to fit all pairs ({len(roots)})')
            for j, root in enumerate(roots):
                i = 0
                for z, (k, v) in enumerate(entry.items()):
                    if root in k:
                        if demean:
                            v = v - np.nanmean(v)
                        ax[i, j].plot(v, lw=0.5)
                        ax[i, j].set_title(f"{z}|{k}")
                        i += 1
        else:
            ax = ax.flatten()
            tuples_list = []
            for i, (k, v) in enumerate(entry.items()):
                if isinstance(v, tuple):
                    # Tuple of things
                    if all(isinstance(v2, (np.ndarray, pd.Series, list)) for v2 in v):
                        # Single (x,y,z) tuple
                        x = v[0]
                        y = v[1]
                        z = v[2] if len(v) == 3 else None
                        tuples_list.append(([x], [y], [z], k))
                    else:
                        # Tuple of (x,y,z) tuples
                        assert all(len(v2) == 2 for v2 in v) or all(len(v2) == 3 for v2 in v)
                        x = [v2[0] if not isinstance(v2, np.ndarray) else np.arange(len(v)) for v2 in v]
                        y = [v2[1] if not isinstance(v2, np.ndarray) else v2 for v2 in v]
                        z = [v2[2] if not isinstance(v2, np.ndarray) and len(v2) == 3 else None for v2 in v]
                        tuples_list.append((x, y, z, k))
                elif isinstance(v, list):
                    # List of things
                    if all(isinstance(v2, (np.ndarray, pd.Series)) for v2 in v):
                        # List of arrays y
                        x = [np.arange(len(v2)) for v2 in v]
                        y = [v2 for v2 in v]
                        z = [None for v2 in v]
                        tuples_list.append((x, y, z, k))
                    else:
                        # List of tuples (x,y,z)
                        assert all(len(v2) == 2 for v2 in v) or all(len(v2) == 3 for v2 in v)
                        x = [v2[0] if not isinstance(v2, np.ndarray) else np.arange(len(v2)) for v2 in v]
                        y = [v2[1] if not isinstance(v2, np.ndarray) else v2 for v2 in v]
                        z = [v2[2] if not isinstance(v2, np.ndarray) and len(v2) == 3 else None for v2 in v]
                        tuples_list.append((x, y, z, k))
                elif isinstance(v, np.ndarray):
                    # Just have x
                    x = np.arange(len(v))
                    y = v
                    tuples_list.append(([x], [y], [None], k))
                else:
                    raise Exception(f'Unknown data type: ({type(v)})')
                if scatter:
                    # must have z
                    assert all(v[2] is not None for v in tuples_list)

            # global_max = np.max([np.max(v[1]) for v in tuples_list])
            for i, (xl, yl, zl, k) in enumerate(tuples_list):
                for x, y, z in zip(xl, yl, zl):
                    if demean:
                        y = y - np.nanmean(y)
                    if normalize:
                        from sklearn.preprocessing import minmax_scale
                        y = minmax_scale(y)  # (y-np.min(y))/np.linalg.norm(y-np.min(y))
                    if scatter:
                        if z is not None:
                            s = ax[i].scatter(x, y, c=z, **kwargs)
                        else:
                            s = ax[i].scatter(x, y, **kwargs)
                        if colorbar:
                            fig.colorbar(s, ax=ax[i])
                    else:
                        ax[i].plot(x, y, **kwargs)
                if not no_title:
                    if no_title_idx:
                        ax[i].set_title(f"{k}", fontsize=fontsize)
                    else:
                        ax[i].set_title(f"{i}|{k}", fontsize=fontsize)
    if len(args) == 1:
        fig = fig_l[0]
        ax = ax_l[0]
    else:
        fig = fig_l
        ax = ax_l
    return fig, ax


def get_ax(fig=None, ax=None):
    """ Get axes or make default """
    if not fig and not ax:
        fig, ax = plt.subplots(1, 1, figsize=(9, 6))
    elif not ax:
        ax = fig.gca()
    else:
        pass
    return fig, ax


def plot_floor_map(fig=None,
                   ax=None,
                   lattice: MagneticLattice = None,
                   monitors: List = None,
                   legend: bool = False,
                   **kwargs):
    fig, ax = get_ax(fig, ax)
    if 'octupole_scaling' in kwargs:
        maxh = kwargs['octupole_scaling']
        octupoles = [el for el in lattice.sequence if isinstance(el, Octupole)]
        max_str = max(el.k3 for el in octupoles)
        kwargs['octupoles_scale_factor'] = max_str
    for el in lattice.sequence:
        draw_element(ax, el, **kwargs)
    if monitors:
        for el in monitors:
            draw_element(ax, el, **kwargs)
    ax.set_aspect('equal')
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    xc = 0.0
    yc = (max([el.o[1] for el in lattice.sequence]) + min(el.o[1] for el in lattice.sequence)) / 2
    if legend:
        classes = [Quadrupole, SBend, Sextupole]
        handles = []
        for c in classes:
            first_el = [el for el in lattice.sequence if isinstance(el, c)][0]
            p0 = first_el.patches[0]
            handles.append(p0)
        ax.legend(handles=handles, labels=[c.__name__ for c in classes], loc='center', fontsize=16)


def draw_element(ax, el: Element, draw_edges=True, debug=True, **kwargs):
    # Transform global frame - move to element origin and rotate
    t = mpl.transforms.Affine2D().rotate(el.rot_v_ent).translate(el.o[0], el.o[1]) + ax.transData
    p_list = []

    ec = 'k' if draw_edges else None
    lw_element_edge = 0.5
    lw_orbit = 4.0
    pkw = {'ec': ec, 'lw': lw_element_edge}

    if isinstance(el, Drift):
        x_list = [el.o[0], el.o[0] + el.v_ent_to_ext[0]]
        y_list = [el.o[1], el.o[1] + el.v_ent_to_ext[1]]
        ax.plot(x_list, y_list, '-', c='k', lw=lw_orbit, zorder=1)
    elif isinstance(el, SBend):
        h = 0.4
        r = np.abs(el.l / el.angle)
        a1 = 90 - el.angle * 180 / np.pi
        p2 = patches.Arc((0, -r), 2 * r, 2 * r, theta1=a1, theta2=90, color="k", lw=4.0, zorder=1)
        p = arc_patch((0, -r), 2 * r, 2 * r, theta1=a1, theta2=90, color="blue", alpha=0.80, **pkw)
        p_list.append(p)
        p_list.append(p2)
    elif isinstance(el, Monitor):
        h = 0.3
        l = 0.05
        # Back off by half the virtual bpm length to center box
        # t2 = mpl.transforms.Affine2D().rotate(el.rot_v_ent)
        # t2 = t2.translate(el.o[0] - el.v_ext[0] * l / 2, el.o[1] - el.v_ext[1] * l / 2) + ax.transData
        p = patches.Rectangle((-l / 2, -h / 2), l, h, color="gray", alpha=0.90, **pkw)
        p_list.append(p)
        h = 1.0
        p_list.append(patches.Rectangle((0, -h / 2), 0, h, color="k", alpha=0.80, **pkw))
        p_list.append(p)
    else:
        x_list = [el.o[0], el.o[0] + el.v_ent_to_ext[0]]
        y_list = [el.o[1], el.o[1] + el.v_ent_to_ext[1]]
        ax.plot(x_list, y_list, '-', c='k', lw=lw_orbit, zorder=1)
        if isinstance(el, Quadrupole) and el.id.startswith('Q'):
            h = 0.5
            p = patches.Rectangle((0, -h / 2), el.l, h, color="green", alpha=0.80, **pkw)
            p_list.append(p)
        elif isinstance(el, Quadrupole) and el.id.startswith('S'):
            h = 0.4
            p = patches.Rectangle((0, -h / 2), el.l, h, color="red", alpha=0.80, **pkw)
            p_list.append(p)
        elif isinstance(el, Sextupole):
            h = 0.3
            p = patches.Rectangle((0, -h / 2), el.l, h, color="purple", alpha=0.80, **pkw)
            p_list.append(p)
        elif isinstance(el, Octupole):
            ocp_maxh = kwargs.get('octupole_scaling', None)
            ocp_sc = kwargs.get('octupoles_scale_factor', None)
            if ocp_maxh and ocp_sc:
                h = ocp_maxh * el.k3 / ocp_sc if el.k3 != 0.0 else 0.2
            else:
                h = 0.2
            p = patches.Rectangle((0, -h / 2), el.l, h, color="orange", alpha=0.80, **pkw)
            p_list.append(p)

    if debug:
        h = 1.0
        p_list.append(patches.Rectangle((0, -h / 2), 0, h, color="k", alpha=0.80, **pkw))

    for p in p_list:
        if isinstance(p, tuple):
            p[0].set_transform(p[1])
            ax.add_patch(p[0])
        else:
            p.set_transform(t)
            ax.add_patch(p)

    el.patches = p_list


def arc_patch(center, w, h, theta1, theta2, resolution=50, t_out=0.2, t_in=0.2, **kwargs):
    """ Generates a filled arc, uses same notation as patches.Arc """
    assert w == h
    radius = w / 2
    r1, r2 = radius + t_out, radius - t_in
    theta = np.linspace(np.radians(theta1), np.radians(theta2), resolution)
    x = np.hstack([r1 * np.cos(theta), r2 * np.cos(theta[::-1])])
    y = np.hstack([r1 * np.sin(theta), r2 * np.sin(theta[::-1])])
    points = np.vstack((x + center[0], y + center[1]))
    poly = patches.Polygon(points.T, closed=True, **kwargs)
    return poly


def plot_opt_func(fig, lat, tws, top_plot=["Dx"], legend=False, fig_name=None, grid=True, font_size=12,
                  excld_legend=None, ontop: bool = False):
    """
    Modified from OCELOT

    function for plotting: lattice (bottom section), vertical and horizontal beta-functions (middle section),
    other parameters (top section) such as "Dx", "Dy", "E", "mux", "muy", "alpha_x", "alpha_y", "gamma_x", "gamma_y"
    :param fig: Figure to use
    :param lat: MagneticLattice,
    :param tws: list if Twiss objects,
    :param top_plot:  ["Dx"] - parameters which displayed in top section. Can be any attribute of Twiss class, e.g. top_plot=["Dx", "Dy", "alpha_x"]
    :param legend: True - displaying legend of element types in bottom section,
    :param fig_name: None - name of figure,
    :param grid: True - grid
    :param font_size: 16 - font size for any element of plot
    :param excld_legend: None, exclude type of element from the legend, e.g. excld_legend=[Hcor, Vcor]
    :return:
    """
    if fig is None:
        if fig_name is None:
            fig = plt.figure()
        else:
            fig = plt.figure(fig_name)

    left, width = 0.1, 0.85

    rect1 = [left, 0.65, width, 0.3]
    rect2 = [left, 0.19, width, 0.46]
    rect3 = [left, 0.07, width, 0.12]

    ax_top = fig.add_axes(rect1)
    ax_b = fig.add_axes(rect2, sharex=ax_top)  # left, bottom, width, height
    ax_el = fig.add_axes(rect3, sharex=ax_top)
    for ax in ax_b, ax_el, ax_top:
        if ax != ax_el:
            for label in ax.get_xticklabels():
                label.set_visible(False)

    ax_b.grid(grid)
    ax_top.grid(grid)
    ax_el.set_yticks([])
    ax_el.grid(grid)

    fig.subplots_adjust(hspace=0)
    if not isinstance(tws[0], list):
        tws_list = [tws]
    else:
        tws_list = tws

    for i, tws in enumerate(tws_list):
        beta_x = [p.beta_x for p in tws]
        beta_y = [p.beta_y for p in tws]
        S = [p.s for p in tws]

        plt.xlim(S[0], S[-1])
        label = str(i + 1) if i > 0 else ''
        plot_disp(ax_top, tws, top_plot, font_size)
        plot_betas(ax_b, S, beta_x, beta_y, font_size, label)
    # new_plot_elems(ax_el, lat, s_point = S[0], legend = legend, y_scale=0.8)  # plot elements
    plot_elems(fig, ax_el, lat, s_point=S[0], legend=legend, y_scale=0.8, font_size=font_size,
               excld_legend=excld_legend)
    return fig, ax_top, ax_b, ax_el


def plot_betas(ax, S, beta_x, beta_y, font_size, label=''):
    """
    From OCELOT
    :param ax:
    :param S:
    :param beta_x:
    :param beta_y:
    :param font_size:
    :return:
    """
    ax.set_ylabel(r"$\beta_{x,y}$ [m]", fontsize=font_size)
    ax.plot(S, beta_x, 'b', lw=2, label=r"$\beta_{x}$" + f" {label}")
    ax.plot(S, beta_y, 'r--', lw=2, label=r"$\beta_{y}$" + f" {label}")
    ax.tick_params(axis='both', labelsize=font_size)
    leg = ax.legend(loc='upper left', shadow=False, fancybox=True, prop=font_manager.FontProperties(size=font_size))
    leg.get_frame().set_alpha(0.2)


def plot_alphas(ax, S, x, y, font_size):
    # ax.set_ylabel(r"$\beta_{x,y}$ [m]", fontsize=font_size)
    ax.plot(S, x, 'b', lw=2, label=r"$\alpha_{x}$")
    ax.plot(S, y, 'r--', lw=2, label=r"$\alpha_{y}$")
    ax.tick_params(axis='both', labelsize=font_size)
    leg = ax.legend(loc='upper left', shadow=False, fancybox=True, prop=font_manager.FontProperties(size=font_size))
    leg.get_frame().set_alpha(0.2)


def plot_disp(ax, tws, top_plot, font_size):
    S = [p.s for p in tws]  # map(lambda p:p.s, tws)
    d_Ftop = []
    Fmin = []
    Fmax = []
    for elem in top_plot:
        # print(elem, tws.__dict__[elem] )
        Ftop = [p.__dict__[elem] for p in tws]
        # for f in Ftop:
        #    print(f)
        # print (max(Ftop))
        Fmin.append(min(Ftop))
        Fmax.append(max(Ftop))
        greek = ""
        if "beta" in elem or "alpha" in elem or "mu" in elem:
            greek = "\\"
        if "mu" in elem:
            elem = elem.replace("mu", "mu_")
        top_label = r"$" + greek + elem + "$"
        ax.plot(S, Ftop, lw=2, label=top_label)
        d_Ftop.append(max(Ftop) - min(Ftop))
    d_F = max(d_Ftop)
    if d_F == 0:
        d_Dx = 1
        ax.set_ylim((min(Fmin) - d_Dx * 0.1, max(Fmax) + d_Dx * 0.1))
    if top_plot[0] == "E":
        top_ylabel = r"$" + "/".join(top_plot) + "$" + ", [GeV]"
    elif top_plot[0] in ["mux", 'muy']:
        top_ylabel = r"$" + "/".join(top_plot) + "$" + ", [rad]"
    else:
        top_ylabel = r"$" + "/".join(top_plot) + "$" + ", [m]"

    yticks = ax.get_yticks()
    yticks = yticks[2::2]
    ax.set_yticks(yticks)
    # for i, label in enumerate(ax.get_yticklabels()):
    #    if i == 0 or i == 1:
    #        label.set_visible(False)
    ax.set_ylabel(top_ylabel, fontsize=font_size)
    ax.tick_params(axis='both', labelsize=font_size)
    # ax.plot(S, Dx,'black', lw = 2, label=lable)
    leg2 = ax.legend(loc='upper right', shadow=False, fancybox=True, prop=font_manager.FontProperties(size=font_size))
    leg2.get_frame().set_alpha(0.2)


def plot_API(lat, fig=None, legend=False, fig_name=1, grid=True, font_size=12):
    """
    Function creates a picture with lattice on the bottom part of the picture and top part of the picture can be
    plot arbitrary lines.
    :param lat: MagneticLattice
    :param legend: True, description of the elements, if False it is switched legend off
    :return: fig, ax
    """
    if not fig:
        fig = plt.figure(fig_name)
    plt.rc('axes', grid=grid)
    plt.rc('grid', color='0.75', linestyle='-', linewidth=0.5)
    left, width = 0.1, 0.85
    rect2 = [left, 0.19, width, 0.69]
    rect3 = [left, 0.07, width, 0.12]

    # rect1 = [left, 0.65, width, 0.3]
    # rect2 = [left, 0.19, width, 0.46]
    # rect3 = [left, 0.07, width, 0.12]

    ax_xy = fig.add_axes(rect2)  # left, bottom, width, height
    ax_el = fig.add_axes(rect3, sharex=ax_xy)

    for ax in ax_xy, ax_el:
        if ax != ax_el:
            for label in ax.get_xticklabels():
                label.set_visible(False)

    ax_xy.grid(grid)
    ax_el.set_yticks([])
    ax_el.grid(grid)
    # plt.xlim(S[0], S[-1])

    ax_xy.tick_params(axis='both', labelsize=font_size)
    # leg = ax_xy.legend(loc='upper left', shadow=False, fancybox=True, prop=font_manager.FontProperties(size=font_size))
    # leg.get_frame().set_alpha(0.2)

    fig.subplots_adjust(hspace=0)

    # plot_xy(ax_xy, S, X, Y, font_size)

    # plot_elems(ax_el, lat, nturns = 1, legend = True) # plot elements
    # new_plot_elems(fig, ax_el, lat, nturns = 1, legend = legend)
    plot_elems(fig, ax_el, lat, legend=legend, y_scale=0.8, font_size=font_size)

    return fig, ax_xy


def combinations_recursive(n):
    accum = []

    def combinations_recursive_inner(n, buf, gaps, sum, accum):
        if gaps == 0:
            accum.append(list(buf))
        else:
            for xx in range(0, n + 1):
                if sum + xx + (gaps - 1) * n < n:
                    continue
                if sum + xx > n:
                    break
                combinations_recursive_inner(n, buf + [xx], gaps - 1, sum + xx, accum)

    combinations_recursive_inner(n, [], 2, 0, accum)
    return pd.DataFrame(accum).values


def plot_resonance_lines(ax, order=4):
    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    xybuffer = []
    for r in range(1, order + 1):
        # print('>>>>>',r)
        xylist = []
        combs = combinations_recursive(r)
        x = np.array([0, 1])
        for c in combs:
            if c[0] != 0 and c[1] != 0:
                for i in np.arange(-7, 7):
                    y = (i - c[0] * x) / c[1]
                    xylist.append((x, y))
                    # print(r,x,y,11)
                    y = (i + c[0] * x) / c[1]
                    xylist.append((x, y))
                    # print(r,x,y,12)
            elif c[0] == 0:
                for i in range(-7, 7):
                    y = np.ones_like(x) * i / c[1]
                    xylist.append((x, y))
                    y = np.ones_like(x) * i / -c[1]
                    xylist.append((x, y))
                    # print(r,x,y,2)
            elif c[1] == 0:
                for i in range(-7, 7):
                    xv = i / c[0]
                    xylist.append((np.array([xv, xv]), np.array([0, 1])))
                    xylist.append((np.array([-xv, -xv]), np.array([0, 1])))
                    # print(x,y,3)
        xyfinal = [(x, y) for (x, y) in xylist if
                   not (x[0] > 1 and x[1] > 1) and not (y[0] > 1 and y[1] > 1) and not (x[0] < 0 and x[1] < 0) and not (
                           y[0] < 0 and y[1] < 0)]
        # xyfinal = xylist
        for (x, y) in xyfinal:
            if (tuple(x), tuple(y)) not in xybuffer:
                ax.plot(x, y, color=colors[r - 1], lw=1)
                xybuffer.append((tuple(x), tuple(y)))
                # print()
            else:
                continue
                # print('bla')
            # print(x,y)


def plot_fft(ax, freq, power, grid=True, font_size=12):
    plt.rc('axes', grid=grid)
    plt.rc('grid', color='0.75', linestyle='-', linewidth=0.5)
    ax.grid(grid)
    # ax.set_yticks([])
    ax.tick_params(axis='both', labelsize=font_size)
    h = ax.plot(freq, power, lw=0.5)
    return ax, h


def plot_fft_semilog(ax, freq, power, grid=True, font_size=12):
    plt.rc('axes', grid=grid)
    plt.rc('grid', color='0.75', linestyle='-', linewidth=0.5)
    ax.grid(grid)
    # ax.set_yticks([])
    ax.tick_params(axis='both', labelsize=font_size)
    ax.semilogy(freq, power, lw=0.5)
    return ax


# @formatter:off
dict_plot = {Quadrupole:    {"scale": 0.7, "color": "r",            "edgecolor": "r",          "label": "quad"},
             Sextupole:     {"scale": 0.5, "color": "g",            "edgecolor": "g",          "label": "sext"},
             Octupole:      {"scale": 0.5, "color": "g",            "edgecolor": "g",          "label": "sext"},
             Cavity:        {"scale": 0.7, "color": "orange",       "edgecolor": "lightgreen", "label": "cav"},
             Bend:          {"scale": 0.7, "color": "lightskyblue", "edgecolor": "k",          "label": "bend"},
             RBend:         {"scale": 0.7, "color": "lightskyblue", "edgecolor": "k",          "label": "bend"},
             SBend:         {"scale": 0.7, "color": "lightskyblue", "edgecolor": "k",          "label": "bend"},
             Matrix:        {"scale": 0.7, "color": "pink",         "edgecolor": "k",          "label": "mat"},
             Multipole:     {"scale": 0.7, "color": "g",            "edgecolor": "k",          "label": "mult"},
             Undulator:     {"scale": 0.7, "color": "pink",         "edgecolor": "k",          "label": "und"},
             Monitor:       {"scale": 0.5, "color": "orange",       "edgecolor": "orange",     "label": "mon"},
             Hcor:          {"scale": 0.7, "color": "c",            "edgecolor": "c",          "label": "cor"},
             Vcor:          {"scale": 0.7, "color": "c",            "edgecolor": "c",          "label": "cor"},
             Drift:         {"scale": 0.,  "color": "k",            "edgecolor": "k",          "label": ""},
             Marker:        {"scale": 0.,  "color": "k",            "edgecolor": "k",          "label": "mark"},
             Edge:          {"scale": 0.,  "color": "k",            "edgecolor": "k",          "label": ""},
             Solenoid:      {"scale": 0.7, "color": "g",            "edgecolor": "g",          "label": "sol"},
             UnknownElement:{"scale": 0.7, "color": "g",            "edgecolor": "g",          "label": "unk"},
             }

dict_plot.update({ILMatrix:     {"scale": 0.7, "color": "pink",         "edgecolor": "k",          "label": "mat"}})
# @formatter:on


# def new_plot_elems(fig, ax, lat, s_point=0, nturns=1, y_lim=None, y_scale=1, legend=True):
#     dict_copy = copy.deepcopy(dict_plot)
#     alpha = 1
#     ax.set_ylim((-1, 1.5))
#     if y_lim != None:
#         ax.set_ylim(y_lim)
#     points_with_annotation = []
#     L = 0.
#     q = []
#     b = []
#     c = []
#     s = []
#     u = []
#     rf = []
#     m = []
#     for elem in lat.sequence:
#         if elem.__class__ == Quadrupole:
#             q.append(elem.k1)
#         elif elem.__class__ in [Bend, RBend, SBend]:
#             b.append(elem.angle)
#         elif elem.__class__ in [Hcor, Vcor]:
#             c.append(elem.angle)
#         elif elem.__class__ == Sextupole:
#             s.append(elem.k2)
#         elif elem.__class__ == Undulator:
#             u.append(elem.Kx + elem.Ky)
#         elif elem.__class__ == Cavity:
#             rf.append(elem.v)
#         elif elem.__class__ == Multipole:
#             m.append(sum(np.abs(elem.kn)))
#     q_max = np.max(np.abs(q)) if len(q) != 0 else 0
#     b_max = np.max(np.abs(b)) if len(b) != 0 else 0
#     s_max = np.max(np.abs(s)) if len(s) != 0 else 0
#     c_max = np.max(np.abs(c)) if len(c) != 0 else 0
#     u_max = np.max(np.abs(u)) if len(u) != 0 else 0
#     rf_max = np.max(np.abs(rf)) if len(rf) != 0 else 0
#     m_max = np.max(m) if len(m) != 0 else 0
#     ncols = np.sign(len(q)) + np.sign(len(b)) + np.sign(len(s)) + np.sign(len(c)) + np.sign(len(u)) + np.sign(
#         len(rf)) + np.sign(len(m))
#
#     labels_dict = {}
#     for elem in dict_copy.keys():
#         labels_dict[elem] = dict_copy[elem]["label"]
#     for elem in lat.sequence:
#         if elem.__class__ in [Marker, Edge]:
#             L += elem.l
#             continue
#         l = elem.l
#         if l == 0:
#             l = 0.03
#         # type = elem.type
#         scale = dict_copy[elem.__class__]["scale"]
#         color = dict_copy[elem.__class__]["color"]
#         label = dict_copy[elem.__class__]["label"]
#         ecolor = dict_copy[elem.__class__]["edgecolor"]
#         ampl = 1
#         s_coord = np.array(
#             [L + elem.l / 2. - l / 2., L + elem.l / 2. - l / 2., L + elem.l / 2. + l / 2., L + elem.l / 2. + l / 2.,
#              L + elem.l / 2. - l / 2.]) + s_point
#         if elem.__class__ == Quadrupole:
#             ampl = elem.k1 / q_max if q_max != 0 else 1
#             point, = ax.fill(s_coord, (np.array([-1, 1, 1, -1, -1]) + 1) * ampl * scale * y_scale, color,
#                              edgecolor=ecolor,
#                              alpha=alpha, label=dict_copy[elem.__class__]["label"])
#             dict_copy[elem.__class__]["label"] = ""
#
#         elif elem.__class__ in [Bend, RBend, SBend]:
#             ampl = elem.angle / b_max if b_max != 0 else 1
#             point, = ax.fill(s_coord, (np.array([-1, 1, 1, -1, -1]) + 1) * ampl * scale * y_scale, color,
#                              alpha=alpha, label=dict_copy[elem.__class__]["label"])
#             dict_copy[elem.__class__]["label"] = ""
#
#         elif elem.__class__ in [Hcor, Vcor]:
#
#             ampl = elem.angle / c_max if c_max != 0 else 0.5
#             # print c_max, elem.angle, ampl
#             if elem.angle == 0:
#                 ampl = 0.5
#                 point, = ax.fill(s_coord, (np.array([-1, 1, 1, -1, -1])) * ampl * scale * y_scale, "lightcyan",
#                                  edgecolor="k",
#                                  alpha=0.5, label=dict_copy[elem.__class__]["label"])
#             else:
#                 point, = ax.fill(s_coord, (np.array([-1, 1, 1, -1, -1]) + 1) * ampl * scale * y_scale, color,
#                                  edgecolor=ecolor,
#                                  alpha=alpha, label=dict_copy[elem.__class__]["label"])
#             dict_copy[Hcor]["label"] = ""
#             dict_copy[Vcor]["label"] = ""
#
#         elif elem.__class__ == Sextupole:
#             ampl = (elem.k2) / s_max if s_max != 0 else 1
#             point, = ax.fill(s_coord, (np.array([-1, 1, 1, -1, -1]) + 1) * ampl * scale * y_scale, color,
#                              alpha=alpha, label=dict_copy[elem.__class__]["label"])
#             dict_copy[elem.__class__]["label"] = ""
#
#         elif elem.__class__ == Cavity:
#             ampl = 1  # elem.v/rf_max if rf_max != 0 else 0.5
#             point, = ax.fill(s_coord, np.array([-1, 1, 1, -1, -1]) * ampl * scale * y_scale, color,
#                              alpha=alpha, edgecolor="lightgreen", label=dict_copy[elem.__class__]["label"])
#             dict_copy[elem.__class__]["label"] = ""
#
#         elif elem.__class__ == Undulator:
#             ampl = elem.Kx / u_max if u_max != 0 else 0.5
#             point, = ax.fill(s_coord, np.array([-1, 1, 1, -1, -1]) * ampl * scale * y_scale, color,
#                              alpha=alpha, label=dict_copy[elem.__class__]["label"])
#             dict_copy[elem.__class__]["label"] = ""
#
#         elif elem.__class__ == Multipole:
#             ampl = sum(elem.kn) / m_max if u_max != 0 else 0.5
#             point, = ax.fill(s_coord, np.array([-1, 1, 1, -1, -1]) * ampl * scale * y_scale, color,
#                              alpha=alpha, label=dict_copy[elem.__class__]["label"])
#             dict_copy[elem.__class__]["label"] = ""
#
#         else:
#             point, = ax.fill(s_coord, np.array([-1, 1, 1, -1, -1]) * ampl * scale * y_scale, color, edgecolor=ecolor,
#                              alpha=alpha)
#         annotation = ax.annotate(elem.__class__.__name__ + ": " + elem.id,
#                                  xy=(L + l / 2., 0),  # xycoords='data',
#                                  # xytext=(i + 1, i), textcoords='data',
#                                  horizontalalignment="left",
#                                  arrowprops=dict(arrowstyle="simple", connectionstyle="arc3,rad=+0.2"),
#                                  bbox=dict(boxstyle="round", facecolor="w", edgecolor="0.5", alpha=0.9),
#                                  fontsize=16
#                                  )
#         # by default, disable the annotation visibility
#         annotation.set_visible(False)
#         L += elem.l
#         points_with_annotation.append([point, annotation])

def plot_elems(fig, ax, lat, s_point=0, nturns=1, y_lim=None, y_scale=1, legend=True, font_size=18, excld_legend=None):
    legend_font_size = font_size

    if excld_legend is None:
        excld_legend = []

    dict_copy = copy.deepcopy(dict_plot)
    alpha = 1
    ax.set_ylim((-1, 1.5))
    ax.tick_params(axis='both', labelsize=font_size)
    if y_lim != None:
        ax.set_ylim(y_lim)
    points_with_annotation = []
    L = 0.
    q = []
    b = []
    c = []
    s = []
    u = []
    rf = []
    m = []
    sol = []
    for elem in lat.sequence:
        if elem.__class__ == Quadrupole:
            q.append(elem.k1)
        elif elem.__class__ in [Bend, RBend, SBend]:
            b.append(elem.angle)
        elif elem.__class__ in [Hcor, Vcor]:
            c.append(elem.angle)
        elif elem.__class__ == Sextupole:
            s.append(elem.k2)
        elif elem.__class__ == Solenoid:
            sol.append(elem.k)
        elif elem.__class__ == Undulator:
            u.append(elem.Kx + elem.Ky)
        elif elem.__class__ in [Cavity, TWCavity, TDCavity]:
            rf.append(elem.v)
        elif elem.__class__ == Multipole:
            m.append(sum(np.abs(elem.kn)))

    q_max = np.max(np.abs(q)) if len(q) != 0 else 0
    b_max = np.max(np.abs(b)) if len(b) != 0 else 0
    s_max = np.max(np.abs(s)) if len(s) != 0 else 0
    c_max = np.max(np.abs(c)) if len(c) != 0 else 0
    u_max = np.max(np.abs(u)) if len(u) != 0 else 0
    sol_max = np.max(np.abs(sol)) if len(sol) != 0 else 0
    rf_max = np.max(np.abs(rf)) if len(rf) != 0 else 0
    m_max = np.max(m) if len(m) != 0 else 0
    ncols = np.sign(len(q)) + np.sign(len(b)) + np.sign(len(s)) + np.sign(len(c)) + np.sign(len(u)) + np.sign(
        len(rf)) + np.sign(len(m))

    labels_dict = {}
    for elem in dict_copy.keys():
        labels_dict[elem] = dict_copy[elem]["label"]
    for elem in lat.sequence:
        if elem.__class__ in excld_legend:
            L += elem.l
            continue

        if elem.__class__ in [Marker, Edge]:
            L += elem.l
            continue
        l = elem.l
        if l == 0:
            l = 0.03
        # type = elem.type
        if elem.__class__ in dict_copy:
            scale = dict_copy[elem.__class__]["scale"]
            color = dict_copy[elem.__class__]["color"]
            label = dict_copy[elem.__class__]["label"]
            ecolor = dict_copy[elem.__class__]["edgecolor"]
        else:
            scale = dict_copy[UnknownElement]["scale"]
            color = dict_copy[UnknownElement]["color"]
            label = dict_copy[UnknownElement]["label"]
            ecolor = dict_copy[UnknownElement]["edgecolor"]
        ampl = 1
        s_coord = np.array(
            [L + elem.l / 2. - l / 2., L + elem.l / 2. - l / 2., L + elem.l / 2. + l / 2., L + elem.l / 2. + l / 2.,
             L + elem.l / 2. - l / 2.]) + s_point

        rect = np.array([-1, 1, 1, -1, -1])

        if elem.__class__ == Quadrupole:
            ampl = elem.k1 / q_max if q_max != 0 else 1
            point, = ax.fill(s_coord, (rect + 1) * ampl * scale * y_scale, color, edgecolor=ecolor,
                             alpha=alpha, label=dict_copy[elem.__class__]["label"])
            dict_copy[elem.__class__]["label"] = ""

        elif elem.__class__ == Solenoid:
            ampl = elem.k / sol_max if sol_max != 0 else 1
            point, = ax.fill(s_coord, (rect + 1) * ampl * scale * y_scale, color, edgecolor=ecolor,
                             alpha=alpha, label=dict_copy[elem.__class__]["label"])
            dict_copy[elem.__class__]["label"] = ""

        elif elem.__class__ in [Bend, RBend, SBend]:
            ampl = elem.angle / b_max if b_max != 0 else 1
            point, = ax.fill(s_coord, (rect + 1) * ampl * scale * y_scale, color,
                             alpha=alpha, label=dict_copy[elem.__class__]["label"])
            dict_copy[elem.__class__]["label"] = ""

        elif elem.__class__ in [Hcor, Vcor]:
            ampl = elem.angle / c_max if c_max != 0 else 0.5
            if elem.angle == 0:
                ampl = 0.5
                point, = ax.fill(s_coord, rect * ampl * scale * y_scale, "lightcyan", edgecolor="k",
                                 alpha=0.5, label=dict_copy[elem.__class__]["label"])
            else:
                point, = ax.fill(s_coord, (rect + 1) * ampl * scale * y_scale, color, edgecolor=ecolor,
                                 alpha=alpha, label=dict_copy[elem.__class__]["label"])
            dict_copy[Hcor]["label"] = ""
            dict_copy[Vcor]["label"] = ""

        elif elem.__class__ == Sextupole:
            ampl = elem.k2 / s_max if s_max != 0 else 1
            point, = ax.fill(s_coord, (rect + 1) * ampl * scale * y_scale, color,
                             alpha=alpha, label=dict_copy[elem.__class__]["label"])
            dict_copy[elem.__class__]["label"] = ""

        elif elem.__class__ in [Cavity, TWCavity, TDCavity]:
            ampl = 1
            point, = ax.fill(s_coord, rect * ampl * scale * y_scale, color,
                             alpha=alpha, edgecolor="lightgreen", label=dict_copy[elem.__class__]["label"])
            dict_copy[elem.__class__]["label"] = ""

        elif elem.__class__ == Undulator:
            ampl = elem.Kx / u_max if u_max != 0 else 0.5
            point, = ax.fill(s_coord, rect * ampl * scale * y_scale, color,
                             alpha=alpha, label=dict_copy[elem.__class__]["label"])
            dict_copy[elem.__class__]["label"] = ""

        elif elem.__class__ == Multipole:
            ampl = sum(elem.kn) / m_max if u_max != 0 else 0.5
            point, = ax.fill(s_coord, rect * ampl * scale * y_scale, color,
                             alpha=alpha, label=dict_copy[elem.__class__]["label"])
            dict_copy[elem.__class__]["label"] = ""

        else:
            point, = ax.fill(s_coord, rect * ampl * scale * y_scale, color, edgecolor=ecolor,
                             alpha=alpha)

        annotation = ax.annotate(elem.__class__.__name__ + ": " + elem.id,
                                 xy=(L + l / 2., 0),  # xycoords='data',
                                 # xytext=(i + 1, i), textcoords='data',
                                 horizontalalignment="left",
                                 arrowprops=dict(arrowstyle="simple", connectionstyle="arc3,rad=+0.2"),
                                 bbox=dict(boxstyle="round", facecolor="w", edgecolor="0.5", alpha=0.9),
                                 fontsize=legend_font_size
                                 )
        # by default, disable the annotation visibility
        annotation.set_visible(False)
        L += elem.l
        points_with_annotation.append([point, annotation])
        ax.set_xlabel("s [m]", fontsize=font_size)

    def on_move(event):
        visibility_changed = False
        for point, annotation in points_with_annotation:
            should_be_visible = (point.contains(event)[0] == True)
            if should_be_visible != annotation.get_visible():
                visibility_changed = True
                annotation.set_visible(should_be_visible)

        if visibility_changed:
            plt.draw()

    on_move_id = fig.canvas.mpl_connect('motion_notify_event', on_move)
    if legend:
        ax.legend(loc='upper center', ncol=ncols, shadow=False, prop=font_manager.FontProperties(size=legend_font_size))
