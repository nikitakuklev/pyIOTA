import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Tuple
from matplotlib import font_manager
from ocelot.gui.accelerator import new_plot_elems


def plot_simple_grid(*args,
                     nperrow: int = 11,
                     sizes: Tuple = (2, 2),
                     demean: bool = True,
                     paired_bpm_mode: bool = False):
    """
    Simple routine to plot a grid of data sharing x/y axes, with 1 plot per each dictionary key
    :param demean:
    :param paired_bpm_mode:
    :param nperrow:
    :param sizes:
    :return:
    """
    for data_dict in args:
        if isinstance(data_dict, list):
            data_dict = {k: v for d in data_dict for k, v in d.items()}
        if not isinstance(data_dict, dict):
            raise Exception(f'Supplied argument is not a dict: {data_dict.__class__}')
        if len(data_dict) == 0:
            print('Skipping empty plot')
            continue
        n_rows = len(data_dict) // nperrow + 1
        fig, ax = plt.subplots(n_rows, nperrow, figsize=(sizes[0] * nperrow, sizes[1] * n_rows),
                               sharex=True, sharey=True)
        if paired_bpm_mode:
            keys = [k for k in data_dict.keys()]
            roots = [k[:-2] for k in keys]
            seen = set()
            roots = [x for x in roots if not (x in seen or seen.add(x))]
            if not nperrow >= len(roots):
                raise Exception(f'Not enough columns ({nperrow}) to fit all pairs ({len(roots)})')
            for j, root in enumerate(roots):
                i = 0
                for z, (k, v) in enumerate(data_dict.items()):
                    if root in k:
                        if demean:
                            v = v - np.mean(v)
                        ax[i, j].plot(v, lw=0.5)
                        ax[i, j].set_title(f"{z}|{k}")
                        i += 1
        else:
            ax = ax.flatten()
            for i, (k, v) in enumerate(data_dict.items()):
                if demean:
                    v = v - np.mean(v)
                ax[i].plot(v)
                ax[i].set_title(f"{i}|{k}")
    return fig, ax


def plot_opt_func(fig, lat, tws, top_plot=["Dx"], legend=False, fig_name=None, grid=True, font_size=12,
                  excld_legend=None):
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

    plt.rc('axes', grid=grid)
    plt.rc('grid', color='0.75', linestyle='-', linewidth=0.5)
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
    beta_x = [p.beta_x for p in tws]
    beta_y = [p.beta_y for p in tws]
    S = [p.s for p in tws]

    plt.xlim(S[0], S[-1])

    plot_disp(ax_top, tws, top_plot, font_size)

    plot_betas(ax_b, S, beta_x, beta_y, font_size)
    # plot_elems(ax_el, lat, s_point = S[0], legend = legend, y_scale=0.8) # plot elements
    new_plot_elems(fig, ax_el, lat, s_point=S[0], legend=legend, y_scale=0.8, font_size=font_size,
                   excld_legend=excld_legend)


def plot_betas(ax, S, beta_x, beta_y, font_size):
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
    ax.plot(S, beta_x, 'b', lw=2, label=r"$\beta_{x}$")
    ax.plot(S, beta_y, 'r--', lw=2, label=r"$\beta_{y}$")
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


def plot_API(lat, fig=None, legend=True, fig_name=1, grid=True, font_size=12):
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
    new_plot_elems(fig, ax_el, lat, legend=legend, y_scale=0.8, font_size=font_size)

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
    ax.plot(freq, power, lw=0.5)
    return ax


def plot_fft_semilog(ax, freq, power, grid=True, font_size=12):
    plt.rc('axes', grid=grid)
    plt.rc('grid', color='0.75', linestyle='-', linewidth=0.5)
    ax.grid(grid)
    # ax.set_yticks([])
    ax.tick_params(axis='both', labelsize=font_size)
    ax.semilogy(freq, power, lw=0.5)
    return ax
