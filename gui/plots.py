import matplotlib.pyplot as plt
from matplotlib import font_manager
from ocelot.gui.accelerator import new_plot_elems


def plot_opt_func(fig, lat, tws, top_plot=["Dx"], legend=True, fig_name=None, grid=True, font_size=12, excld_legend=None):
    """
    Modified from OCELOT

    function for plotting: lattice (bottom section), vertical and horizontal beta-functions (middle section),
    other parameters (top section) such as "Dx", "Dy", "E", "mux", "muy", "alpha_x", "alpha_y", "gamma_x", "gamma_y"
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
    ax_b = fig.add_axes(rect2, sharex=ax_top)  #left, bottom, width, height
    ax_el = fig.add_axes(rect3, sharex=ax_top)
    for ax in ax_b, ax_el, ax_top:
        if ax!=ax_el:
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

    plot_disp(ax_top,tws, top_plot, font_size)

    plot_betas(ax_b, S, beta_x, beta_y, font_size)
    #plot_elems(ax_el, lat, s_point = S[0], legend = legend, y_scale=0.8) # plot elements
    new_plot_elems(fig, ax_el, lat, s_point=S[0], legend=legend, y_scale=0.8, font_size=font_size, excld_legend=excld_legend)


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
    ax.plot(S, beta_y, 'r', lw=2, label=r"$\beta_{y}$")
    ax.tick_params(axis='both', labelsize=font_size)
    leg = ax.legend(loc='upper left', shadow=False, fancybox=True, prop=font_manager.FontProperties(size=font_size))
    leg.get_frame().set_alpha(0.2)


def plot_disp(ax, tws, top_plot, font_size):
    S = [p.s for p in tws]#map(lambda p:p.s, tws)
    d_Ftop = []
    Fmin = []
    Fmax = []
    for elem in top_plot:
        #print(elem, tws.__dict__[elem] )
        Ftop = [p.__dict__[elem] for p in tws]
        #for f in Ftop:
        #    print(f)
        #print (max(Ftop))
        Fmin.append(min(Ftop))
        Fmax.append(max(Ftop))
        greek = ""
        if "beta" in elem or "alpha" in elem or "mu" in elem:
            greek = "\\"
        if "mu" in elem:
            elem = elem.replace("mu", "mu_")
        top_label = r"$" + greek + elem+"$"
        ax.plot(S, Ftop, lw = 2, label=top_label)
        d_Ftop.append( max(Ftop) - min(Ftop))
    d_F = max(d_Ftop)
    if d_F == 0:
        d_Dx = 1
        ax.set_ylim(( min(Fmin)-d_Dx*0.1, max(Fmax)+d_Dx*0.1))
    if top_plot[0] == "E":
        top_ylabel = r"$"+"/".join(top_plot) +"$"+ ", [GeV]"
    elif top_plot[0] in ["mux", 'muy']:
        top_ylabel = r"$" + "/".join(top_plot) + "$" + ", [rad]"
    else:
        top_ylabel = r"$"+"/".join(top_plot) +"$"+ ", [m]"

    yticks = ax.get_yticks()
    yticks = yticks[2::2]
    ax.set_yticks(yticks)
    #for i, label in enumerate(ax.get_yticklabels()):
    #    if i == 0 or i == 1:
    #        label.set_visible(False)
    ax.set_ylabel(top_ylabel, fontsize=font_size)
    ax.tick_params(axis='both', labelsize=font_size)
    #ax.plot(S, Dx,'black', lw = 2, label=lable)
    leg2 = ax.legend(loc='upper right', shadow=False, fancybox=True, prop=font_manager.FontProperties(size=font_size))
    leg2.get_frame().set_alpha(0.2)