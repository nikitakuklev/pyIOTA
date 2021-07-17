import itertools

import numpy as np
import pyiota.math as pmath


def get_distances2(combinations, a, n, offsets):
    """

    :param combinations:
    :param a: origin of reference line
    :param n: direction of line
    :param offsets:
    :return:
    """
    distances = []
    distances_o = []
    is_b = []
    for nux, nuy in combinations:
        p = np.array([nux, nuy])
        d_line = pmath.point_line_distance(a, n, p)
        #dx = np.sqrt((p[0] - a[0])**2 + (p[0] - a[0])**2)
        #dx = (p[0] - a[0])/-n[1]
        ofs = offsets[0]
        #abot = np.array([a[0] - n[1] * ofs, a[0] + n[0] * ofs])
        #ybot = (abot + n * (dx + n[1] * ofs))[1]
        abot = np.array([a[0] - n[1] * ofs, a[1] + n[0] * ofs])
        dx = (p[0]-abot[0])/n[0]
        ybot = (abot + n * dx)[1]
        #print(f'Dbot: {dx=:.5f}, {n[0]=:.5f}, {n[1]=:.5f}, {ofs=}, {(abot + n * dx)=}')
        ofs = offsets[2]
        atop = np.array([a[0] - n[1] * ofs, a[1] + n[0] * ofs])
        dx = (p[0] - atop[0]) / n[0]
        ytop = (atop + n * dx)[1]
        #print(dx, n[1], ofs, dx, (abot + n * dx))
        #print(f'Dtop: {dx=:.5f}, {n[0]=:.5f}, {n[1]=:.5f}, {ofs=}, {(atop + n * dx)=}')

        if ytop < ybot:
            temp = ybot
            ybot = ytop
            ytop = temp

        d_origin = np.linalg.norm(a - p)
        is_behind_ref = False  # is_behind_ref = (a-p)@n > 0#2e-3

        if nuy > ytop or nuy < ybot:
            print(f'Point {p[0]:.5f}/{p[1]:.5f} outside bounds of ({ybot:.5f}-{ytop:.5f})')
            distances.append(np.nan)
            is_b.append(False)
            distances_o.append(np.nan)
        else:
            if is_behind_ref:
                # print(f'Point {nux:.3f} {nuy:.3f} behind line {n} start - {d_line:.4f} vs {d_origin:.4f}')
                distances.append(d_origin)
            else:
                # assert d_line < d_origin
                if d_line >= d_origin:
                    print(d_line, d_origin, nux, nuy)
                    raise Exception
                distances.append(d_line)
            is_b.append(is_behind_ref)
            distances_o.append(d_origin)
    if np.all(np.isnan(distances)):
        print('DIST CHECK: ALL POINTS REJECTED')
        return np.nan, combinations, distances
    if any([not b and d < 5e-4 and not np.isnan(d) for d, b in zip(distances, is_b)]):
        mask = [not b for b in is_b]
        fwd_masked_do = [do if m and dl < 5e-4 else -np.inf for dl, do, m in zip(distances, distances_o, mask)]
        print(f'Points {[c for c, m in zip(combinations, mask)]} - dorigin {fwd_masked_do}')
        minidx = np.nanargmax(fwd_masked_do)
    else:
        minidx = np.nanargmin(distances)
    return minidx, combinations, distances


def filter_tunes(tunes_l, bmin, bmax):
    # Filter all tunes in list
    tl2 = []
    for t in tunes_l:
        t2 = t[np.logical_and(t>bmin,t<bmax)]
        t2 = t2[t2<bmax]
        if len(t2) == 0:
            t2 = np.array([np.nan])
        tl2.append(t2.astype(np.float64))
        #print(t,t2)
    return tl2


from sklearn.utils.extmath import cartesian
def find_closest_tunes(tunes_l):
    """ Returns sorted distances of various tune combinations """
    # mesh = np.meshgrid(*tunes_filtered)
    # np.stack(mesh, -1).reshape(-1, 18).shape
    N = np.product(np.array([len(l) for l in tunes_l]))
    print(f'Checking {np.product(np.array([len(l) for l in tunes_l]))} tune combinations')
    if N > 1e8: raise Exception
    combinations = cartesian(tunes_l) # numpy way of itertools.product(*tunes_l)
    assert len(combinations) == N
    distances = np.nanstd(combinations, axis=1) # RMS distance of points from mean, roughly how spread out cluster is
    cd = [(d, c) for d, c in zip(distances,combinations)]
    cd = sorted(cd, key=lambda x: x[0])
    return cd


def reject_outliers(data, m = 2.):
    if len(data) <= 2:
        return data
    d = np.abs(data - np.nanmedian(data))
    mdev = np.nanmedian(d)
    s = d/mdev if mdev else 0.
    data[(s>m) & (d>5e-4)] = np.nan
    return data


def select_tunes_cluster_pairs(k, bounds, families, model=1, **kwargs):
    family = families[0]
    bd = bounds[0]
    bpms = k.get_bpms(family)
    tunes_l = [np.array(k.peaks[b][0]) for b in bpms]
    tunes_filtered = filter_tunes(tunes_l, bd[0], bd[1])
    if np.all([np.all(np.isnan(l)) for l in tunes_filtered]):
        print(tunes_l, tunes_filtered, bd)
    print(f'Fam {family}: {tunes_l}')
    print(f'Fam {family}F: {tunes_filtered}')
    resultsH = find_closest_tunes(tunes_filtered)

    family = families[1]
    bd = bounds[1]
    bpms = k.get_bpms(family)
    tunes_l = [np.array(k.peaks[b][0]) for b in bpms]
    tunes_filtered = filter_tunes(tunes_l, bd[0], bd[1])
    if np.all([np.all(np.isnan(l)) for l in tunes_filtered]):
        print(tunes_l, tunes_filtered, bd)
    print(f'{family}: {tunes_l}')
    print(f'{family}F: {tunes_filtered}')
    resultsV = find_closest_tunes(tunes_filtered)

    clustersH, clustersV = resultsH[:model], resultsV[:model]
    print(f'Have {len(clustersH)}/{len(resultsH)} H and {len(clustersV)}/{len(resultsV)} V clusters')

    meansH, meansV = [np.nanmean(c[1]) for c in clustersH], [np.nanmean(c[1]) for c in clustersV]
    combinations = list(itertools.product(meansH, meansV))
    combinations_idx = list(itertools.product(range(len(meansH)), range(len(meansV))))
    nldict = kwargs['nldict']
    n = nldict[kwargs['line']]
    minidx, combinations, distances = get_distances2(combinations, kwargs['a'], n, kwargs['offsets'])

    if np.isnan(minidx):
        clusterH = np.ones(len(clustersH[0][1])) * np.nan
        clusterV = np.ones(len(clustersH[0][1])) * np.nan
    else:
        clusterH = resultsH[combinations_idx[minidx][0]][1]
        clusterV = resultsV[combinations_idx[minidx][1]][1]
        clusterH = reject_outliers(clusterH, m=4)
        clusterV = reject_outliers(clusterV, m=4)

    if kwargs.get('debug', False):
        print(f'Tune means: {(meansH, meansV)}')
        for i, ((c1, c2), d) in enumerate(zip(combinations, distances)):
            print(f'{i} | ({c1:.3f},{c2:.3f}) = {d:.4f}')
        print(f'#{minidx} selected')
        print()

    for family, cluster in zip(families, (clusterH, clusterV)):
        for b, v in zip(k.get_bpms(family), cluster):
            k.df[b + k.Datatype.NU.value + '2'] = v