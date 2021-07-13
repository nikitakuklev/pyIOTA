import copy
from typing import Tuple

import numpy as np
import pandas as pd
from . import LatticeContainer
from ocelot import SecondOrderMult, lattice_transfer_map, Particle
from .tracking import track_nturns_with_bpms, track_nturns, track_nturns_store_particles
import scipy.optimize


def get_closed_orbit(box: LatticeContainer,
                     backend: str = 'ocelot',
                     lattice_options_extra: dict = None,
                     task_options: dict = None,
                     **kwargs) -> (pd.DataFrame, Tuple):
    """
    Read out closed orbit
    Available backends:
    - OCELOT
    - Elegant

    For elegant:
     See https://ops.aps.anl.gov/manuals/elegant_latest/elegantsu25.html#x33-320007.15
     Results should be correct in 6D, since full tracking is used
     Output has coordinates at s = s_end

    For OCELOT:
     Uses either matrices (no feed-down from misalignments!!!) or tracking
     Initial conditions use exact solution of p = M*p + B
     Results with tracking should be accurate in 4D, but not 6D

    :param backend:
    :param box:
    :return: class Particle
    """
    if backend == 'elegant':
        import pyiota.sim as sim
        import pyiota.elegant as elegant
        import pyiota.util.config as cfg

        # Create elegant task
        dc = sim.DaskClient()
        lattice_options = {'sr': 0, 'isr': 0, 'dip_kicks': 64, 'quad_kicks': 32, 'sext_kicks': 16, 'oct_kicks': 16}
        if lattice_options_extra:
            lattice_options.update(lattice_options_extra)
        task_options = task_options or {}
        params = {'label': 'auto_closed_orbit'}
        et = elegant.routines.standard_sim_job(work_folder=cfg.DASK_DEFAULT_WORK_FOLDER,
                                               lattice_options=lattice_options,
                                               add_random_id=True,
                                               task_options=task_options,
                                               **params)
        t = elegant.Task(relative_mode=True, run_folder=et.run_subfolder, lattice_path=et.lattice_file_abs_path)
        elegant.template_task_closed_orbit(box, t, **task_options)
        et.task_file_contents = t.compile()
        et.lattice_file_contents = box.to_elegant(lattice_options=lattice_options, silent=True)

        futures = dc.submit_to_elegant([et], dry_run=False, pure=False)
        future = futures[0]
        (data, etaskresp) = future.result(30)
        assert etaskresp.state == sim.STATE.ENDED
        if data.returncode != 0:
            print(etaskresp)
            print(data)
            raise Exception

        futures = dc.read_out([etaskresp], dry_run=False)
        future = futures[0]
        (data2, etaskresp2) = future.result(30)
        assert etaskresp2.state == sim.STATE.ENDED_READ

        clo = data2['clo']

        df_ocelot = box.df()
        df_elegant = clo.df()
        df_elegant = df_elegant.drop(0).reset_index(drop=True)
        df_ocelot = df_ocelot[df_ocelot.loc[:, 'class'] != 'Edge'].reset_index(drop=True)
        if not len(df_ocelot) <= len(df_elegant) <= len(df_ocelot) + 2:
            raise Exception(f'Orbit missing elements: {len(df_ocelot)} vs {len(df_elegant)}') # apertures
        #assert np.all(df_ocelot['id'].str.upper() == df_elegant['ElementName'].str.upper())
        #assert np.all(np.isclose(df_ocelot['s_end'], df_elegant['s']))
        assert np.isclose(df_ocelot['s_end'].iloc[-1], df_elegant['s'].iloc[-1])
        df_merged = df_ocelot.join(df_elegant)

        p = Particle(x=df_merged.x[0], px=df_merged.xp[0], y=df_merged.y[0], py=df_merged.yp[0])
        return df_merged, (p, None, {'sim_data': data, 'read_data': data2, 'sim_task': etaskresp, 'read_task': etaskresp2, 'job': et}),

    elif backend == 'ocelot':
        options = {'algorithm': 'tracking'}
        options.update(kwargs)

        energy = box.pc
        lattice = box.lattice

        R = lattice_transfer_map(lattice, energy)
        smult = SecondOrderMult()
        # This is (I-R)^-1
        ME = np.eye(4) - R[:4, :4]
        X_init = np.dot(np.linalg.inv(ME), lattice.B[:4])

        def track_and_average(p: Particle, n_turns: int):
            cp = copy.copy(p)
            averages = np.zeros((n_turns, 4))
            for turn in range(n_turns):
                track_nturns(box, cp, 1)
                averages[turn, :] = np.array([cp.x, cp.px, cp.y, cp.py])
            avg_x, avg_px, avg_y, avg_py = np.mean(averages, axis=0)[:]
            return avg_x, avg_px, avg_y, avg_py

        def iterate_tracking(x_init):
            def f(x_arr):
                p = Particle(x=x_arr[0], px=x_arr[1], y=x_arr[2], py=x_arr[3])  # , tau=X[4], p=X[5]
                p = track_nturns(box, p, 1)
                # X = np.array([[p.x], [p.px], [p.y], [p.py], [p.tau], [p.p]])
                X = np.array([p.x, p.px, p.y, p.py])
                X += lattice.B[:4, 0]
                err = np.sum((1e6 * (X - x_arr)) ** 2)
                return err

            bounds = [(x_init[0] - 5.e-3, x_init[0] + 5.e-3),
                      (x_init[1] - 5.e-3, x_init[1] + 5.e-3),
                      (x_init[2] - 5.e-3, x_init[2] + 5.e-3),
                      (x_init[3] - 5.e-3, x_init[3] + 5.e-3)]
            eps = np.array([1.e-6, 1.e-7, 1.e-6, 1.e-7])
            initial_simplex = np.zeros((5, 4))
            initial_simplex[0, :] = x_init
            initial_simplex[1, :] = x_init + np.array([1.e-6, 0, 0, 0])
            initial_simplex[2, :] = x_init + np.array([0, 1.e-6, 0, 0])
            initial_simplex[3, :] = x_init + np.array([0, 0, 1.e-6, 0])
            initial_simplex[4, :] = x_init + np.array([0, 0, 0, 1.e-6])
            r = scipy.optimize.minimize(f, x_init,
                                        tol=1.e-10,
                                        method='Nelder-Mead',
                                        # method='BFGS',
                                        # bounds=bounds,
                                        options={'xatol': 1.e-7,
                                                 'maxiter': 1000,
                                                 'disp': True,
                                                 'initial_simplex': initial_simplex
                                                 },
                                        # options={'maxiter': 1000,
                                        #          'disp': True,
                                        #          # 'eps': eps
                                        #          }
                                        )
            return r

        if options['algorithm'] == 'tracking':
            # Because of poor initial guess, this doesn't converge well :(
            # Really need that feed-down in matrices
            X_init = X_init.T[0]
            res = iterate_tracking(X_init)
            p = Particle(x=res.x[0], px=res.x[1], y=res.x[2], py=res.x[3])
        elif options['algorithm'] == 'averaging':
            X_init = X_init.T[0]
            p = Particle(x=X_init[0], px=X_init[1], y=X_init[2], py=X_init[3])
            x, px, y, py = track_and_average(p, 500)
            p = Particle(x=x, px=px, y=y, py=py)
            res = None
        elif options['algorithm'] == 'averaging_and_tracking':
            X_init = X_init.T[0]
            p = Particle(x=X_init[0], px=X_init[1], y=X_init[2], py=X_init[3])
            n_turns = options.get('ocelot_tracking_turns', 40)
            x, px, y, py = track_and_average(p, n_turns)
            res = iterate_tracking(np.array([x, px, y, py]))
            p = Particle(x=res.x[0], px=res.x[1], y=res.x[2], py=res.x[3])
        elif options['algorithm'] == 'iterate':
            # OCELOT does not implement feed-down in matrices, so this is only valid with dx/dy = 0
            def errf(x):
                X = np.zeros((6, 1))
                X[:4, 0] = x
                smult.numpy_apply(X, R, lattice.T)
                X += lattice.B
                err = np.sum(1000 * (X[:4, 0] - x) ** 2)
                return err

            res = scipy.optimize.fmin(errf, X_init, xtol=1e-7, maxiter=2e3, maxfun=2e3)
            p = Particle(x=res[0], px=res[1], y=res[2], py=res[3])
        else:
            raise Exception(f'Unknown algorithm {options["algorithm"]}')

        p_list = track_nturns_store_particles(box, copy.copy(p), 1)

        df_ocelot = box.df()
        df_ocelot['x'] = [p.x for p in p_list[1:]]
        df_ocelot['y'] = [p.y for p in p_list[1:]]
        df_ocelot['px'] = [p.px for p in p_list[1:]]
        df_ocelot['py'] = [p.py for p in p_list[1:]]
        df_ocelot = df_ocelot[df_ocelot.loc[:, 'class'] != 'Edge'].reset_index(drop=True)

        return df_ocelot, (p, res)


def get_closed_orbit_at_bpms(box, backend='ocelot'):
    if backend == 'ocelot':
        df, (p, res) = get_closed_orbit(box)
        return track_nturns_with_bpms(box, p, 1, store_at_start=False)
    else:
        # TODO: parse df for BPM locations
        raise AttributeError(f'Backend {backend} not supported yet')
