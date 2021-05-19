import subprocess
from enum import Enum
from shutil import which
from typing import List, Optional, Dict
from pathlib import PurePath
import pandas as pd


def slurm_available():
    return which('srun') is not None


def execute(command, *args, **kwargs):
    # subprocess.run(['elegant',str(fpath)], capture_output=True, text=True)
    return subprocess.run([command, *args], **kwargs)


class Sim:
    def __init__(self, eid=None, ):
        self.id = eid


class DaskSLURMSim(Sim):
    def __init__(self, eid: str = None, fallback: bool = True, **kwargs):
        """
        Create SLURM cluster
        :param eid: Name for reference
        :param fallback: Whether to create a local cluster if SLURM is not found
        """
        import dask
        import dask.distributed

        super().__init__(eid)
        if slurm_available():
            import dask_jobqueue
            cluster_opts = {'name': 'dask-worker',
                            'cores': 1,
                            'memory': '2GB',
                            'processes': 1,
                            # 'interface': 'ib0',
                            'shebang': '#!/usr/bin/env bash',
                            'queue': 'broadwl',
                            'walltime': '36:00:00',
                            # 'job-cpu': 1,
                            # 'job-mem': '2GB',
                            # 'log-directory': "~/scratch/slurm_logs/",
                            }
            cluster_opts.update(kwargs)
            cluster = dask_jobqueue.SLURMCluster(**cluster_opts)
        else:
            if fallback:
                cluster = dask.distributed.LocalCluster()
            else:
                raise Exception('SLURM is not available on this node*-')
        dask.config.set({'distributed.admin.tick.interval': 1000, 'distributed.worker.profile.interval': 1000})

        # print(cluster.job_script())
        print(f'Dask cluster started at ({cluster.dashboard_link})')
        client = dask.distributed.Client(cluster)
        self.client = client
        self.cluster = cluster
        self.adaptive = None
        self.d = dask.distributed  # mega hacks here we go

    def get_endpoints(self):
        return self.client, self.cluster

    def submit(self, tasks: List, fnf: bool = False, limit: int = 100, no_scaling: bool = False):
        """
        Executes the tasks (functions) on the cluster
        """
        if not no_scaling:
            wi = min(limit, len(tasks))
            print(f'Cluster {self.id} - scaling to 1')
            self.cluster.scale(1)
            self.client.wait_for_workers(1, timeout=60)
        print(f'Cluster {self.id} - starting jobs')
        futures = []
        for f in tasks:
            future = self.client.submit(*f)
            if fnf:
                self.d.fire_and_forget(future)
            else:
                futures.append(future)
        if fnf:
            del future
        if not no_scaling:
            print(f'Cluster {self.id} - scaling to {wi} workers')
            for i in range(min(5, wi), wi + 1, 10):
                self.cluster.scale(i)
                self.client.wait_for_workers(i, timeout=60)
                print(i)
            self.cluster.scale(wi)
            self.client.wait_for_workers(wi)

            ad = self.cluster.adapt(minimum=0, maximum=wi, target_duration=3)
            self.adaptive = ad
        return futures or None


class DaskClient:
    """ Wrapper for dask client used to submit and read out tasks """

    def __init__(self, address: str = None, autorestart: bool = False):
        from distributed import Client
        from ..util import config as cfg
        address = address or cfg.DASK_SCHEDULER_ADDRESS
        if address in cfg.CLIENT_CACHE:
            client = cfg.CLIENT_CACHE[address]
        else:
            client = Client(address, timeout=2)
            cfg.CLIENT_CACHE[address] = client
        self.client = client
        self.address = address
        self.restart = autorestart

    def self_test(self):
        if self.restart:
            self.client.restart()
        args = (42, 2)
        future = self.client.submit(run_dummy_task, *args)
        r = future.result(timeout=2)
        assert r == run_dummy_task(*args)

    def submit_to_elegant(self, tasks: List, fnf: bool = False, dry_run: bool = True, pure: bool = True):
        import dask.distributed
        # assert all(isinstance(t, ElegantSimJob) for t in tasks)
        # assert all(t.__class__ == ElegantSimJob.__class__ for t in tasks) # to help autoreload

        fun = run_elegant_job
        futures = []
        for t in tasks:
            future = self.client.submit(fun, t, dry_run, pure=pure)
            if fnf:
                dask.distributed.fire_and_forget(future)
            else:
                futures.append(future)
        if fnf:
            del future
        return futures or None

    def read_out(self, tasks: List, dry_run: bool = True, pure: bool = True):
        assert all(isinstance(t, ElegantSimJob) for t in tasks)

        fun = run_elegant_sdds_import
        futures = []
        for t in tasks:
            future = self.client.submit(fun, t, dry_run, pure=pure)
            futures.append(future)
        return futures or None

    def reset(self):
        self.client.restart()

    @property
    def tasks(self):
        def get_all_tasks(dask_scheduler=None):
            return str(dask_scheduler.tasks)

        stream = self.client.run_on_scheduler(get_all_tasks)
        return stream


class STATE(Enum):
    NOT_STARTED = 0
    PREP = 1
    STARTED = 2
    ENDED = 10
    ENDED_READ = 20
    ERROR = 99


class ElegantSimJob:
    """
    Object for storing all job file paths and metadata - gets passed around by dask many times
    """

    def __init__(self,
                 label: str,
                 lattice_options: Dict,
                 work_folder: PurePath,
                 data_folder: Optional[PurePath],
                 run_subfolder: PurePath,
                 task_file_name: PurePath,
                 lattice_file_name: PurePath,
                 parameter_file_map: Dict[PurePath, pd.DataFrame] = None,
                 task_file_contents: str = None,
                 lattice_file_contents: str = None):
        self.label = label
        assert work_folder == data_folder or data_folder is None
        assert work_folder.is_absolute()
        self.work_folder = work_folder
        self.data_folder = data_folder
        self.run_subfolder = run_subfolder
        self.run_folder = work_folder / run_subfolder
        self.task_file_name = task_file_name
        self.task_file_contents = task_file_contents
        self.lattice_file_name = lattice_file_name
        self.lattice_file_contents = lattice_file_contents
        # This is a map of relative file path to dataframe
        if parameter_file_map:
            self.parameter_file_map = {}
            for (k, v) in parameter_file_map.items():
                assert isinstance(k, PurePath)
                assert isinstance(v, pd.DataFrame)
                self.parameter_file_map[self.run_subfolder / k] = v
        else:
            self.parameter_file_map = None
        self.lattice_options = lattice_options
        self.task_file_abs_path = self.run_subfolder / task_file_name
        self.lattice_file_abs_path = self.run_subfolder / lattice_file_name
        self.state = STATE.NOT_STARTED

    def __str__(self):
        return (f'ElegantSimJob - status ({self.state.name})\n' +
                f'Label: {self.label}\n' +
                f'Work folder: {self.work_folder}\n' +
                f'Run folder: {self.run_folder}\n' +
                f'Lattice file name: {self.lattice_file_name}\n' +
                f'Lattice file path: {self.lattice_file_abs_path}\n' +
                f'Task file name: {self.task_file_name}\n' +
                f'Task file path: {self.task_file_abs_path}\n' +
                f'Task contents OK: {self.task_file_contents is not None}\n' +
                f'Lattice contents OK: {self.lattice_file_contents is not None}')


def run_dummy_task(*args, **kwargs):
    if len(args) >= 2:
        return args[0] * args[1]
    else:
        return 42


def run_elegant_job(task: ElegantSimJob, dry_run: bool = True):
    """ Dask elegant worker function """
    from time import perf_counter
    start = perf_counter()

    def delta():
        return perf_counter() - start

    import subprocess, os, logging
    l = logging.getLogger("distributed.worker")
    from pathlib import Path
    l.info('--------------------------')
    l.info(f'{delta():.3f} Elegant sim task starting')

    task.state = STATE.PREP
    # raise ValueError('bla')
    assert os.name == 'posix'
    work_folder = Path(task.work_folder)
    task_file_name = Path(task.task_file_name)
    run_folder = Path(task.run_folder)
    task_file_abs_path = Path(task.task_file_abs_path)
    lattice_file_abs_path = Path(task.lattice_file_abs_path)

    l.info(f'>Label: {task.label}')
    l.info(f'>Work folder: {work_folder}')
    l.info(f'>Run folder: {run_folder}')
    l.info(f'>Task file name: {task_file_name}')
    l.info(f'>Task file path: {task_file_abs_path}')
    l.info(f'>Task lattice path: {lattice_file_abs_path}')
    l.info(f'>CWD: {os.getcwd()}')

    assert work_folder.is_absolute()
    assert work_folder.is_dir()
    assert work_folder.exists()
    if run_folder.exists():
        raise Exception(f'Run folder {run_folder} exists!')
    assert not task_file_abs_path.exists()
    if task.parameter_file_map is not None:
        for (k, v) in task.parameter_file_map.items():
            if not isinstance(k, PurePath):
                raise AttributeError(f'Parameter key ({k}) is not a path but ({k.__class__.__name__})')
            if not isinstance(v, pd.DataFrame):
                raise AttributeError(f'Parameter file ({k}) is not a dataframe but ({v.__class__.__name__})')
            k2 = Path(k)
            assert not k2.exists()

    l.info(f'>{delta():.3f} Writing elegant files')
    os.chdir(work_folder)
    if not dry_run:
        run_folder.mkdir()
        task_file_abs_path.write_text(task.task_file_contents)
        lattice_file_abs_path.write_text(task.lattice_file_contents)
        if task.parameter_file_map is not None:
            from ..elegant.io import SDDS
            for (k, v) in task.parameter_file_map.items():
                k2 = Path(k)
                assert not k2.exists()
                l.info(f'>{delta():.3f} Writing ({k2})')
                sdds = SDDS(k2, blank=True)
                sdds.set(v)
                sdds.write(k2)
    else:
        if task.parameter_file_map is not None:
            from ..elegant.io import SDDS
            for (k, v) in task.parameter_file_map.items():
                l.info(f'>{delta():.3f} FAKE Writing ({k})')
                sdds = SDDS(k, blank=True)
                sdds.set(v)

    l.info(f'>{delta():.3f} Calling elegant')
    task.state = STATE.STARTED
    if dry_run:
        result = subprocess.run(['echo', str(42)], capture_output=True, text=True)
    else:
        result = subprocess.run(['elegant', str(task_file_abs_path)], capture_output=True, text=True)
    task.state = STATE.ENDED
    l.info(f'{delta():.3f} Finished')
    l.info('--------------------------')
    return result, task


def run_elegant_sdds_import(task, dry_run: bool = True):
    # Clear pickle cache
    # from distributed.worker import cache_loads
    # cache_loads.data.clear()
    # import importlib
    # importlib.reload(pyIOTA.sim.runtime.)

    from time import perf_counter
    start = perf_counter()

    def delta():
        return perf_counter() - start

    import logging
    l = logging.getLogger("distributed.worker")
    l.info('--------------------------')
    l.info(f'{delta():.3f} Elegant SDDS read starting')

    from pathlib import Path
    from ..elegant.io import SDDS, SDDSTrack

    run_folder = Path(task.run_folder)
    l.info(f'{delta():.3f} Run folder: {run_folder}')

    extensions_to_import = ['twi', 'clo', 'cen', 'fin', 'track', 'sdds']
    data = {}
    for ext in extensions_to_import:
        l.info(f'>{delta():.3f} Checking ({ext})')
        files = list(run_folder.glob('*.' + ext))
        # .sdds is not expected, so raise error
        if ext == 'sdds' and len(files) > 0:
            raise Exception('.sdds file found?')
        elif ext == 'track':
            if len(files) > 0:
                if dry_run:
                    data[ext] = ['42']
                else:
                    tracks = []
                    for f in files:
                        try:
                            sdds = SDDSTrack(f, fast=True, as_df=True, clear_sd=True, clear_cdata=True)
                            sdds.prepare_for_serialization()
                        except Exception as e:
                            l.error(f'>{delta():.3f} Error parsing ({f}), dumping as SDDS:')
                            sdds = SDDS(files[0], fast=True)
                            l.error(sdds.summary())
                            raise e
                        tracks.append(sdds)
                    data[ext] = tracks
            else:
                l.info(f'>{delta():.3f} Missing ({ext})')
        else:
            if len(files) > 1:
                raise ValueError(f'Too many files matched - {[str(f) for f in files]}')
            elif len(files) == 0:
                l.info(f'>{delta():.3f} Missing ({ext})')
                continue
            else:
                l.info(f'>{delta():.3f} Importing ({ext})')
                if dry_run:
                    data[ext] = '42'
                else:
                    sdds = SDDS(files[0], fast=True)
                    sdds.prepare_for_serialization()
                    data[ext] = sdds

    task.state = STATE.ENDED_READ
    l.info(f'{delta():.3f} Finished')
    l.info('--------------------------')
    return data, task
