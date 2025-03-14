import subprocess
import sys
import traceback
from enum import Enum
from shutil import which
from typing import List, Optional, Dict, Union
from pathlib import PurePath, Path
import pandas as pd
import lzma


def slurm_available():
    return which('srun') is not None


def execute(command, *args, **kwargs):
    # subprocess.run(['elegant',str(fpath)], capture_output=True, text=True)
    return subprocess.run([command, *args], **kwargs)


class Sim:
    def __init__(self, eid=None, ):
        self.id = eid


class Utilities:
    def get_env(*args, **kwargs):
        import os
        return os.environ


class BinaryFileWrapper:
    def __init__(self, path: Path, compression: str = 'lzma'):
        assert path.is_file()
        self.compression = compression
        with open(path, 'rb') as f:
            buffer = f.read()
        self.uncompressed_size = len(buffer)
        if compression == 'lzma':
            cbuffer = lzma.compress(buffer, preset=9 | lzma.PRESET_EXTREME)
            del buffer
        elif compression is None:
            cbuffer = buffer
        else:
            raise Exception
        self.cbuffer = cbuffer
        self.size = len(self.cbuffer)
        del buffer

    def unpack(self):
        if self.compression is None:
            return self.cbuffer
        elif self.compression == 'lzma':
            return lzma.decompress(self.cbuffer)


class DaskSLURMSim(Sim):
    def __init__(self, eid: str = None, fallback: bool = True, **kwargs):
        """
        Create SLURM cluster
        :param eid: Name for reference
        :param fallback: Whether to create a local cluster if SLURM is not found
        """
        import dask.distributed

        super().__init__(eid)
        if slurm_available():
            from .dask import SLURMCluster
            cluster_opts = {'name': 'dask-worker',
                            'cores': 1,
                            # 'memory': '2GB',
                            'processes': 1,
                            # 'interface': 'ib0',
                            'shebang': '#!/usr/bin/env bash',
                            # 'queue': 'broadwl',
                            'walltime': '36:00:00',
                            # 'job-cpu': 1,
                            # 'job-mem': '2GB',
                            # 'log_directory': "~/scratch/slurm_logs",
                            }
            cluster_opts.update(kwargs)
            cluster = SLURMCluster(**cluster_opts)
        else:
            if fallback:
                cluster = dask.distributed.LocalCluster()
            else:
                raise Exception('SLURM is not available on this node*-')
        dask.config.set({'distributed.admin.tick.interval': 1000,
                         'distributed.worker.profile.interval': 1000
                         })

        # print(cluster.job_script())
        print(f'Dask cluster started at ({cluster.dashboard_link})')
        client = dask.distributed.Client(cluster)
        self.client = client
        self.cluster = cluster
        self.adaptive = None
        self.d = dask.distributed  # mega hacks here we go

    def get_endpoints(self):
        return self.client, self.cluster

    def get_scheduler_environment(self):
        fut = self.client.run_on_scheduler(Utilities.get_env)
        return fut

    def get_environment(self):
        future = self.client.submit(Utilities.get_env)
        result = future.result()
        return result

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
        import dask
        dask.config.set({"distributed.comm.timeouts.tcp": 180})
        dask.config.set({"distributed.comm.retry.count": 5})
        dask.config.set({"distributed.comm.timeouts.connect": 180})
        from distributed import Client
        from ..util import config as cfg
        address = address or cfg.DASK_SCHEDULER_ADDRESS
        if address in cfg.CLIENT_CACHE:
            client = cfg.CLIENT_CACHE[address]
            if client.status == 'closed':
                client = Client(address, timeout=5)
                cfg.CLIENT_CACHE[address] = client
        else:
            client = Client(address, timeout=5)
            cfg.CLIENT_CACHE[address] = client
        self.client = client
        self.address = address
        self.autorestart = autorestart

    def restart(self, timeout=90):
        self.client.restart(timeout=timeout)

    def self_test(self):
        if self.autorestart:
            self.client.restart()
        args = (42, 2)
        future = self.client.submit(run_dummy_task, *args)
        r = future.result(timeout=2)
        assert r == run_dummy_task(*args)

    def submit_to_elegant(self, tasks: List, fnf: bool = False, dry_run: bool = True, mpi: int = 0,
                          pure: bool = True
                          ):
        import dask.distributed
        # assert all(isinstance(t, ElegantSimJob) for t in tasks)
        # assert all(t.__class__ == ElegantSimJob.__class__ for t in tasks) # to help autoreload
        fun = run_elegant_job
        futures = []
        for t in tasks:
            if dry_run:
                future = self.client.submit(fun, t, dry_run, mpi, pure=pure, priority=10)
            else:
                future = self.client.submit(fun, t, dry_run, mpi, pure=pure)
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
            if dry_run:
                future = self.client.submit(fun, t, dry_run, pure=pure, priority=10, retries=2)
            else:
                future = self.client.submit(fun, t, dry_run, pure=pure, priority=1, retries=2)
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


# class STATE(Enum):
#     # Currently not used due to serialization issues
#     NOT_STARTED = 0
#     PREP = 1
#     STARTED = 2
#     ENDED = 10
#     ENDED_READ = 20
#     ERROR = 99

class STATE:
    NOT_STARTED = "NOT_STARTED"
    PREP = "PREP"
    STARTED = "STARTED"
    ENDED = "ENDED"
    ENDED_READ = "ENDED_READ"
    ERROR = "ERROR"


class ElegantSimJob:
    """
    Object for storing all job file paths and metadata - gets passed around by dask many times
    """

    def __init__(self,
                 label: str,
                 lattice_options: Dict,
                 task_options: Dict,
                 work_folder: PurePath,
                 data_folder: Optional[PurePath],
                 run_subfolder: PurePath,
                 task_file_name: PurePath,
                 lattice_file_name: PurePath,
                 parameter_file_map: Dict[PurePath, pd.DataFrame] = None,
                 task_file_contents: str = None,
                 lattice_file_contents: str = None
                 ):
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
        self.parameter_file_map: Optional[Dict] = None
        if parameter_file_map:
            self.parameter_file_map = {}
            for (k, v) in parameter_file_map.items():
                assert isinstance(k, PurePath)
                assert isinstance(v, pd.DataFrame)
                self.parameter_file_map[self.run_subfolder / k] = v
        self.lattice_options = lattice_options
        self.task_options = task_options
        self.task_file_abs_path = self.run_subfolder / task_file_name
        self.lattice_file_abs_path = self.run_subfolder / lattice_file_name
        self.state: str = STATE.NOT_STARTED
        self.params: Dict = {}

    def __str__(self):
        return (f'ElegantSimJob - status ({self.state})\n' +
                f'Label:                {self.label}\n' +
                f'Work folder:          {self.work_folder}\n' +
                f'Run folder:           {self.run_folder}\n' +
                f'Lattice file name:    {self.lattice_file_name}\n' +
                f'Lattice file path:    {self.lattice_file_abs_path}\n' +
                f'Task file name:       {self.task_file_name}\n' +
                f'Task file path:       {self.task_file_abs_path}\n' +
                f'Task contents OK:     {self.task_file_contents is not None}\n' +
                f'Lattice contents OK:  {self.lattice_file_contents is not None}')


def run_dummy_task(*args, **kwargs):
    if len(args) >= 2:
        return args[0] * args[1]
    else:
        return 42


def run_generic_elegant_job(task: ElegantSimJob,
                            dry_run: bool = True,
                            mpi: int = 0,
                            mpi_lib: str = None
                            ):
    """ Dask elegant worker function """
    from time import perf_counter
    import subprocess, os, logging, datetime
    from pathlib import Path
    start = perf_counter()

    def delta():
        return perf_counter() - start

    l = logging.getLogger("distributed.worker")
    l.info('--------------------------')
    l.info(f'{delta():.3f} Elegant sim task starting')
    l.info(f'{delta():.3f} Local time: {datetime.datetime.now()}')

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
        if not getattr(task, 'file_exists_override', False):
            raise Exception(f'Run folder ({run_folder}) exists')
        else:
            l.warning(f'>Run folder ({run_folder}) exists but overriden')
    if task_file_abs_path.exists():
        if not getattr(task, 'file_exists_override', False):
            raise Exception(f'Task file ({task_file_abs_path}) exists')
        else:
            l.warning(f'>Task file ({task_file_abs_path}) exists but overriden')
    if task.parameter_file_map is not None:
        for (k, v) in task.parameter_file_map.items():
            if not isinstance(k, PurePath):
                raise AttributeError(
                    f'Parameter key ({k}) is not a path but ({k.__class__.__name__})')
            if not isinstance(v, pd.DataFrame):
                raise AttributeError(
                    f'Parameter file ({k}) is not a dataframe but ({v.__class__.__name__})')
            k2 = Path(k)
            assert not k2.exists()

    l.info(f'>{delta():.3f} Writing elegant files')
    os.chdir(work_folder)
    if not dry_run:
        run_folder.mkdir(exist_ok=getattr(task, 'file_exists_override', False))
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
                sdds.sd.mode = sdds.sd.SDDS_BINARY
                sdds.write(k2)
    else:
        if task.parameter_file_map is not None:
            from ..elegant.io import SDDS
            for (k, v) in task.parameter_file_map.items():
                l.info(f'>{delta():.3f} FAKE Writing ({k})')
                sdds = SDDS(k, blank=True)
                sdds.set(v)

    task.state = STATE.STARTED
    if dry_run:
        l.info(f'>{delta():.3f} Calling elegant (dry run)')
        result = subprocess.run(['echo', str(42)], capture_output=True, text=True)
    else:
        from ..util.config import ELEGANT_RPN_PATH
        if ELEGANT_RPN_PATH != '':
            if 'RPN_DEFNS' not in os.environ:
                os.environ['RPN_DEFNS'] = ELEGANT_RPN_PATH

        if mpi is not None and mpi > 0:
            from ..util.config import ELEGANT_MPI_MODULE, ELEGANT_MPI_ARGS
            mpi_lib = ELEGANT_MPI_MODULE
            mpi_extra = ELEGANT_MPI_ARGS
            assert mpi_lib is not None
            cmd = f'module load {mpi_lib}; mpiexec -n {mpi} {mpi_extra} Pelegant {str(task_file_abs_path)}'
            l.info(f'>{delta():.3f} Calling: {cmd}')
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        else:
            l.info(f'>{delta():.3f} Calling: elegant ({str(task_file_abs_path)})')
            result = subprocess.run(['elegant', str(task_file_abs_path)], capture_output=True,
                                    text=True)
    task.state = STATE.ENDED
    # task.sim_result = result
    l.info(f'{delta():.3f} Finished')
    l.info('--------------------------')
    return result, task


def run_elegant_job(task: ElegantSimJob, dry_run: bool = True, mpi: int = 0, mpi_lib: str = None):
    """ Dask elegant worker function """
    from time import perf_counter
    import subprocess, os, logging, datetime
    from pathlib import Path
    start = perf_counter()

    def delta():
        return perf_counter() - start

    l = logging.getLogger("distributed.worker")
    l.info('--------------------------')
    l.info(f'{delta():.3f} Elegant sim task starting')
    l.info(f'{delta():.3f} Local time: {datetime.datetime.now()}')

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
        if not getattr(task, 'file_exists_override', False):
            raise Exception(f'Run folder ({run_folder}) exists')
        else:
            l.warning(f'>Run folder ({run_folder}) exists but overriden')
    if task_file_abs_path.exists():
        if not getattr(task, 'file_exists_override', False):
            raise Exception(f'Task file ({task_file_abs_path}) exists')
        else:
            l.warning(f'>Task file ({task_file_abs_path}) exists but overriden')
    if task.parameter_file_map is not None:
        for (k, v) in task.parameter_file_map.items():
            if not isinstance(k, PurePath):
                raise AttributeError(
                    f'Parameter key ({k}) is not a path but ({k.__class__.__name__})')
            if not isinstance(v, pd.DataFrame):
                raise AttributeError(
                    f'Parameter file ({k}) is not a dataframe but ({v.__class__.__name__})')
            k2 = Path(k)
            assert not k2.exists()

    l.info(f'>{delta():.3f} Writing elegant files')
    os.chdir(work_folder)
    if not dry_run:
        run_folder.mkdir(exist_ok=getattr(task, 'file_exists_override', False))
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
                sdds.sd.mode = sdds.sd.SDDS_BINARY
                sdds.write(k2)
    else:
        if task.parameter_file_map is not None:
            from ..elegant.io import SDDS
            for (k, v) in task.parameter_file_map.items():
                l.info(f'>{delta():.3f} FAKE Writing ({k})')
                sdds = SDDS(k, blank=True)
                sdds.set(v)

    task.state = STATE.STARTED
    if dry_run:
        l.info(f'>{delta():.3f} Calling elegant (dry run)')
        result = subprocess.run(['echo', str(42)], capture_output=True, text=True)
    else:
        from ..util.config import ELEGANT_RPN_PATH
        if ELEGANT_RPN_PATH != '':
            if 'RPN_DEFNS' not in os.environ:
                os.environ['RPN_DEFNS'] = ELEGANT_RPN_PATH

        if mpi is not None and mpi > 0:
            from ..util.config import ELEGANT_MPI_MODULE, ELEGANT_MPI_ARGS
            mpi_lib = ELEGANT_MPI_MODULE
            mpi_extra = ELEGANT_MPI_ARGS
            assert mpi_lib is not None
            cmd = f'module load {mpi_lib}; mpiexec -n {mpi} {mpi_extra} Pelegant {str(task_file_abs_path)}'
            l.info(f'>{delta():.3f} Calling: {cmd}')
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        else:
            l.info(f'>{delta():.3f} Calling: elegant ({str(task_file_abs_path)})')
            result = subprocess.run(['elegant', str(task_file_abs_path)], capture_output=True,
                                    text=True)
    task.state = STATE.ENDED
    # task.sim_result = result
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
    import datetime
    start = perf_counter()

    def delta():
        return perf_counter() - start

    import logging
    l = logging.getLogger("distributed.worker")
    l.info('--------------------------')
    l.info(f'{delta():.3f} Elegant SDDS read starting')
    l.info(f'{delta():.3f} Local time: {datetime.datetime.now()}')

    from pathlib import Path
    from ..elegant.io import SDDS, SDDSTrack

    run_folder = Path(task.run_folder)
    l.info(f'{delta():.3f} Run folder: {run_folder}')

    extensions_to_import = ['twi', 'clo', 'cen', 'fin', 'track', 'ctrack',
                            'sdds', 'fma', 'mom', 'bun', 'twi2']
    data = {}
    for ext in extensions_to_import:
        l.info(f'>{delta():.3f} Checking ({ext})')
        files = list(run_folder.glob('*.' + ext))
        # .sdds is not expected, so raise error
        if ext == 'sdds' and len(files) > 0:
            raise Exception('.sdds file found?')
        elif ext == 'track':
            # Track watchpoints
            if len(files) > 0:
                if dry_run:
                    data[ext] = ['42']
                else:
                    tracks = []
                    for f in files:
                        try:
                            sdds = SDDSTrack(f, fast=True, as_df=False, clear_sd=True,
                                             clear_cdata=False)
                            sdds.prepare_for_serialization()
                        except Exception as ex:
                            l.error(
                                    f'>{delta():.3f} Error parsing ({f}), {ex=} {traceback.format_exc()=}, dumping as SDDS:')
                            sdds = SDDS(f, fast=True)
                            l.error(f'>' + sdds.summary())
                            # raise e
                        tracks.append(sdds)
                    data[ext] = tracks
            else:
                l.info(f'>{delta():.3f} Missing ({ext})')
        elif ext == 'ctrack':
            # Centroids watchpoints
            if len(files) > 0:
                if dry_run:
                    data[ext] = ['42']
                else:
                    tracks = []
                    for f in files:
                        try:
                            sdds = SDDS(f, fast=False)
                            sdds.prepare_for_serialization()
                        except Exception as e:
                            l.error(f'>{delta():.3f} Error parsing ({f})')
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
                    sdds = SDDS(files[0], fast=False)
                    sdds.prepare_for_serialization()
                    data[ext] = sdds

    task.state = STATE.ENDED_READ
    l.info(f'{delta():.3f} Finished')
    l.info('--------------------------')
    return data, task


def run_file_import(task, dry_run: bool = True):
    from time import perf_counter
    import datetime
    start = perf_counter()

    def delta():
        return perf_counter() - start

    import logging
    l = logging.getLogger("distributed.worker")
    l.info('--------------------------')
    l.info(f'{delta():.3f} Read starting')
    l.info(f'{delta():.3f} Local time: {datetime.datetime.now()}')

    from pathlib import Path

    run_folder = Path(task.run_folder)
    l.info(f'{delta():.3f} Run folder: {run_folder}')

    extensions_to_import = ['twi', 'clo', 'cen', 'fin', 'track', 'ctrack',
                            'sdds', 'fma', 'mom', 'bun', 'twi2']
    data = {}
    for ext in extensions_to_import:
        l.info(f'>{delta():.3f} Checking ({ext})')
        files = list(run_folder.glob('*.' + ext))
        # .sdds is not expected, so raise error
        if ext == 'sdds' and len(files) > 0:
            raise Exception('.sdds file found?')
        elif ext in ['track', 'ctrack']:
            # Track watchpoints
            if len(files) > 0:
                if dry_run:
                    data[ext] = ['42']
                else:
                    tracks = []
                    for f in files:
                        try:
                            w = BinaryFileWrapper(f)
                            tracks.append(w)
                        except Exception as ex:
                            l.error(
                                    f'>{delta():.3f} Error parsing ({f}), {ex=} {traceback.format_exc()=}')
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
                    w = BinaryFileWrapper(files[0])
                    data[ext] = w

    task.state = STATE.ENDED_READ
    l.info(f'{delta():.3f} Finished')
    l.info('--------------------------')
    return data, task
