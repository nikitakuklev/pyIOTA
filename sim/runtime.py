import subprocess
from shutil import which
from typing import List, Callable


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
                            'local-directory': None,
                            'shebang': '#!/usr/bin/env bash',
                            'queue': 'broadwl',
                            'walltime': '36:00:00',
                            # 'job-cpu': 1,
                            # 'job-mem': '2GB',
                            'log-directory': "/home/nkuklev/scratch/slurm_logs/",
                            }
            cluster_opts.update(kwargs)
            cluster = dask_jobqueue.SLURMCluster(interface='ib0', **cluster_opts)
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
