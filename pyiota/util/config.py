import pathlib

DASK_SCHEDULER_PORT = 10421
DASK_SCHEDULER_ADDRESS = f'tcp://localhost:{DASK_SCHEDULER_PORT}'

DASK_DEFAULT_WORK_FOLDER = pathlib.PurePosixPath('/home/nkuklev/studies/dask_tasks/')

ELEGANT_MPI_MODULE = 'openmpi/1.10+intel-16.0' #mpich
ELEGANT_MPI_ARGS = '--map-by core:nooversubscribe'


CLIENT_CACHE = {}
