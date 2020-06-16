__all__ = ['load_data_tbt']

import uuid
from pathlib import Path
import h5py
import pandas as pd
import numpy as np
import datetime

special_keys = ['idx', 'kickv', 'kickh', 'state', 'custom']


def load_data_tbt(fpath: Path,
                  verbose: bool = True,
                  version: int = 2,
                  soft_fail: bool = None,
                  force_load: bool = None,
                  idx: int = None):
    """
    Loads data from experimental HDF5 TBT format
    :param fpath: Full directory or file path
    :param verbose: Print various info
    :param version: Which format version to load. Roughly, v1 was used start-middle of run 2, v2 for rest of run 2.
    :param soft_fail: Whether to suppress exception is data inconsistency is found
    :param force_load: Whether to load data in any case (implied soft_fail=True)
    :param idx: index of tbt kick - overrides that of data
    :return:
    """
    if fpath.is_file():
        files = [fpath]
        if verbose: print(f'Loading single file: {fpath}')
    else:
        assert fpath.is_dir()
        files = list(fpath.glob('*.hdf5'))
        if verbose: print(f'Loading {len(files)} files from {fpath} ({files[0]})')
    rowlist = []

    if soft_fail is None:
        soft_fail = True if version == 1 else False
    if force_load is None:
        force_load = False

    if version == 1:
        for i, file in enumerate(files):
            with h5py.File(str(file), 'r', libver='latest') as h5f:
                kick_arrays = {}
                for (k, v) in h5f.items():
                    if isinstance(v, h5py.Dataset):
                        kick_arrays[k] = v[:]
                vlist = [v[0] for (k, v) in h5f.items() if isinstance(v, h5py.Dataset)]
                if len(set(vlist)) != 1:
                    if soft_fail:
                        print(f'File {fpath.name} has corrupted data from multiple kicks ({set(vlist)})')
                        if not force_load: return None
                    else:
                        raise Exception(
                            f"Kick acquisition numbers ({set(vlist)}) are not the same - data is corrupted!")
                rowlist.append({'idx': idx or i,
                                'kickv': h5f['state'].attrs.get('kickv', np.nan),
                                'kickh': h5f['state'].attrs.get('kickh', np.nan),
                                'state': dict(h5f['state'].attrs),
                                # 'ts': h5f.attrs['time_utcstamp'],
                                **kick_arrays})
                # df.loc[i] = [i, h5f['state'].attrs['kickv'], h5f['state'].attrs['kickh'], kick_arrays]
        df = pd.DataFrame(data=rowlist)
        if verbose: print(f'Read in {len(df)} files with {len(kick_arrays)} BPMs')
        return df
    if version == 2:
        for i, file in enumerate(files):
            with h5py.File(str(file), 'r', libver='latest') as h5f:
                kick_arrays = {}
                vlist = []
                for (k, v) in h5f.items():
                    if isinstance(v, h5py.Dataset):
                        kick_arrays[k] = v[:]
                        vlist.append(v[0])
                if len(set(vlist)) != 1:
                    if soft_fail:
                        print(f'File {fpath.name} has corrupted data from multiple kicks ({set(vlist)})')
                        if not force_load: return None
                    else:
                        raise Exception(f"Kick acquisition numbers ({set(vlist)})are not the same - data is corrupted!")
                rowlist.append({'idx': idx or i,
                                'kickv': h5f['state'].attrs.get('kickv', np.nan),
                                'kickh': h5f['state'].attrs.get('kickh', np.nan),
                                'state': dict(h5f['state'].attrs),
                                'custom': dict(h5f['custom'].attrs),
                                # 'ts': h5f.attrs['time_utcstamp'],
                                **kick_arrays})
                # df.loc[i] = [i, h5f['state'].attrs['kickv'], h5f['state'].attrs['kickh'], kick_arrays]

        df = pd.DataFrame(data=rowlist)
        if verbose: print(f'Read in {len(df)} files with {len(kick_arrays)} BPMs')
        return df
    else:
        raise Exception("Incorrect file version specified")


def save_data_tbt(fpath: Path, df: pd.DataFrame, name_format: str = "iota_kicks_%Y%m%d-%H%M%S.hdf5",
                  verbose: bool = True, version: int = 1):
    if version == 1:
        if len(df) != 1:
            raise Exception('Cannot save multiple kicks')
        assert not fpath.exists() or fpath.is_dir()
        if not fpath.exists():
            print(f'Save directory {fpath} missing - creating')
            fpath.mkdir(parents=True)
        fname = datetime.datetime.now().strftime(name_format);
        fnamefull = fpath / fname
        if verbose: print(f'Full save path: {fnamefull}')
        if fnamefull.exists():
            print(f'Warning - path {fnamefull} exists, aborting')
            raise Exception("File already exists")
        df_dict = dict(df.iloc[0])
        with h5py.File(str(fnamefull), 'w', libver='latest') as f:
            for (k, v) in df_dict.items():
                if k not in special_keys:
                    f.create_dataset(k, data=v, compression='gzip', compression_opts=9, shuffle=True, fletcher32=True)

            stategr = f.create_group('state')
            for (k, v) in df_dict['state'].items():
                # print(k,v)
                stategr.attrs[k] = v
            stategr = f.create_group('custom')
            for (k, v) in df_dict['custom'].items():
                # print(k,v)
                stategr.attrs[k] = v

            f.attrs['time_utcstamp'] = datetime.datetime.utcnow().timestamp();
            # f.attrs['time_acnet'] = aqtime
            # f.attrs['time_run'] = time_run.strftime("%Y%m%d-%H%M%S")
            # f.attrs['units'] = units
            f.attrs['datatype'] = 'TBT_R2V1'
            f.attrs['uuid'] = str(uuid.uuid4())
    else:
        raise Exception("Incorrect file version specified")
