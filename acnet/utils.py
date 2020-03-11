import uuid
from pathlib import Path
import h5py
import pandas as pd
import datetime

special_keys = ['idx', 'kickv', 'kickh', 'state']


def load_data_tbt(fpath: Path, verbose: bool = True, version: int = 1, skip_corrupted=True):
    if version == 1:
        if fpath.is_file():
            files = [fpath]
            if verbose: print(f'Loading single file: {fpath}')
        else:
            assert fpath.is_dir()
            files = list(fpath.glob('*.hdf5'))
            if verbose: print(f'Loading {len(files)} files from {fpath} ({files[0]})')
        rowlist = []
        for i, file in enumerate(files):
            with h5py.File(str(file), 'r', libver='latest') as h5f:
                kick_arrays = {}
                for (k, v) in h5f.items():
                    if isinstance(v, h5py.Dataset):
                        kick_arrays[k] = v[()]
                rowlist.append({'idx': i,
                                'kickv': h5f['state'].attrs['kickv'],
                                'kickh': h5f['state'].attrs['kickh'],
                                'state': dict(h5f['state'].attrs),
                                'ts': h5f.attrs['time_utcstamp'],
                                **kick_arrays})
                # df.loc[i] = [i, h5f['state'].attrs['kickv'], h5f['state'].attrs['kickh'], kick_arrays]
                vlist = [v[0] for v in kick_arrays.values()]
                if len(set(vlist)) != 1:
                    if skip_corrupted:
                        print(f'File {fpath.name} has corrupted data from multiple kicks ({set(vlist)}) - skipping')
                        return None
                    else:
                        raise Exception(f"Kick acquisition numbers ({set(vlist)})are not the same - data is corrupted!")
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
                #print(k,v)
                stategr.attrs[k] = v

            f.attrs['time_utcstamp'] = datetime.datetime.utcnow().timestamp();
            # f.attrs['time_acnet'] = aqtime
            # f.attrs['time_run'] = time_run.strftime("%Y%m%d-%H%M%S")
            # f.attrs['units'] = units
            f.attrs['datatype'] = 'TBT_R2V1'
            f.attrs['uuid'] = str(uuid.uuid4())
    else:
        raise Exception("Incorrect file version specified")
