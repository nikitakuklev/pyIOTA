__all__ = ['SDDS', 'read_parameters_to_df', 'write_df_to_parameter_file', 'prepare_folders', 'prepare_folder_structure']

"""
Collection of IO-related functions for elegant and SDDS file formats
"""

import os
import shutil
import sys
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd

try:
    import sdds
except ImportError:
    sys.path.insert(1, Path.home().joinpath('rpms/usr/lib/python3.6/site-packages').as_posix())
    import sdds

sd = sdds.SDDS(15)


class SDDS:
    """
    SDDSPython wrapper that provides nice dictionary-like interface, or a DataFrame view
    """

    def __init__(self, path: Path, fast: bool = False):
        if isinstance(path, Path):
            path = str(path)  # legacy code :(
        if not os.path.exists(path):
            raise Exception(f'Path {path} is missing you fool!')
        sd.load(path)
        self.path = path
        self.cname = [sd.columnName[i] for i in range(len(sd.columnName))]
        self.pname = [sd.parameterName[i] for i in range(len(sd.parameterName))]
        self.cdict = {sd.columnName[i]: i for i in range(len(sd.columnName))}
        self.pdict = {sd.parameterName[i]: i for i in range(len(sd.parameterName))}
        if not fast:
            self.cdef = [sd.columnDefinition[i].copy() for i in range(len(sd.columnName))]
            self.pdef = [sd.parameterDefinition[i].copy() for i in range(len(sd.parameterName))]
        self.cdata = [np.array(sd.columnData[i]) for i in range(len(sd.columnName))]
        self.pdata = [np.array(sd.parameterData[i]) for i in range(len(sd.parameterName))]
        if len(self.cname) > 0:
            self.pagecnt = self.cdata[0].shape[0]
        else:
            self.pagecnt = self.pdata[0].shape[0]

    def summary(self):
        """
        Print a brief file summary
        :return:
        """
        print(f'File:{self.path}')
        print(f'Pages:{self.pagecnt} | Cols:{len(self.cname)} | Pars:{len(self.pname)}')

    def __getitem__(self, name: str):
        """
        Key access method. Prioritizes columns over parameters, and returns page 0 only.
        :param name:
        :return:
        """
        if name in self.cname:
            return self.c(name)
        elif name in self.pname:
            return self.p(name)
        else:
            raise IndexError

    def c(self, name: str, p: int = None):
        """
        Get SDDS column.
        :param name:
        :param p: Page number, or all if None
        :return:
        """
        if name not in self.cname:
            raise IndexError
        if p is None:
            return self.cdata[self.cdict[name]][:, :]
        else:
            return self.cdata[self.cdict[name]][p, :]

    def p(self, name: str):
        """
        Get SDDS parameter.
        :param name:
        :return:
        """
        return self.pdata[self.pdict[name]]

    def df(self, p: int = 0):
        """
        Return columns as a DataFrame. To save on memory, it IS NOT GUARANTEED to be a deep copy.
        :param p: Page number
        :return:
        """
        d = {k: v[p, :] for (k, v) in zip(self.cname, self.cdata)}
        return pd.DataFrame(data=d)


def read_parameters_to_df(knob: Path,
                          columns: Optional[List] = None,
                          p: int = 0,
                          enforce_unique_index: bool = True) -> pd.DataFrame:
    """
    Helper method to read elegant parameter file to a dataframe.
    :param knob: file path
    :param columns: columns to read, or the default 4 if None
    :param p: page number
    :param enforce_unique_index: checks if all indices are unique, except for markers
    :return: pandas dataframe with all column data
    """
    assert knob.exists() and knob.is_file()
    if columns is None:
        columns = ['ElementName', 'ElementType', 'ElementParameter', 'ParameterValue']
    sd2 = SDDS(knob)
    data = {k: sd2.c(k)[p, :] for k in columns}
    df = pd.DataFrame(data=data, index=[data['ElementName'], data['ElementType'], data['ElementParameter']])
    if enforce_unique_index:
        df = df[df.ElementType != 'MARK']
        if not df.index.is_unique:
            raise Exception(f'Duplicate non-marker elements found: {df.index.values[df.index.duplicated()]}')
    del sd2
    return df.sort_index()


def interpolate_parameters(reference_df: pd.DataFrame,
                           value: float,
                           max_df: pd.DataFrame,
                           max_val: float,
                           min_df: Optional[pd.DataFrame] = None,
                           min_val: Optional[float] = None,
                           el_name: Optional[str] = None,
                           el_type: Optional[str] = None,
                           el_parameter: Optional[str] = None):
    """
    Interpolates from reference parameters to another parameter set, proportional to the desired value.
    If no min state given, assumed to be the negative of max knob.
    :param reference_df: The reference (no-knob) state
    :param value: Scaling factor for knob application
    :param max_df: Knob for a positive shift
    :param max_val: Value of positive shift knob
    :param min_df:
    :param min_val:
    :param el_name: Filter by element name
    :param el_type: Filter by element type
    :param el_parameter: Filter by parameter name
    :return:
    """
    df = reference_df
    if el_name:
        df = df.loc[(df.ElementName.str.startswith(el_name))]
    if el_type:
        df = df.loc[(df.ElementType == el_type)]
    if el_parameter:
        df = df.loc[(df.ElementParameter == el_parameter)]
    assert len(df) > 0
    assert max_val > 0
    df = df.copy()
    # print('Initial:',df.iloc[0].values)
    if value == 0:
        return df
    elif value < 0:
        if min_df is None:
            endpoint = max_df
            ratio = value / max_val
        else:
            assert min_val < 0
            endpoint = min_df
            ratio = value / min_val
        # print(minfile)
    else:
        endpoint = max_df
        ratio = value / max_val
    # if not df.index.equals(__filter_df(endpoint, el_name, el_type, el_parameter).index):
    if not np.all(df.index.isin(endpoint.index)):
        print(df.index)
        print(endpoint.index)
        raise Exception('Parameter file index mismatch - aborting!')
    assert df.index.is_unique
    delta = ratio * (endpoint.loc[:, 'ParameterValue'] - df.loc[:, 'ParameterValue'])
    df.loc[:, 'ParameterValue'] = df.loc[:, 'ParameterValue'] + delta
    # print(ratio)
    # print(dfend.iloc[0], dfref.iloc[0])
    # print('Final:',df.iloc[0].values)
    return df


def __filter_df(df: pd.DataFrame,
                el_name: str,
                el_type: str,
                el_parameter: str) -> pd.DataFrame:
    """
    Remove entries not matching whatever strings are provided for element name, type, and parameter name
    :param df:
    :param el_name:
    :param el_type:
    :param el_parameter:
    :return:
    """
    if el_name:
        df = df.loc[(df.ElementName.str.startswith(el_name))]
    if el_type:
        df = df.loc[(df.ElementType == el_type)]
    if el_parameter:
        df = df.loc[(df.ElementParameter == el_parameter)]
    return df


# pyIOTA.elegant.io.interpolate_parameters(df_baseline, 0.1, quad_knobs['nux'][0][0], 0.1).loc['QA2R','KQUAD','K1']
# pyIOTA.elegant.io.interpolate_parameters(df_baseline, -0.1, quad_knobs['nux'][0][0], 0.1).loc['QA2R','KQUAD','K1']
# pyIOTA.elegant.io.interpolate_parameters(df_baseline, -0.1, quad_knobs['nux'][0][0], 0.1, quad_knobs['nux'][0][0], -0.1).loc['QA2R','KQUAD','K1']
# pyIOTA.elegant.io.interpolate_parameters(df_baseline, 0, quad_knobs['nux'][0][0], 0.1).loc['QA2R','KQUAD','K1']
# pyIOTA.elegant.io.interpolate_parameters(quad_knobs['nux'][0][0], 0, quad_knobs['nux'][0][0], 0.1).loc['QA2R','KQUAD','K1']


def write_df_to_parameter_file(fpath: Path,
                               df: pd.DataFrame,
                               parameters: dict = None):
    """
    Helper method - write dataframe to knob file (ASCII encoding).
    :param parameters:
    :param fpath:
    :param df:
    """
    assert not fpath.is_dir()
    if fpath.exists():
        print(f'Warning - knob {fpath} already exists, overwriting')
    df = df.loc[:, ['ElementName', 'ElementParameter', 'ParameterValue']]
    # print(df)
    x = sdds.SDDS(10)
    x.setDescription("params", str(fpath))
    names = ['ElementName', 'ElementParameter', 'ParameterValue']
    types = [x.SDDS_STRING, x.SDDS_STRING, x.SDDS_DOUBLE]
    if parameters:
        for (k, v) in parameters.items():
            if isinstance(v, int):
                v = float(v)
            if not isinstance(v, float):
                raise Exception("Only numeric parameters supported - add strings/etc. manually plz")
            x.defineSimpleParameter(k, x.SDDS_DOUBLE)
            x.setParameterValueList(k, [v])
    for i, n in enumerate(names):
        x.defineSimpleColumn(names[i], types[i])
    column_data = [[[]], [[]], [[]]]
    for row in df.itertuples():
        # print(row)
        for i, r in enumerate(row):
            if i > 0:
                column_data[i - 1][0].append(r)
    for i, n in enumerate(names):
        x.setColumnValueLists(names[i], column_data[i])
    # print(column_data)
    x.save(str(fpath))
    del x


def __wipe_directory(d: Path, wipe: bool = False):
    if wipe:
        n_deleted = 0
        if d.exists():
            for p in d.iterdir():
                if p.is_dir():
                    shutil.rmtree(p)
                else:
                    p.unlink()
                n_deleted += 1
        return n_deleted
    else:
        return -1


def prepare_folder_structure(work_folder: Path,
                             data_folder: Path,
                             work_dirs_to_wipe: List[Path] = None,
                             work_dirs_to_create: List[Path] = None,
                             data_dirs_to_wipe: List[Path] = None,
                             data_dirs_to_create: List[Path] = None,
                             wipe: bool = True):
    """
    Wipes and recreates folder structure for elegant simulation run
    :param work_dirs_to_wipe: Subdirectories to wipe - by default, lattices+tasks+parameters
    :param work_dirs_to_create: Subdirectories to create
    :param data_dirs_to_wipe:
    :param data_dirs_to_create:
    :param work_folder: Root folder for storing all input files/knobs/etc.
    :param data_folder: Root folder for storing all output
    :param wipe:
    :return:
    """
    print('Work folder: ', work_folder)
    print('Data folder: ', data_folder)

    if work_dirs_to_wipe is None:
        work_dirs_to_wipe = [Path('lattices/'), Path('tasks/'), Path('parameters/')]

    if data_dirs_to_wipe is None:
        data_dirs_to_wipe = [Path('.')]

    if work_dirs_to_create is None:
        work_dirs_to_create = work_dirs_to_wipe.copy()

    if data_dirs_to_create is None:
        data_dirs_to_create = data_dirs_to_wipe.copy()

    for path in work_dirs_to_wipe + work_dirs_to_create + data_dirs_to_wipe + data_dirs_to_create:
        if not isinstance(path, Path) or path.is_absolute() or (path.exists() and not path.is_dir()):
            raise Exception(f'Path {path} is not a valid subdirectory')

    for path in [work_folder / p for p in work_dirs_to_wipe] + [data_folder / p for p in data_dirs_to_wipe]:
        n = __wipe_directory(path, wipe)
        print(f'Wiping: {str(path):<60s} deleted {n} objects')

    for path in [work_folder / p for p in work_dirs_to_create] + [data_folder / p for p in data_dirs_to_create]:
        if not path.is_dir():
            print('Making dir: ', path)
            path.mkdir(parents=True, exist_ok=True)
        else:
            print('Directory exists: ', path)


# Deprecated
def prepare_folders(work_folder, data_folder, dirs_to_wipe, dirs_to_create, wipe=True):
    print('Work folder: ', work_folder)
    print('Data folder: ', data_folder)

    def wipe_directory(d):
        if wipe:
            n_deleted = 0
            dobj = Path(d)
            if dobj.exists():
                for p in Path(d).iterdir():
                    if p.is_dir():
                        shutil.rmtree(p)
                    else:
                        p.unlink()
                    n_deleted += 1
            return n_deleted
        else:
            return -1

    for path in dirs_to_wipe:
        n = wipe_directory(path)
        print(f'Wiping {path:<60s} deleted {n} objects')

    for path in dirs_to_create:
        pp = Path(path)
        if not pp.is_dir():
            print('Making dir: ', path)
            pp.mkdir(parents=True, exist_ok=True)
        else:
            print('Directory exists: ', path)
