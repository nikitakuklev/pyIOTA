__all__ = ['Generator', 'Parser']

import itertools
import random
import collections
from pathlib import Path
from typing import Union, Any, Iterable, Tuple

import numpy as np
import pandas as pd


class Parser:
    def __init__(self):
        pass

    def parse(self, file: Path):
        """ Parse filename to extract parameters """
        roots = str(file.name).split('__')
        assert len(roots) == 2
        sim_name = roots[0]
        param_string = roots[1].rsplit('.', 1)[0]
        param_string = param_string.strip('_')
        param_splits = param_string.split('_')
        if not len(param_splits) % 2 == 0:
            raise Exception(f'Parse issue: {param_splits}')
        keys = param_splits[::2]
        values_temp = param_splits[1::2]
        values = []
        for v in values_temp:
            try:
                if '.' in v and 'e' in v and ('+' in v or '-' in v):
                    values.append(float(v))
                else:
                    values.append(int(v))
            except ValueError:
                values.append(v)
        d = {k: v for (k, v) in zip(keys, values)}
        d['name'] = sim_name
        return d

    def parse_list(self, name_list):
        """ Parse list of filenames """
        data_dicts = []
        for name in name_list:
            data_dicts.append(self.parse(name))
        return pd.DataFrame(data=data_dicts)


class Generator:
    """
    Generates sets of simulation parameters by permuting given set of value lists, and generates
    strings suitable for use in file names. Names can be parsed back with Parser class. Can be used as UUID.
    """

    def __init__(self, parameter_spec: dict = None):
        self.parameters = parameter_spec or {}
        self.name_links = {}

    def reset(self):
        self.parameters = {}
        self.name_links = {}

    def _flatten_parameters(self) -> list:
        # TODO: Reimplement with proper traversal generator
        ls = []
        [ls.append(sublist) if isinstance(sublist, str) else ls.extend(sublist) for sublist in self.parameters]
        return ls

    def add_parameter(self,
                      parameter: Union[str, Tuple[str]],
                      values: Union[Any, Iterable[Any]],
                      override: bool = False):
        """
        Add a parameter to list. Acceptable syntax:
            'param',value or list of values
            ('param1','param2',...),value or list of values
            ('param1','param2',...),[('value1','value2',...),('value1','value2',...)]
        :param override: Whether value replacement is allowed
        :param parameter: Either string or tuple of strings. If tuple, parameters are treated as linked (have same value)
        :param values: List or array of values
        :return:
        """
        assert not self.name_links
        if not override: assert parameter not in self.parameters
        is_single_param = isinstance(parameter, str)
        is_multiple_param = (isinstance(parameter, tuple) and all((isinstance(ps, str) for ps in parameter)))
        is_value_list = isinstance(values, (collections.Sequence, np.ndarray, tuple))
        assert is_single_param or is_multiple_param
        if is_multiple_param:
            if is_value_list:
                if all(isinstance(v, tuple) for v in values):
                    # tuple of values, 1 for each parameter
                    assert all(len(v) == len(parameter) for v in values)
                else:
                    # parameters share same value
                    pass
        if is_value_list and not isinstance(values, str):
            values = list(values)
        else:
            values = [values]
        self.parameters[parameter] = values

    def add_parameters(self, parameter_spec: dict) -> None:
        """
        Add parameters from dictionary.
        :param parameter_spec:
        :return:
        """
        assert isinstance(parameter_spec, dict)
        assert not self.name_links
        [self.add_parameter(k, v) for (k, v) in parameter_spec.items()]

    def set_parameters(self, parameter_spec: dict) -> None:
        self.parameters = parameter_spec

    def add_label_link(self, link_name: str, links: list) -> None:
        assert link_name not in self.name_links
        assert isinstance(links, list)
        assert all([l in self._flatten_parameters() for l in links])
        self.name_links[link_name] = links.copy()

    def generate_sets(self, downsample_to: int = None, generate_labels: bool = True) -> pd.DataFrame:
        """
        Use all current parameters to generate permutations, and output resulting set as DataFrame.
        :param generate_labels:
        :param downsample_to:
        :return:
        """
        keys = list(self.parameters.keys())
        values = list(self.parameters.values())
        permutations = list(itertools.product(*values))

        print(f'Generated {len(permutations)} permutations of {len(keys)} parameters ({self._flatten_parameters()})')
        if downsample_to:
            permutations = random.sample(permutations, downsample_to)
            print(f'Downsampled to: {len(permutations)}')

        df_data = {}
        p_idx = 0
        for p in self.parameters:
            if isinstance(p, str):
                df_data[p] = np.array([v[p_idx] for v in permutations])
                p_idx += 1
                # print(p, df_data[p][0])
            elif isinstance(p, tuple):
                for j, sub_p in enumerate(p):
                    if isinstance(permutations[0][p_idx], tuple):
                        df_data[sub_p] = np.array([v[p_idx][j] for v in permutations])
                    else:
                        df_data[sub_p] = np.array([v[p_idx] for v in permutations])
                    # print(sub_p, df_data[sub_p][0])
                p_idx += 1
        # print(df_data)
        df = pd.DataFrame(data=df_data)
        if generate_labels:
            self.name_links['label'] = df.columns
            label_strings = {k: [] for k in self.name_links.keys()}
            for r in df.itertuples(index=False):
                for (label, links) in self.name_links.items():
                    s = ['_']
                    for (k, v) in zip(r._fields, r):
                        if k in links:
                            if isinstance(v, str):
                                s.append(f'{k}_{v}_')
                            elif isinstance(v, int):
                                assert v < 1e6
                                s.append(f'{k}_{v:+06d}_')
                            elif isinstance(v, float):
                                s.append(f'{k}_{v:+.6e}_')
                            else:
                                raise Exception(f'Unknown parameter type {type(v)}')
                    s = ''.join(s)
                    if len(s) > 200:
                        raise ValueError(f'Label too long: {s}')
                    label_strings[label].append(s)
            for (k, v) in label_strings.items():
                df[k] = v
        return df
