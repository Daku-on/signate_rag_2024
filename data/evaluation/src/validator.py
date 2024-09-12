import os
import time
import codecs
from typing import Any
import tiktoken
import pandas as pd


class Validator():
    def __init__(self, data_format: dict, verbose=False) -> None:
        assert isinstance(data_format, dict)
        self.data_format = data_format
        assert {'samples', 'keys', 'dtype'}.issubset(set(self.data_format.keys()))
        if verbose:
            print('\nValidation details:')
            for k, v in self.data_format.items():
                print('  {}: {}'.format(k, v))
        self.data = None
        print('\nValidation:')


    def check_data(self, result) -> None:
        raise NotImplementedError


    def check_samples(self, result) -> None:
        raise NotImplementedError


    def check_dtype(self, result) -> None:
        raise NotImplementedError


    def check_keys(self, result) -> None:
        raise NotImplementedError


    def check_details(self, result) -> None:
        raise NotImplementedError


    def validate(self, result) -> None:
        start = time.time()
        self.check_data(result)
        self.check_samples(result)
        self.check_dtype(result)
        self.check_keys(result)
        self.check_details(result)
        print('  time elapsed: {}[s]'.format(time.time()-start))


    def get_data(self) -> Any:
        return self.data


class DataFrameValidator(Validator):
    def check_data(self, result: str) -> None:
        msg = '  Checking data...'
        print(msg, end='\r')
        assert isinstance(result, str) and 'ext' in self.data_format
        ext = os.path.splitext(result)[-1]
        if ext != self.data_format['ext']:
            raise ExtentionError('Invalid file extention: {}!={}(expected)'.format(ext, self.data_format['ext']))

        sep = ' '
        if self.data_format['ext']=='.csv':
            sep = ','
        elif self.data_format['ext']=='.tsv':
            sep = '\t'
        elif self.data_format['ext']=='.txt':
            sep = ' '

        with codecs.open(result, 'r', 'utf-8-sig') as f:
            for i, line in enumerate(f):
                sp = line.rstrip().split(sep)
                if len(sp) <= 1 or (len(sp)==2 and len(sp[-1])==0):
                    raise DelimiterError('Invalid delimiter found in line {} or not enough data.'.format(i+1))
        try:
            self.data = pd.read_csv(result, header=None, sep=sep, encoding='utf-8', index_col=0)
        except Exception as e:
            raise e
        print(msg+'Done')


    def check_samples(self, result) -> None:
        msg = '  Checking samples...'
        print(msg, end='\r')
        samples = set(self.data.index)
        if samples != self.data_format['samples']:
            raise SampleError('Missing samples or invalid samples found.')
        isnull = self.data.isnull().sum(axis=0)
        count = 0
        for k, v in isnull.items():
            if v:
                raise NullError('Missing value found in column {}'.format(k))
            count+=1
            print(msg+'{}%'.format(int(100*count/len(self.data))), end='\r')
        print(msg+'Done')


    def check_dtype(self, result) -> None:
        msg = '  Checking dtype...'
        print(msg, end='\r')
        assert isinstance(self.data_format['dtype'], list)
        if len(self.data.columns)!=len(self.data_format['dtype']):
            raise NumColumnsError('Column mismatch: {}!={}(expected)'.format(len(self.data.columns), len(self.data_format['dtype'])))
        count = 0
        for k, v in self.data.dtypes.items():
            if v != self.data_format['dtype'][k-1]: # type: ignore
                raise DtypeError('Invalid data type found in column {}: {}!={}(expected)'.format(k, v, self.data_format['dtype'][k])) # type: ignore
            count+=1
            print(msg+'{}%'.format(int(100*count/len(self.data))), end='\r')
        print(msg+'Done')


    def check_keys(self, result) -> None:
        pass


    def check_details(self, result) -> None:
        try:
            encoding = tiktoken.encoding_for_model(self.data_format['model'])
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        msg = '  Checking tokens...'
        print(msg, end='\r')
        count = 0
        for i, d in self.data.iterrows():
            tokens = encoding.encode(d[1])
            if len(tokens)>self.data_format['max_num_tokens']:
                raise MaximumExceedError('Number of tokens exceeded in sample {}: {}(must be < {})'.format(i, len(tokens), self.data_format['max_num_tokens']))
            count+=1
            print(msg+'{}%'.format(int(100*count/len(self.data))), end='\r')
        print(msg+'Done')


class SampleError(Exception):
    pass


class ElementError(Exception):
    pass


class DtypeError(Exception):
    pass


class ExtentionError(Exception):
    pass


class DelimiterError(Exception):
    pass


class NumColumnsError(Exception):
    pass


class NullError(Exception):
    pass


class DiscreteDataError(Exception):
    pass


class MaximumExceedError(Exception):
    pass


class InstanceError(Exception):
    pass