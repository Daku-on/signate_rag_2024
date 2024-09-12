import time
import pandas as pd

class FormatSetter():
    def __init__(self, ans_txt_path: str, max_num_tokens: int, keys: dict, ext: str, model: str) -> None:
        """
        no header, index column=0
        """
        print('\nFormat Setting:')
        self.start = time.time()
        self.ans_txt = pd.read_csv(ans_txt_path, index_col=0, header=None)
        self.keys = keys
        self.ext = ext
        self.max_num_tokens = max_num_tokens
        self.model = model


    def get_format(self) -> dict:
        data_format = {
            'samples': set(self.ans_txt.index),
            'dtype': list(self.ans_txt.dtypes),
            'keys': self.keys,
            'ext': self.ext,
            'max_num_tokens': self.max_num_tokens,
            'model': self.model
        }
        print('  time elapsed: {}[s]'.format(time.time()-self.start))

        return data_format
