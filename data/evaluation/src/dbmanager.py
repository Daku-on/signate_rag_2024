import time
import os
from typing import Any
import pandas as pd


class DBLoader():
    def __init__(self, db_path: str) -> None:
        """
        no header, index column=0
        """
        print('\nLoading Ground Truth:')
        start = time.time()
        self.db = pd.read_csv(db_path, index_col=0, header=None)
        print('  db shape: {}'.format(self.db.shape))
        print('  time elapsed: {}[s]'.format(time.time()-start))


    def get_db(self) -> pd.DataFrame:
        return self.db


class ResultHandler():
    def __init__(self, score: float, result: dict[str, Any], eval_result_dir:str) -> None:
        print('\nSaving the Results:')
        self.score = score
        self.result = result
        self.eval_result_dir = eval_result_dir


    def save(self):
        start = time.time()
        if not os.path.exists(self.eval_result_dir):
            os.mkdir(self.eval_result_dir)

        print('  Score: {}'.format(self.score))

        pd.DataFrame(self.result).T.to_csv(os.path.join(self.eval_result_dir, 'scoring.csv'), header=False)

        print('  time elapsed: {}[s]'.format(time.time()-start))
