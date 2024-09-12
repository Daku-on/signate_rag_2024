import time
import json
import pandas as pd
import tiktoken
from openai import OpenAI
from tqdm import tqdm


class Evaluator():
    def __init__(self, sub: pd.DataFrame, ans: pd.DataFrame)->None:
        print('\nEvaluation:')
        self.sub = sub
        self.ans = ans


    def evaluate(self)->None:
        raise NotImplementedError


class CRAGEvaluator(Evaluator):
    def evaluate(self, model_name: str, save_sims: bool=True) -> tuple:
        print('  By CRAG')
        print('  llm: {}'.format(model_name))
        try:
            encoding = tiktoken.encoding_for_model(model_name)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")

        client = OpenAI()
        start = time.time()
        ans_sims = {}
        score = 0
        for i, true in tqdm(self.ans.iterrows()):
            res, num_tokens = self._judge_by_crag(self.sub.loc[i][1], true[1], client, model_name, encoding) # type: ignore
            if save_sims:
                ans_sims[i] = {
                    'judge_result': res['judged'],
                    'num_tokens': num_tokens
                }
            if res['judged']=='Perfect':
                score += 1
            elif res['judged']=='Acceptable':
                score += 0.5
            elif res['judged']=='Incorrect':
                score += -1

        score /= len(self.ans)
        print('  time elapsed: {}[s]\n'.format(time.time()-start))
        if save_sims:
            return score, ans_sims
        else:
            return score, None


    def _judge_by_crag(self, pred: str, true: str, client, model_name: str, encoding)->tuple[dict, int]:
        system_prompt = """
        与えられた問題のground_truthとanswerを比較してその結果を"Perfect", "Acceptable", "Missing", "Incorrect"の中から一つだけ選んで答えてください. それぞれの定義は以下の通り.

        # 定義
        Perfect: answerが問題に正しく回答しており, 幻覚的な内容を含んでいない.
        Acceptable: answerが問題の回答として有効な内容を含んでいるが, わずかな誤りも含んでいる. ただし, 有効性を壊すほどではない.
        Missing: answerが「わかりません」,「見つかりません」, 空の回答, または元の質問を明確にするための要求を含んでいる.
        Incorrect: answerが間違っているか問題と無関係な内容を含んでいる.
        
        JSON形式でkeyとして"judged"を含みそのvalueに結果を記載して出力すること.
        """
        prompt = 'ground_truth: {} answer: {}\n'.format(true, pred)
        num_tokens = len(encoding.encode(pred))
        
        input_messages = [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        for _ in range(3):
            try:
                response = client.chat.completions.create(
                    model = model_name,
                    temperature = 0,
                    seed = 0,
                    timeout = 1200,
                    response_format={"type": "json_object"},
                    messages=input_messages
                ).choices[0].message.content

                return json.loads(response), num_tokens

            except Exception as e:
                print("openai chatcompletion failed. retry...\n")
                print(e)
                time.sleep(60)

        else:
            print("maximum trial.\n")
            raise MaximumTrialError


class MaximumTrialError(Exception):
    pass
