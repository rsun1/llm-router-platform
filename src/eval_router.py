import os
import pandas as pd
from llm_router_part1_router import _classify_int
from classify_lora import _classify_lora
import numpy as np

EVAL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'eval', 'router_eval.csv')
SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'output', 'eval_compare.csv')

def evaluate(classify_fn, csv_path=EVAL_PATH, verbose=True):
    df = pd.read_csv(csv_path)

    preds = []
    for i, q in enumerate(df['query'], start=1):
        pred = classify_fn(q)
        preds.append(pred)
        if verbose:
            print(f'[{i}/{len(df)}] pred={pred} | {q[:45]}')

    df['pred'] = preds
    df['ok'] = df['pred'] == df['label']

    clear = df[~df.is_boundary]
    boundary = df[df.is_boundary]

    print()
    print('total   :', df.ok.sum(), '/', len(df))
    print('clear   :', clear.ok.sum(), '/', len(clear))
    print('boundary:', boundary.ok.sum(), '/', len(boundary))

    wrong = df[~df.ok]
    if len(wrong) > 0:
        print('\nwrong:')
        print(wrong[['id', 'label', 'pred']].to_string(index=False))

    return df


if __name__ == '__main__':
    df_lora = evaluate(_classify_lora)
    df_ollama = evaluate(_classify_int)
    merged = pd.merge(df_ollama, df_lora, on='id', suffixes=('_ollama', '_lora'))
    
    conditions = [
      merged.ok_ollama & merged.ok_lora,
      merged.ok_ollama & ~merged.ok_lora,
      ~merged.ok_ollama & merged.ok_lora,
    ]
    choices = ['both', 'ollama', 'lora']
    merged['winner'] = np.select(conditions, choices, default='neither')

    out = merged[['id', 'query_ollama', 'label_ollama', 'is_boundary_ollama',
                'pred_ollama', 'pred_lora', 'winner']]
    out.to_csv(SAVE_PATH, index=False)

