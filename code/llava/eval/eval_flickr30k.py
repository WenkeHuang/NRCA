############### for debugging ###############
import debugpy
try:
    # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
    debugpy.listen(("localhost", 9505))
    print("Waiting for debugger attach")
    debugpy.wait_for_client()
except Exception as e:
    print('Debug Failed.')
############### for debugging ###############




import sys
sys.path.append('/home/huangwenke/Wenke_Project/LLaVAPro/llava/eval/pycocoevalcap')

from .pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from .pycocoevalcap.cider.cider import Cider
import pandas as pd
import json
import os
import argparse



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file', type=str, default=None)
    parser.add_argument('--result-file', type=str, default=None)
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--summary-output-dir', type=str, default=None)
    return parser.parse_args()


class Evaluator:
    def __init__(self) -> None:
        self.tokenizer = PTBTokenizer()
        self.scorer_list = [
            (Cider(), "CIDEr"),
        ]
        self.evaluation_report = {}

    def do_the_thing(self, golden_reference, candidate_reference):
        golden_reference = self.tokenizer.tokenize(golden_reference)
        candidate_reference = self.tokenizer.tokenize(candidate_reference)
        
        # From this point, some variables are named as in the original code
        # I have no idea why they name like these
        # The original code: https://github.com/salaniz/pycocoevalcap/blob/a24f74c408c918f1f4ec34e9514bc8a76ce41ffd/eval.py#L51-L63
        for scorer, method in self.scorer_list:
            score, scores = scorer.compute_score(golden_reference, candidate_reference)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    self.evaluation_report[m] = sc
            else:
                self.evaluation_report[method] = score




def eval_single(annotation_file, result_file):
    df = pd.read_csv(annotation_file)
    df = df[df['split'] == 'test']
    golden_reference = []

    outputs = [json.loads(line)['text'] for line in open(result_file)]

    candidate_reference = []

    for i, x in enumerate(df.iloc):
        s = x['raw'][2:-2].replace('"','').split(',')
        golden_reference.append(s)
        print(outputs[i])
        candidate_reference.append(outputs[i])

    golden_reference = {k: [{'caption': x} for x in v] for k, v in enumerate(golden_reference)}

    candidate_reference = {k: [{'caption': v}] for k, v in enumerate(candidate_reference)}
    # breakpoint()
    evaluator = Evaluator()
    evaluator.do_the_thing(golden_reference, candidate_reference)

    print(evaluator.evaluation_report)

    cider_val = evaluator.evaluation_report['CIDEr']


    ### 将结果写到文件中 ###
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'result-flickr30k.txt')
        with open(output_file, 'w') as f:
            f.write('Samples: {}\nCIDEr: {:.4f}\n'.format(len(outputs), cider_val))
    

    ### 将计算结果统一输出到一个汇总的txt文件中 ###
    if args.summary_output_dir is not None: 
        with open(args.summary_output_dir, 'a') as f_sum:
            f_sum.write('\nSamples: {}\nCIDEr on Flickr30k: {:.4f}\n'.format(len(outputs), cider_val))





if __name__ == "__main__":
    args = get_args()

    if args.result_file is not None:
        eval_single(args.annotation_file, args.result_file)

