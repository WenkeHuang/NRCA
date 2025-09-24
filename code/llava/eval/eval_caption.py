import os
import argparse
import json
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import tempfile
class COCOEvaler(object):
    def __init__(self, annfile):
        super(COCOEvaler, self).__init__()
        self.coco = COCO(annfile)
        if not os.path.exists('./tmp'):
            os.mkdir('./tmp')

    def eval(self, result):
        in_file = tempfile.NamedTemporaryFile(mode='w', delete=False, dir='./tmp')
        json.dump(result, in_file)
        in_file.close()

        cocoRes = self.coco.loadRes(in_file.name)
        cocoEval = COCOEvalCap(self.coco, cocoRes)
        cocoEval.evaluate()
        os.remove(in_file.name)
        return cocoEval.eval
def jsonl2json(pred):
    data=[]
    for pred_ in pred:
        data.append(pred_)
    return data

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation-file',default='/data0/data_wk/playground/DataEval/coco-cap/captions_test5k.json', type=str)
    parser.add_argument('--result-file',default='/home/huangwenke/Wenke_Project/VILAPro/runs/eval/3b/4-2-5/coco/Normal/coco_capation/outputs.jsonl', type=str)
    parser.add_argument('--result-dir', type=str)
    parser.add_argument('--summary-output-dir', type=str, default=None)
    return parser.parse_args()

def main() -> None:
    args = get_args()
    evaler = COCOEvaler(args.annotation_file)
    preds= [json.loads(line) for line in open(args.result_file)]
    preds=jsonl2json(preds)
    json.dump(preds,open('./tmp/preds,json','w')) # 将每一行的 JSON 格式字符串转换为 Python 字典对象
    res=evaler.eval(json.load(open('./tmp/preds,json')))
    print(res)

    # 汇总输出到指定的txt文件
    if args.summary_output_dir is not None:
        with open(args.summary_output_dir, 'a') as f_sum:
            f_sum.write('\n### Evaluation Results ###\n')
            f_sum.write('Bleu_1: {:.4f}\n'.format(res['Bleu_1']))
            f_sum.write('Bleu_2: {:.4f}\n'.format(res['Bleu_2']))
            f_sum.write('Bleu_3: {:.4f}\n'.format(res['Bleu_3']))
            f_sum.write('Bleu_4: {:.4f}\n'.format(res['Bleu_4']))
            f_sum.write('CIDEr: {:.4f}\n'.format(res['CIDEr']))
            f_sum.write('METEOR: {:.4f}\n'.format(res['METEOR']))
            f_sum.write('ROUGE_L: {:.4f}\n'.format(res['ROUGE_L']))
            # f_sum.write('SPICE: {:.4f}\n'.format(res['SPICE']))
            f_sum.write('\nSummary Completed.\n')

if __name__ == "__main__":
    main()


    # ### 将计算结果统一输出到一个汇总的txt文件中 ###
    # if args.summary_output_dir is not None:
    #     with open(args.summary_output_dir, 'a') as f_sum:
    #         f_sum.write('\nSamples: {}\nCIDEr on Flickr30k: {:.4f}\n'.format(len(outputs), cider_val))
