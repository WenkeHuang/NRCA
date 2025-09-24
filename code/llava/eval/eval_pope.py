import os
import json
import argparse


# ############### for debugging ###############
# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9505))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     print('Debug Failed.')
# ############### for debugging ###############


def eval_pope(answers, label_file):
    label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]

    for answer in answers:
        text = answer['text']

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['text'] = 'no'
        else:
            answer['text'] = 'yes'

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer['text'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    print('TP\tFP\tTN\tFN\t')
    print('{}\t{}\t{}\t{}'.format(TP, FP, TN, FN))

    precision = float(TP) / float(TP + FP)
    recall = float(TP) / float(TP + FN)
    f1 = 2*precision*recall / (precision + recall)
    acc = (TP + TN) / (TP + TN + FP + FN)
    print('Accuracy: {}'.format(acc))
    print('Precision: {}'.format(precision))
    print('Recall: {}'.format(recall))
    print('F1 score: {}'.format(f1))
    print('Yes ratio: {}'.format(yes_ratio))
    print('%.3f, %.3f, %.3f, %.3f, %.3f' % (f1, acc, precision, recall, yes_ratio) )



def eval_pope_intofile(answers, label_file, output_file):
    label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]

    for answer in answers:
        text = answer['text']

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['text'] = 'no'
        else:
            answer['text'] = 'yes'

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer['text'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    yes_ratio = pred_list.count(1) / len(pred_list)

    TP, TN, FP, FN = 0, 0, 0, 0
    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    with open(output_file, 'a') as f:
        f.write('TP\tFP\tTN\tFN\n')
        f.write('{}\t{}\t{}\t{}\n'.format(TP, FP, TN, FN))

        precision = float(TP) / float(TP + FP)
        recall = float(TP) / float(TP + FN)
        f1 = 2 * precision * recall / (precision + recall)
        acc = (TP + TN) / (TP + TN + FP + FN)
        
        f.write('Accuracy: {}\n'.format(acc))
        f.write('Precision: {}\n'.format(precision))
        f.write('Recall: {}\n'.format(recall))
        f.write('F1 score: {}\n'.format(f1))
        f.write('Yes ratio: {}\n'.format(yes_ratio))
        f.write('%.3f, %.3f, %.3f, %.3f, %.3f\n' % (f1, acc, precision, recall, yes_ratio))






def eval_pope_intosumfile(answers, label_file, output_file):
    label_list = [json.loads(q)['label'] for q in open(label_file, 'r')]

    for answer in answers:
        text = answer['text']

        # Only keep the first sentence
        if text.find('.') != -1:
            text = text.split('.')[0]

        text = text.replace(',', '')
        words = text.split(' ')
        if 'No' in words or 'not' in words or 'no' in words:
            answer['text'] = 'no'
        else:
            answer['text'] = 'yes'

    for i in range(len(label_list)):
        if label_list[i] == 'no':
            label_list[i] = 0
        else:
            label_list[i] = 1

    pred_list = []
    for answer in answers:
        if answer['text'] == 'no':
            pred_list.append(0)
        else:
            pred_list.append(1)

    pos = 1
    neg = 0
    TP, TN, FP, FN = 0, 0, 0, 0

    for pred, label in zip(pred_list, label_list):
        if pred == pos and label == pos:
            TP += 1
        elif pred == pos and label == neg:
            FP += 1
        elif pred == neg and label == neg:
            TN += 1
        elif pred == neg and label == pos:
            FN += 1

    with open(output_file, 'a') as f:
        acc = (TP + TN) / (TP + TN + FP + FN)
        f.write('Accuracy: {}\n'.format(acc))









if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-dir", type=str)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--result-file", type=str)

    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--summary-output-dir", type=str)

    args = parser.parse_args()

    questions = [json.loads(line) for line in open(args.question_file)]
    questions = {question['question_id']: question for question in questions}
    answers = [json.loads(q) for q in open(args.result_file)]
    for file in os.listdir(args.annotation_dir):
        assert file.startswith('coco_pope_')
        assert file.endswith('.json')
        category = file[10:-5]
        cur_answers = [x for x in answers if questions[x['question_id']]['category'] == category]
        print('Category: {}, # samples: {}'.format(category, len(cur_answers)))
        eval_pope(cur_answers, os.path.join(args.annotation_dir, file))
        print("====================================")


    ##### 将结果写到文件中 #####
    if args.output_dir is not None:
        output_file = os.path.join(args.output_dir, 'result-pope.txt')
        with open(output_file, 'w') as f:
            pass    # 先清空文件

    for file in os.listdir(args.annotation_dir):
        assert file.startswith('coco_pope_')
        assert file.endswith('.json')
        category = file[10:-5]
        cur_answers = [x for x in answers if questions[x['question_id']]['category'] == category]

        if args.output_dir is not None:
            output_file = os.path.join(args.output_dir, 'result-pope.txt')
            with open(output_file, 'a') as f:
                f.write('Category: {}, # samples: {}\n'.format(category, len(cur_answers)))
                
            eval_pope_intofile(cur_answers, os.path.join(args.annotation_dir, file), output_file)
            
            with open(output_file, 'a') as f:
                f.write("====================================\n")


    ### 将计算结果统一输出到一个汇总的txt文件中 ###
    with open(args.summary_output_dir, 'a') as f_sum:
        f_sum.write('\nAccuracy on pope:\n')

    for file in os.listdir(args.annotation_dir):
        assert file.startswith('coco_pope_')
        assert file.endswith('.json')
        category = file[10:-5]
        cur_answers = [x for x in answers if questions[x['question_id']]['category'] == category]
     
        if args.summary_output_dir is not None:
            with open(args.summary_output_dir, 'a') as f_sum:
                f_sum.write('Category: {}, # samples: {}\n'.format(category, len(cur_answers)))
                    
            eval_pope_intosumfile(cur_answers, os.path.join(args.annotation_dir, file), args.summary_output_dir)
                
            with open(args.summary_output_dir, 'a') as f_sum:
                f_sum.write("====================================\n")
