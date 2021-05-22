import numpy as np
import torch
import argparse
import time
import torch.utils.data as Data
from hierarchical_att_model import HierAttNet
import torch.nn as nn
from utils.prepare_data import *
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", dest="b", type=str, default='0')
parser.add_argument("--lr", dest="lr", type=float, default=0.005)
args = parser.parse_args()


def train():
    SEED = 129
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.b
    os.environ['PYTHONHASHSEED'] =str(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.backends.cudnn.deterministic = True

    label2id = {}
    label2id["O"] = 0
    label2id["E-happiness"] = 1
    label2id["E-fear"] = 2
    label2id["E-surprise"] = 3
    label2id["E-sadness"] = 4
    label2id["E-disgust"] = 5
    label2id["E-anger"] = 6
    label2id["C-happiness"] = 7
    label2id["C-fear"] = 8
    label2id["C-surprise"] = 9
    label2id["C-sadness"] = 10
    label2id["C-disgust"] = 11
    label2id["C-anger"] = 12
    label2id["B-happiness"] = 13
    label2id["B-fear"] = 14
    label2id["B-surprise"] = 15
    label2id["B-sadness"] = 16
    label2id["B-disgust"] = 17
    label2id["B-anger"] = 18
    emo2id = {}
    emo2id["O"] = 0
    emo2id["happiness"] = 1
    emo2id["fear"] = 2
    emo2id["surprise"] = 3
    emo2id["sadness"] = 4
    emo2id["disgust"] = 5
    emo2id["anger"] = 6

    word_idx_rev, word_idx, embedding, embedding_pos = load_w2v(200, 100, 'data/clause_keywords.csv',
                                                                'data/w2v_200.txt')
    p4, r4, f2 = [], [], []
    p5, r5, f5 = [], [], []
    p6, r6, f6 = [], [], []

    for fold in range(1, 11):
        print('############# fold {} begin ###############'.format(fold))
        flag_1,flag_2,flag_3 = [],[],[]
        best_pair = [-1, -1, -1, -1]
        best_emotion = [-1, -1, -1, -1]
        best_cause = [-1, -1, -1, -1]
        train_file_name = 'fold{}_train.txt'.format(fold)
        test_file_name = 'fold{}_test.txt'.format(fold)
        tr_doc_id, tr_y_unified, tr_y_position, tr_y_cause,  tr_x, tr_sen_len = load_data('data_combine/' + train_file_name, word_idx,label2id, emo2id, 75, 30)
        te_doc_id, te_y_unified, te_y_position, te_y_cause,  te_x, te_sen_len = load_data('data_combine/' + test_file_name, word_idx,label2id, emo2id, 75, 30)

        train_pair = load_pair(train_file_name)
        test_pair = load_pair(test_file_name)
        #print(test_pair)
        index = load_index('data_combine/all_data_pair.txt')

        print('train docs: {}    test docs: {}'.format(len(tr_x), len(te_x)))
        tr_x = torch.LongTensor(tr_x)
        te_x = torch.LongTensor(te_x)
        tr_y_position, tr_y_cause, tr_y_unified= torch.LongTensor(tr_y_position).argmax(2), torch.LongTensor(tr_y_cause).argmax(2), torch.LongTensor(tr_y_unified).argmax(2)

        tr_sen_len = torch.LongTensor(tr_sen_len).unsqueeze(1)
        tr_sen_len = tr_sen_len.expand_as(tr_y_position)
        tr_label = torch.cat((tr_y_position.unsqueeze(2), tr_y_cause.unsqueeze(2), tr_y_unified.unsqueeze(2), tr_sen_len.unsqueeze(2)), -1)  # shape 1750 * 75 * 4

        #print('before',te_y_unified.shape,te_y_unified)
        te_y_position, te_y_cause, te_y_unified = torch.LongTensor(te_y_position).argmax(2), torch.LongTensor(te_y_cause).argmax(2), torch.LongTensor(te_y_unified).argmax(2)
        te_sen_len = torch.LongTensor(te_sen_len).unsqueeze(1)
        te_sen_len = te_sen_len.expand_as(te_y_position)
        te_label = torch.cat((te_y_position.unsqueeze(2), te_y_cause.unsqueeze(2), te_y_unified.unsqueeze(2), te_sen_len.unsqueeze(2)), -1)

        #print('after',te_y_unified.shape,te_y_unified)

        train_data = Data.TensorDataset(tr_x, tr_label)
        test_data = Data.TensorDataset(te_x, te_label)
        train_loader = Data.DataLoader(dataset=train_data, batch_size=32, shuffle=True)
        test_loader = Data.DataLoader(dataset=test_data, batch_size=250, shuffle=True)
        embedding = torch.tensor(embedding)

        model = HierAttNet(100, 100, 32, 10, embedding, 75, 30)
        model.word_att_net.lookup.weight.requires_grad = False

        if torch.cuda.is_available():
            model.cuda()
        for name, parameters in model.named_parameters():
            print(name, ':', parameters.shape)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
        model.train()
        for epoch in range(30):
            step = 1
            for feature, label in train_loader:
                if torch.cuda.is_available():
                    feature = feature.cuda()
                    label = label.cuda()

                optimizer.zero_grad()
                output1,output2,output3 = model(feature)
                loss = model.loss_pre(output1,output2,output3,label)
                #print(loss)
                re_l2 = ['sent_att_net.fc1.weight','sent_att_net.fc2.weight','sent_att_net.fc3.weight','sent_att_net.fc1.bias','sent_att_net.fc2.bias','sent_att_net.fc3.bias']
                for name,parameters in model.named_parameters():
                    if name in re_l2 :
                        loss += torch.norm(parameters) *  1e-4
                loss.backward()
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 10)
                optimizer.step()
                print('epoch', epoch, 'step', step, "loss :", loss)
                step = step + 1

            if epoch >= 0:
                model.eval()
                loss_ls = []
                pre_test_1,pre_test_2,pre_test_3 = [],[],[]

                for te_feature, te_label in test_loader:
                    if torch.cuda.is_available():
                        te_feature = te_feature.cuda()
                        te_label = te_label.cuda()  # batch* seq * 4
                        #print('test label shape',te_label.shape)
                    with torch.no_grad():
                        predict1,predict2,predict3 = model(te_feature)

                    pred_y_cause, true_y_cause, pred_y_pos, true_y_pos, pp, pr, pf1 = mapcausepos(predict3.permute(1,0,2).cpu().numpy(), te_label[:,:,2].cpu().numpy())
                    print('pair_predict: test p {:.4f} r {:.4f} f1 {:.4f}'.format(pp, pr, pf1))
                    if pf1 > best_pair[-1]:
                        best_pair[0] = epoch
                        best_pair[1] = pp
                        best_pair[2] = pr
                        best_pair[3] = pf1

                        acc, p, r, f1 = acc_prf(pred_y_cause, true_y_cause, te_label[:,-1,-1].tolist())
                        result_avg_cause = [acc, p, r, f1]
                        print('cause extraction p is {} r is {} f1 is {}'.format(p,r,f1))
                        best_cause[0] = epoch
                        best_cause[1] = p
                        best_cause[2] = r
                        best_cause[3] = f1

                        acc, p, r, f1 = acc_prf(pred_y_pos, true_y_pos, te_label[:,-1,-1].tolist())
                        result_avg_pos = [acc, p, r, f1]
                        print('emotion extraction p is {} r is {} f1 is {}'.format(p,r,f1))
                        best_emotion[0] = epoch
                        best_emotion[1] = p
                        best_emotion[2] = r
                        best_emotion[3] = f1

            model.train()

        print('this fold p is {} r is {} f1 is {} best epoch is {}'.format(best_pair[1], best_pair[2], best_pair[3], best_pair[0]))
        p4.append(best_pair[1])
        r4.append(best_pair[2])
        f2.append(best_pair[3])
        print('this fold best emotion p is {} r is {} f1 is {} best epoch is {}'.format(best_emotion[1], best_emotion[2], best_emotion[3], best_emotion[0]))

        print('this fold best cause p is {} r is {} f1 is {} best epoch is {}'.format(best_cause[1], best_cause[2], best_cause[3], best_cause[0]))

        p5.append(best_emotion[1])
        r5.append(best_emotion[2])
        f5.append(best_emotion[3])

        p6.append(best_cause[1])
        r6.append(best_cause[2])
        f6.append(best_cause[3])

    all_results = [p4, r4, f2]
    p4, r4, f1 = map(lambda x: np.array(x).mean(), all_results)
    print(p4, r4, f1)

    all_results = [p5, r5, f5]
    p4, r4, f1 = map(lambda x: np.array(x).mean(), all_results)
    print('finally emotion result: p is {} r is {} f1 is {}'.format(p4, r4, f1))

    all_results = [p6, r6, f6]
    p4, r4, f1 = map(lambda x: np.array(x).mean(), all_results)
    print('finally cause result: p is {} r is {} f1 is {}'.format(p4, r4, f1))

if __name__ == "__main__":
    train()
