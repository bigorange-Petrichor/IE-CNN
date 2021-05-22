import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score


def acc_prf(pred_y, true_y, length, average='binary'):
    mask = np.zeros((pred_y.shape[0],pred_y.shape[1]))
    tmp1, tmp2 = [], []
    for i,j in enumerate(length):
        mask[i][0:j] = 1
    for i in range(pred_y.shape[0]):
        for j in range(pred_y.shape[1]):
            if mask[i][j] == 1:
                tmp1.append(pred_y[i][j])
                tmp2.append(true_y[i][j])
    y_pred, y_true = np.array(tmp1), np.array(tmp2)
    acc = precision_score(y_true, y_pred, average='micro')
    p = precision_score(y_true, y_pred, average=average)
    r = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)
    return acc, p, r, f1

def load_pair(input_file, max_doc_len=75, max_sen_len=45):
    print('load data_file: {}'.format(input_file))
    pair_id_all = []
    inputFile = open('data_combine/' + input_file)
    while True:
        line = inputFile.readline()
        if line == '': break
        line = line.strip().split()
        doc_id = int(line[0])
        d_len = int(line[1])
        pairs = eval('[' + inputFile.readline().strip() + ']')
        pair_id_all.extend([doc_id * 10000 + p[0] * 100 + p[1] for p in pairs])  # give true pair a index
        for i in range(d_len):
            line = inputFile.readline().strip().split(',')

    return pair_id_all

def load_index(input_file, max_doc_len=75, max_sen_len=45):
    index = np.zeros((2110,76))
    inputFile = open( input_file)
    while True:
        line = inputFile.readline()
        if line == '': break
        line = line.strip().split()
        doc_id = int(line[0])
        d_len = int(line[1])
        pairs = eval('[' + inputFile.readline().strip() + ']')
        for p in pairs:
            index[doc_id][p[1]] = p[0]


        for i in range(d_len):
            line = inputFile.readline().strip().split(',')
    return index

def prf(pair_id_all, pred_y):
    s1, s3 = set(pair_id_all), set(pred_y)
    # print(s3)
    acc_num = len(s1 & s3)
    p, r = acc_num / (len(s3) + 1e-8), acc_num / (len(s1) + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p, r, f1


def document_result(pair_id_all, pred_y):
    s1, s3 = set(pair_id_all), set(pred_y)
    # print(s3)
    doc_id = []
    for i in pair_id_all:
        doc_id.append(int(i/10000))
    doc_id_1 = []  # multiple
    for i in doc_id:
        if doc_id.count(i) > 1:
            doc_id_1.append(i)
    pair_id_all_1,pair_id_all_2 = [],[]
    pred_y_1,pred_y_2 = [],[]
    for i in pair_id_all:
        if int(i/10000) in doc_id_1:
            pair_id_all_1.append(i)
        else:
            pair_id_all_2.append(i)
            
    for i in pred_y:
        if int(i/10000) in doc_id_1:
            pred_y_1.append(i)
        else:
            pred_y_2.append(i)
    p,r,f1 = prf(pair_id_all_1,pred_y_1)
    print('multiple is',p,r,f1)
    p,r,f1 = prf(pair_id_all_2,pred_y_2)
    print('one is ',p,r,f1)
    return p

def prf_2nd_step(pair_id_all, pred_y):
    s1, s3 = set(pair_id_all), set(pred_y)
    # print(s3)
    acc_num = len(s1 & s3)
    p, r = acc_num / (len(s3) + 1e-8), acc_num / (len(s1) + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p, r, f1

def emotion_step(pair_id_all, pred_y):
    new_cause, y1_cause = [], []
    for i in pair_id_all:
        new_cause.append(int(i / 10000) * 10000 + int(i % 10000 / 100))
    for i in pred_y:
        y1_cause.append(int(i / 10000) * 10000 + int(i % 10000 / 100))
    s1, s2 = set(new_cause), set(y1_cause)
    acc_num = len(s1 & s2)
    p, r = acc_num / (len(s2) + 1e-8), acc_num / (len(s1) + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p, r, f1

def cause_step(pair_id_all, pred_y):
    new_cause, y1_cause = [], []
    for i in pair_id_all:
        new_cause.append(int(i / 10000) * 10000 + i % 100)
    for i in pred_y:
        y1_cause.append(int(i / 10000) * 10000 + i % 100)
    s1, s2 = set(new_cause), set(y1_cause)
    acc_num = len(s1 & s2)
    p, r = acc_num / (len(s2) + 1e-8), acc_num / (len(s1) + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p, r, f1


def prf_and_step(pair_id_all, pred_y1, pred_y2):
    s1, s2, s3 = set(pair_id_all), set(pred_y1), set(pred_y2)
    s3 = s2 & s3
    acc_num = len(s1 & s3)
    p, r = acc_num / (len(s3) + 1e-8), acc_num / (len(s1) + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p, r, f1


def prf_union_step(pair_id_all, pred_y1, pred_y2):
    s1, s2, s3 = set(pair_id_all), set(pred_y1), set(pred_y2)
    s3 = s2 | s3
    acc_num = len(s1 & s3)
    p, r = acc_num / (len(s3) + 1e-8), acc_num / (len(s1) + 1e-8)
    f1 = 2 * p * r / (p + r + 1e-8)
    return p, r, f1


# input:two list to form two sets,then we can solve p,r,f1

def load_w2v(embedding_dim, embedding_dim_pos, train_file_path, embedding_path):
    print('\nload embedding...')

    words = []
    inputFile1 = open(train_file_path, 'r')
    for line in inputFile1.readlines():
        line = line.strip().split(',')
        emotion, clause = line[2], line[-1]
        words.extend([emotion] + clause.split())
    words = set(words)  # the set of all words
    word_idx = dict((c, k + 1) for k, c in enumerate(words))  # key:word value:index
    word_idx_rev = dict((k + 1, c) for k, c in enumerate(words))  # key:index value:word

    w2v = {}
    inputFile2 = open(embedding_path, 'r')
    inputFile2.readline()
    for line in inputFile2.readlines():
        line = line.strip().split(' ')
        w, ebd = line[0], line[1:]
        w2v[w] = ebd

    embedding = [list(np.zeros(embedding_dim))]
    hit = 0
    for item in words:
        if item in w2v:
            vec = list(map(float, w2v[item]))
            hit += 1
        else:
            vec = list(np.random.rand(embedding_dim) / 5. - 0.1)
        embedding.append(vec)
    print('w2v_file: {}\nall_words: {} hit_words: {}'.format(embedding_path, len(words), hit))

    embedding_pos = [list(np.zeros(embedding_dim_pos))]
    embedding_pos.extend([list(np.random.normal(loc=0.0, scale=0.1, size=embedding_dim_pos)) for i in range(200)])

    embedding, embedding_pos = np.array(embedding), np.array(embedding_pos)

    print("embedding.shape: {} embedding_pos.shape: {}".format(embedding.shape, embedding_pos.shape))
    print("load embedding done!\n")
    return word_idx_rev, word_idx, embedding, embedding_pos
    # two dict, embedding ,pos_embendding


def load_data(input_file, word_idx, label2id, emo2id, max_doc_len = 75, max_sen_len = 45):
    print('load data_file: {}'.format(input_file))
    x, sen_len, doc_len, y, l, el = [], [], [], [], [], []
    doc_id = []

    n_cut = 0
    inputFile = open(input_file, 'r', encoding='utf-8')
    while True:
        line = inputFile.readline()
        if line == '': break
        line = line.strip().split()
        doc_id.append(int(line[0]))
        d_len = int(line[1])
        pairs = eval('[' + inputFile.readline().strip() + ']')
        doc_len.append(d_len)
        pos, cause = zip(*pairs)

        y_tmp = np.zeros((max_doc_len, 19), np.int32)
        l_tmp = np.zeros((max_doc_len, 4), np.int32)
        el_tmp = np.zeros((max_doc_len, 7), np.int32)
        sen_len_tmp = np.zeros(max_doc_len, dtype=np.int32)
        x_tmp = np.zeros((max_doc_len, max_sen_len), dtype=np.int32)
        emolabeldict = {}

        for i in range(d_len):
            [sen_id, emolabel, _, words] = inputFile.readline().strip().split(',')
            emolabeldict[i] = emolabel.split("&")[0]
            sen_id = int(sen_id)

            if sen_id not in pos and sen_id not in cause:
                # if a clause is irrevelant, assign the label "O (Outside)"
                y_tmp[sen_id-1][label2id["O"]] = 1
                l_tmp[sen_id-1][0] = 1 # 0 is for "O"
                el_tmp[sen_id-1][0] = 1

            sen_len_tmp[i] = min(len(words.split()), max_sen_len)
            for j, word in enumerate(words.split()):
                if j >= max_sen_len:
                    n_cut += 1
                    break
                x_tmp[i][j] = int(word_idx[word])

        # Create separate emotion labels and cause labels for each clause
        for pair in pairs:
            if pair[0] == pair[1]:
                y_tmp[pair[0]-1][label2id["B-"+emolabeldict[pair[0]-1]]] = 1
                l_tmp[pair[0]-1][3] = 1 # 3 stands for both e and c
            else:
                if l_tmp[pair[0]-1][3] != 1:
                    y_tmp[pair[0]-1][label2id["E-"+emolabeldict[pair[0]-1]]] = 1
                    l_tmp[pair[0]-1][1] = 1 # 1 stands for e
                y_tmp[pair[1]-1][label2id["C-"+emolabeldict[pair[0]-1]]] = 1
                l_tmp[pair[1]-1][2] = 1 # 2 stands for c
            el_tmp[pair[0]-1][emo2id[emolabeldict[pair[0]-1]]] = 1
            el_tmp[pair[1]-1][emo2id[emolabeldict[pair[0]-1]]] = 1

        y.append(y_tmp)
        l.append(l_tmp)
        el.append(el_tmp)
        x.append(x_tmp)
        sen_len.append(sen_len_tmp)

    y, l, el, x, sen_len, doc_len = map(np.array, [y, l, el, x, sen_len, doc_len])
    for var in ['y', 'l', 'el', 'x', 'sen_len', 'doc_len']:
        print('{}.shape {}'.format( var, eval(var).shape ))
    print('n_cut {}'.format(n_cut))
    print('load data done!\n')
    return doc_id, y, l, el,x,doc_len


def mapcausepos(pred_y, true_y):
    print(true_y)
    pred_y = np.argmax(pred_y, 2) if len(pred_y.shape) > 2 else pred_y
    true_y = np.argmax(true_y, 2) if len(true_y.shape) > 2 else true_y
    pred_y_cause = np.zeros((pred_y.shape[0], pred_y.shape[1], 2), np.int32)
    true_y_cause = np.zeros((pred_y.shape[0], pred_y.shape[1], 2), np.int32)
    pred_y_pos = np.zeros((pred_y.shape[0], pred_y.shape[1], 2), np.int32)
    true_y_pos = np.zeros((pred_y.shape[0], pred_y.shape[1], 2), np.int32)

    pair_tp = 0
    pair_fp = 0
    pair_fn = 0
    for i in range(pred_y.shape[0]):  # should be batch_size
        pred_cause = []
        pred_emo = []
        true_cause = []
        true_emo = []
        pred_pair = []
        true_pair = []
        for j in range(pred_y.shape[1]):  # should be max_doc_len
            if pred_y[i][j] >= 7 and pred_y[i][j] <= 12:
                pred_cause.append((j, pred_y[i][j]-6))
            elif pred_y[i][j] >= 1 and pred_y[i][j] < 7:
                pred_emo.append((j, pred_y[i][j]))
            elif pred_y[i][j] >= 13:
                pred_cause.append((j, pred_y[i][j]-12))
                pred_emo.append((j, pred_y[i][j]-12))

            if true_y[i][j] >= 7 and true_y[i][j] <= 12:
                true_cause.append((j, true_y[i][j]-6))
            elif true_y[i][j] >= 1 and true_y[i][j] < 7:
                true_emo.append((j, true_y[i][j]))
            elif true_y[i][j] >= 13:
                true_cause.append((j, true_y[i][j]-12))
                true_emo.append((j, true_y[i][j]-12))

            pred_y_cause[i][j][int(pred_y[i][j] >= 7)] = 1
            true_y_cause[i][j][int(true_y[i][j] >= 7)] = 1
            pred_y_pos[i][j][int(pred_y[i][j] >= 1 and pred_y[i][j] < 7 or pred_y[i][j] >= 13)] = 1
            true_y_pos[i][j][int(true_y[i][j] >= 1 and true_y[i][j] < 7 or true_y[i][j] >= 13)] = 1

        if len(pred_cause) != 0 and len(pred_emo) == 0 or len(pred_cause) == 0 and len(pred_emo) != 0:
            continue
        else:
            for cause in pred_cause:
                for emo in pred_emo:
                    if cause[1] == emo[1]:
                        pred_pair.append([cause[0], emo[0]])
            for cause in true_cause:
                for emo in true_emo:
                    if cause[1] == emo[1]:
                        true_pair.append([cause[0], emo[0]])
            tmp_tp = 0
            for pair in pred_pair:
                if pair in true_pair:
                    tmp_tp += 1

            pair_tp += tmp_tp
            pair_fp += len(pred_pair) - tmp_tp
            pair_fn += len(true_pair) - tmp_tp

    pred_y_cause = np.argmax(pred_y_cause, 2)
    true_y_cause = np.argmax(true_y_cause, 2)
    pred_y_pos = np.argmax(pred_y_pos, 2)
    true_y_pos = np.argmax(true_y_pos, 2)

    pair_pre = 0.
    pair_rec = 0.
    pair_f1 = 0.
    pair_pre = pair_tp / (pair_tp + pair_fp + 1e-6)
    pair_rec = pair_tp / (pair_tp + pair_fn + 1e-6)
    pair_f1 = 2 * pair_pre * pair_rec / (pair_pre + pair_rec + 1e-6)

    return pred_y_cause, true_y_cause, pred_y_pos, true_y_pos, pair_pre, pair_rec, pair_f1
