from tqdm import tqdm
import numpy as np
import jieba
import argparse
import untangle
import math
import csv
import os
import re
# avoid divide zero errors
np.seterr(divide='ignore', invalid='ignore')

def parse_args():
    parser = argparse.ArgumentParser(description='Vector Space Model (VSM) with Rocchio Relevance Feedback')
    parser.add_argument('-r', action='store_true', help='If specified, turn on the relevance feedback in your program.')
    parser.add_argument('-i', dest='query_file', required=True, metavar='query-file', help='The input query file.')
    parser.add_argument('-o', dest='ranked_list', required=True, metavar='ranked-list', help='The output ranked list file.')
    parser.add_argument('-m', dest='model_dir', required=True, metavar='model-dir', help='The input model directory, which includes three files: (1) model-dir/vocab.all (2) model-dir/file-list (3) model-dir/inverted-index')
    parser.add_argument('-d', dest='NTCIR_dir', required=True, metavar='NTCIR-dir', help='The directory of NTCIR documents, which is the path name of CIRB010 directory.')
    return parser.parse_args()

def parse_queries(args):
    obj = untangle.parse(args.query_file).xml.topic
    return {
        'topics': obj,
        'number': [x.number.cdata[-3:] for x in obj]
    }

def parse_model(args):
    allFileList = os.listdir(args.model_dir)
    whiteList = ['vocab.all', 'file-list', 'inverted-file']
    model = {}
    for file in whiteList:
        fd = open(os.path.join(args.model_dir, file), 'r')
        model[file] = fd.read().splitlines()
        fd.close()
    return model

def get_doc_length(args, model):
    doc_lengths = []
    total_length = 0
    for i, file_path in enumerate(model['file-list']):
        path = args.NTCIR_dir + file_path
        obj = untangle.parse(path).xml.doc.text
        doc_length = 0
        for o in obj.children:
            for specialChar in '，。；、':
                o.cdata = o.cdata.replace(specialChar, '')
            doc_length += len(o.cdata)
        doc_lengths.append([str(i), str(doc_length)])
        total_length += doc_length
    doc_lengths.append(['avgdoclen', str(total_length / len(model['file-list']))])
    return doc_lengths

"""
def parse_docs(args, model):
    total_length = 0
    with open('doc_attribute.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['doc_id', 'doc_length'])

        for i, file_path in enumerate(model['file-list']):
            path = args.NTCIR_dir + file_path
            obj = untangle.parse(path).xml.doc.text
            doc_length = 0
            for o in obj.children:
                for specialChar in '，。；、':
                    o.cdata = o.cdata.replace(specialChar, '')
                doc_length += len(o.cdata)
            writer.writerow([str(i), str(doc_length)])
            total_length += doc_length
        writer.writerow(['avgdoclen', str(total_length / len(model['file-list']))])

def get_doc_length(args):
    doc_length = 0
    with open('doc_attribute.csv', 'r', newline='') as csvfile:
        doc_length = list(csv.reader(csvfile))

    del doc_length[0]
    return doc_length
"""
"""
extend query vector
"""
def extended_vector(args, topic, vocab, model):
    vocab_bigram = []
    for i in range(0, len(model['inverted-file'])):
        info1 = model['inverted-file'][i].split()
        if len(info1) == 3 and info1[1] == '-1':
            vocab_bigram.append(info1[0] + ' ' + info1[1])

    query_vec = get_query_vector_m(args, topic, vocab)
    extended_vec = np.zeros(len(vocab_bigram))
    for i, ids in enumerate(query_vec['ids']):
        vocab_idx = np.where(vocab_bigram == ids)
        if vocab_idx[0].size != 0:
            extended_vec[vocab_idx[0][0]] += query_vec['weights'][i]

    return {
        'ids': np.array(vocab_bigram),
        'weights': extended_vec
    }

"""
Mode:
0: 1-gram + bi-gram
1: 1-gram
2: bi-gram
"""
def get_query_vector_m(args, topic, vocab, mode=0):
    queries = topic.concepts.cdata
    question = topic.question.cdata[2:]
    title = topic.title.cdata
    narrative = topic.narrative.cdata
    # remove 。
    narrative = narrative.split('。')
    """
    if narrative[-2].find('不相關') != -1:
        del narrative[-2]
    """
    # concate narrative
    narrative = ''.join(x for x in narrative)
    # remove special char
    for specialChar in "，、。「」：（）":
        queries = queries.replace(specialChar, ' ')
        question = question.replace(specialChar, ' ')
        title = title.replace(specialChar, ' ')
        narrative = narrative.replace(specialChar, ' ')
    tmp = question + ' ' + title + ' ' + narrative
    tmp = ' '.join(jieba.cut(tmp, cut_all=False, HMM=True))
    queries = queries + ' ' + tmp
    queries = queries.split()
    # transform into unigram
    queries = [x for x in queries if x != '\n']
    phase_id = []
    for query in queries:
        """
        if mode == 3:
            # jieba
            if len(query) == 1:
                term_ids = []
                term_ids.append(vocab.index(term))
                term_ids.append(-1)
                phase_id.append(term_ids)
        """
        if mode == 0 or mode == 1:
            # 1-gram
            for term in query:
                term_ids = []
                term_ids.append(vocab.index(term))
                term_ids.append(-1)
                phase_id.append(term_ids)
        if mode == 0 or mode == 2 or mode == 3:
            # bi-gram
            term_ids = []
            for term in query:
                term_ids.append(vocab.index(term))
            phase_id.append(term_ids)
            if len(term_ids) > 1:
                for ids in term_ids:
                    phase_id.append([ids, -1])

    query_ids = []
    for x in phase_id:
        for i in range(0, len(x)):
            tmp = []
            if i + 1 < len(x):
                tmp.append(x[i])
                tmp.append(x[i+1])
                query_ids.append(tmp)

    query_ids.sort()

    str_query_ids = []
    # convert list to str
    for tmp in query_ids:
        tmp = " ".join(str(x) for x in tmp)
        str_query_ids.append(tmp)
    query_dict = {x:str_query_ids.count(x) for x in str_query_ids}
    str_query_ids = [x for x in query_dict]
    query_weight = [query_dict[x] for x in query_dict]
    return {
        'ids': np.array(str_query_ids),
        'weights': np.array(query_weight),
    }

def similarity(query_vector, doc_matrix):
    top_x_cnt = 100
    query_vector_normal = math.sqrt(np.sum(np.square(query_vector)))
    doc_matrix_normal = np.sqrt(np.square(doc_matrix).sum(axis=0))
    cosine = np.divide(query_vector.dot(doc_matrix), doc_matrix_normal) / query_vector_normal
    index = cosine.argsort()[::-1]
    top_x_index = []
    for x in index:
        if not np.isnan(cosine[x]):
            top_x_index.append(x)
            top_x_cnt -= 1
            if top_x_cnt == 0:
                break
    return {
        'similarity_x': cosine[top_x_index],
        'top_x': top_x_index
    }

def rocchio(similarity, doc_matrix, query, threshold = 0.25, beta = 0.70, gamma = 0):
    alpha, beta, gamma = 1, beta, gamma
    acc_query_r, acc_query_nr = 0, 0

    acc_query_r = np.zeros(len(query))
    acc_query_nr = np.zeros(len(query))

    # condition = similarity[similarity_x'] >= threshold
    condition = np.array(range(0, 15))
    cuttof_idx = np.where(condition)[0]
    acc_query_r = np.average(doc_matrix[:, np.array(similarity['top_x'])[cuttof_idx]].T, axis=0)
    if np.any(np.isnan(acc_query_r)):
        acc_query_r = np.zeros(len(query))

    # cuttof_idx = np.where(similarity['similarity_x'] < threshold)[0]
    condition = np.array(range(16, 100))
    cuttof_idx = np.where(condition)[0]
    acc_query_nr = np.average(doc_matrix[:, np.array(similarity['top_x'])[cuttof_idx]].T, axis=0)
    if np.any(np.isnan(acc_query_nr)):
        acc_query_nr = np.zeros(len(query))

    return alpha * query + beta * acc_query_r - gamma * acc_query_nr

"""
Implement SVD
"""
def LSI(doc_matrix, ratio = 0.2):
    U, sigma, Vt = np.linalg.svd(doc_matrix, full_matrices=False)
    eigen_vectors = np.zeros((U.shape[1], Vt.shape[0]))
    i, j = np.indices(eigen_vectors.shape)
    eigen_vectors[i == j] = sigma
    eigen_vectors[:, int(len(sigma) * ratio):] = 0
    # eigen_vectors[:, 50:] = 0
    doc_matrix = np.dot(np.dot(U, eigen_vectors), Vt)
    return doc_matrix

"""
weighting: idf * tf
Mode:
0: baseline - tf only (without tf normalization and idf)
1: tf normalization
2: idf
3: tf normalization + idf
"""
def get_doc_matrix_idf_tf(args, model, query, doc_length, mode=3, isLSI=True, ratio=0.2):
    cnt = 0
    phase = ''
    flag = True
    query_id = -1
    avgdoclen = int(float(doc_length[-1][1]))
    doc_matrix = np.zeros((len(query['ids']), len(model['file-list'])))
    tf, qtf, N, df = 0, 0, len(model['file-list']), 0
    for row in model['inverted-file']:
        if cnt == 0 and flag:
            row_info = row.split()
            phase = row_info[0] + ' ' + row_info[1]
            cnt = int(row_info[2])
            result = np.where(query['ids'] == phase)
            if result[0].size != 0:
                df = int(row_info[2])
                query_id = result[0][0]
                flag = False
        else:
            if flag:
                cnt -= 1
            else:
                row_info = row.split()
                tf, qtf = int(row_info[1]), query['weights'][query_id]
                k1, b, k3 = 1, 0.75, 0
                tf_normalized = tf
                idf = 1
                if mode == 1:
                    tf_normalized = ((k1 + 1) * tf) / (k1 * (1 - b + b * int(doc_length[int(row_info[0])][1]) / avgdoclen) + tf)
                elif mode == 2:
                    idf = math.log((N - df + 0.5) / (df + 0.5))
                elif mode == 3:
                    tf_normalized = ((k1 + 1) * tf) / (k1 * (1 - b + b * int(doc_length[int(row_info[0])][1]) / avgdoclen) + tf)
                    idf = math.log((N - df + 0.5) / (df + 0.5))

                weight = idf * tf_normalized * (k3 + 1) * qtf / (k3 + qtf)
                doc_matrix[query_id][int(row_info[0])] += weight
                cnt -= 1
                flag = cnt == 0
    if isLSI:
        doc_matrix = LSI(doc_matrix, ratio)

    return doc_matrix

def accuracy(args):
    we = []
    ans = []
    with open(args.ranked_list, 'r', newline='') as csvfile:
        we = list(csv.reader(csvfile))

    with open('./queries/ans_train.csv', 'r', newline='') as csvfile:
        ans = list(csv.reader(csvfile))

    total = 0
    # MAP, Mean Average Precision
    for i in range(1, len(we)):
        precision_accu = 0
        correct = 0
        we_tmp = we[i][1].split()
        ans_tmp = ans[i][1].split()
        for j in range(0, len(we_tmp)):
            if we_tmp[j] in ans_tmp:
                correct += 1
                precision_accu += correct / (j + 1)
        total += precision_accu / len(ans_tmp)
    MAP = total / (len(we) - 1)
    return MAP

def write_csv(args, model, indices, number):
    with open(args.ranked_list, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['query_id', 'retrieved_docs'])

        for i, top_idx in enumerate(indices):
            doc_list = []
            for x in np.array(model['file-list'])[top_idx]:
                tmp = x.split('/')
                doc_list.append(tmp[len(tmp) - 1].lower())
            writer.writerow([number[i], " ".join(x for x in doc_list)])

def run_query(args, topic, model, doc_length, representation_method=3, weight_method=3, isLSI=False, LSI_ratio=0.2, rocchio_time=2, gamma=0.15, beta=0.75):
    cnt = 0
    if args.r:
        cnt = rocchio_time
    # query = extended_vector(args, topic, model['vocab.all'], model)
    query = get_query_vector_m(args, topic, model['vocab.all'], representation_method)
    doc_matrix = get_doc_matrix_idf_tf(args, model, query, doc_length, mode=weight_method, isLSI=isLSI, ratio=LSI_ratio)
    sim = similarity(query['weights'], doc_matrix)
    while (cnt):
        query['weights'] = rocchio(sim, doc_matrix, query['weights'], gamma=gamma, beta=beta)
        # doc_matrix = get_doc_matrix_idf_tf(args, model, query, doc_length)
        sim = similarity(query['weights'], doc_matrix)
        cnt -= 1

    return sim['top_x']

if __name__ == '__main__':
    args = parse_args()
    query_topics = parse_queries(args)
    model = parse_model(args)
    # parse_docs(args, model)
    doc_length = get_doc_length(args, model)
    indices = []
    MAP = 0
    indices = []
    progress = tqdm(total=len(query_topics['topics']))
    for topic in query_topics['topics']:
        top_x_index = run_query(args, topic, model, doc_length, isLSI=True,beta=0.7, gamma=0.4)
        indices.append(top_x_index)
        progress.update(1)
    write_csv(args, model, indices, query_topics['number'])
    if args.query_file == 'queries/query-train.xml' or args.query_file == '../queries/query-train.xml':
        MAP = accuracy(args)
    print(MAP)

