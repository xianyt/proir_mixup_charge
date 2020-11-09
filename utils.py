#  -*- coding: utf-8 -*-
import csv
import math
import random
import re

import numpy as np
import torch
from tqdm import tqdm

SPECIAL_WORDS = ['[PAD]', '[UNK]']


def clean_str(string):
    """
    Tokenization and cleaning string
    """
    string = string.strip().strip('"')
    string = re.sub(r"[^A-Za-z0-9(),;!?.\'`\u4E00-\u9FA5。，；？]", " ", string)

    string = re.sub(r"\d+年", "x年", string)
    string = re.sub(r"\d+月", "x月", string)
    string = re.sub(r"\d+月份", "x月份", string)
    string = re.sub(r"\d+日", "x日", string)
    string = re.sub(r"\d+时", "x时", string)
    string = re.sub(r"\d+\s?点", "x点", string)
    string = re.sub(r"\d+分", "x分", string)
    string = re.sub(r"\d+\s?分许", "x分许", string)
    string = re.sub(r"\d+\s?万", "x万", string)
    string = re.sub(r"\d+\s?元", "x元", string)
    string = re.sub(r"\d+\s?吨", "x吨", string)
    string = re.sub(r"\d+(\s?\.\s?\d+)\s?克", "x克", string)
    string = re.sub(r"\d+(\s?\.\s?\d+)\s?千克", "x千克", string)
    string = re.sub(r"\d+(\s?\.\s?\d+)\s?公斤", "x公斤", string)
    string = re.sub(r"\d+(\s?\.\s?\d+)\s?米", "x米", string)
    string = re.sub(r"\d+(\s?\.\s?\d+)\s?公里", "x公里", string)
    string = re.sub(r"第\s?\d+", "第x", string)

    string = re.sub(r" \d+(\.\d\+)? ", " <num> ", string)

    string = re.sub(r"[,;)(!'.]", " ", string)

    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\.", " \. ", string)
    string = re.sub(r"\"", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()


def post_clean(tokens):
    def _clean(tok):
        tok = re.sub(r"^\d+年$", "x年", tok)
        tok = re.sub(r"^\d+月$", "x月", tok)
        tok = re.sub(r"^\d+月份$", "x月份", tok)
        tok = re.sub(r"^\d+日$", "x日", tok)
        tok = re.sub(r"^\d+$", "<num>", tok)
        tok = re.sub(r"^\d+万$", "x万", tok)
        return tok

    return [_clean(t) for t in tokens]


def build_vocab(text_files, token_min_count, multi_label, field_delimiter=',', label_delimiter=';'):
    """
    build vocab from data file
    :param multi_label:
    :param text_files: input train data file
    :param token_min_count:
    :param field_delimiter: delimiter for CSV fields
    :param label_delimiter: delimiter for labels
    :return: token2id, label2id
    """
    csv.field_size_limit(500000)
    token_count = {}
    label_count = {}
    with open(text_files, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=field_delimiter)
        for row in tqdm(csv_reader):
            labels = row[0].strip()
            labels = [la.strip() for la in labels.split(label_delimiter)]
            labels = [la for la in labels if len(la) > 0]

            if not multi_label and len(labels) > 1:
                continue

            for la in labels:
                label_count[la] = label_count.get(la, 0) + 1

            text = " ".join([t.strip() for t in row[1:]])
            text = clean_str(text)
            tokens = re.split(r'\s+', text)
            tokens = set(tokens)

            for tok in tokens:
                token_count[tok] = token_count.get(tok, 0) + 1

    # remove token when count < token_min_count
    token_count = {k: v for k, v in token_count.items() if v >= token_min_count}

    # sort token by count descent
    token_vocab = sorted(token_count.items(), key=lambda kv: -kv[1])
    token_vocab = SPECIAL_WORDS + [tok for tok, _ in token_vocab]

    return token_vocab  # , label2id


def count_label(text_file, field_delimiter=',', label_delimiter=';'):
    label_count = {}
    with open(text_file, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=field_delimiter)
        for row in tqdm(csv_reader):
            labels = row[0].strip()
            labels = [la.strip() for la in labels.split(label_delimiter)]

            for la in labels:
                label_count[la] = label_count.get(la, 0) + 1

    return label_count


def token_to_id(text_file, token2id, label2id, multi_label, encoder, max_seq_len, field_delimiter=',',
                label_delimiter=';'):
    """
    convert csv file into vocabulary index file
    :param text_file: text data file
    :param label2id:
    :param token2id: vocabulary file
    :param field_delimiter: delimiter in data file
    :param label_delimiter: delimiter in data file
    :return:
    """
    data = []
    with open(text_file, 'r', encoding='utf-8') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=field_delimiter)
        unk_id = token2id.get('[UNK]') if encoder is None else None
        for row in tqdm(csv_reader):
            labels = row[0].strip()
            labels = [la.strip() for la in labels.split(label_delimiter)]
            label_ids = [label2id.get(la) for la in labels if len(la) > 0 and la in label2id]

            if not multi_label and len(label_ids) > 1:
                continue

            if len(label_ids) == 0:
                continue

            text = " ".join([t.strip() for t in row[1:]])
            text = clean_str(text)
            tokens = re.split(r'\s+', text)
            if encoder is None:
                token_ids = [token2id.get(t, unk_id) for t in tokens]
            else:
                max_length = max_seq_len - encoder.num_special_tokens_to_add()
                token_ids = encoder(tokens, max_length=max_length, truncation=True, is_pretokenized=True)['input_ids']

            if len(token_ids) == 0:
                print("empty data.")
                continue

            data.append((label_ids, token_ids))

    return data


def pad_seq(seq, max_len, pad=0, pad_left=False):
    """
    padding or truncate sequence to fixed length
    :param seq: input sequence
    :param max_len: max length
    :param pad: padding token id
    :return: padded sequence
    """
    if max_len < len(seq):
        seq = seq[:max_len]
    elif max_len > len(seq):
        padding = [pad] * (max_len - len(seq))
        if pad_left:
            seq = padding + seq
        else:
            seq = seq + padding
    return seq


def load_embeddings(token2id, embed_file, embed_dim):
    """
    Load pre-trained embeddings
    :param token2id: vocabulary
    :param embed_file: pre-trained embedding file
    :param embed_dim: dimension of pre-trained embeddings
    :return: pre-trained word embeddings
    """

    # print("loading pre-trained word vectors form %s" % embed_file)
    word2embed = {}
    with open(embed_file, encoding='utf8', errors='ignore') as f:
        for line in tqdm(f):
            values = re.split(r"\s+", line.strip())
            if len(values) == embed_dim + 1:
                word = values[0]
                embed = np.asarray(values[1:], dtype='float32')
                word2embed[word] = embed

    # prepare embedding matrix
    num_token = len(token2id)
    pad_id = token2id.get('[pad]')
    count = 0
    weights = np.random.rand(num_token, embed_dim)
    for word, i in token2id.items():
        if word == pad_id:
            embedding_vector = np.zeros(shape=(1, embed_dim), dtype='float32')
        else:
            embedding_vector = word2embed.get(word)

        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            weights[i] = embedding_vector
            count = count + 1

    print("%d vectors loaded." % count)
    return weights


def data_generator(data, batch_size, repeat=True):
    batch_num = math.ceil(len(data) / batch_size)
    field_num = len(data[0])
    while True:
        shuffled_idx = [i for i in range(len(data))]
        random.shuffle(shuffled_idx)
        data = [data[i] for i in shuffled_idx]
        batch_idx = 0
        while batch_idx < batch_num:
            offset = batch_idx * batch_size
            batch_data = data[offset: offset + batch_size]

            batch = tuple([row[f] for row in batch_data] for f in range(field_num))
            yield batch

            batch_idx += 1
        if repeat:
            continue
        else:
            break


def width(text):
    return sum([2 if '\u4E00' <= c <= '\u9FA5' else 1 for c in text])


def print_table(tab):
    col_width = [max(width(x) for x in col) for col in zip(*tab)]
    print("+-" + "-+-".join("{:-^{}}".format('-', col_width[i]) for i, x in enumerate(tab[0])) + "-+")
    for line in tab:
        print("| " + " | ".join("{:{}}".format(x, col_width[i]) for i, x in enumerate(line)) + " |")
    print("+-" + "-+-".join("{:-^{}}".format('-', col_width[i]) for i, x in enumerate(tab[0])) + "-+")


def parameters_string(module):
    lines = [
        "",
        "List of model parameters:",
        "=========================",
    ]

    row_format = "{name:<40} {shape:>20} ={total_size:>12,d}"
    params = list(module.named_parameters())
    for name, param in params:
        lines.append(row_format.format(
            name=name,
            shape=" * ".join(str(p) for p in param.size()),
            total_size=param.numel()
        ))
    lines.append("=" * 75)
    lines.append(row_format.format(
        name="all parameters",
        shape="sum of above",
        total_size=sum(int(param.numel()) for name, param in params)
    ))
    lines.append("")
    return "\n".join(lines)


def create_tokenizer(cfg):
    # if cfg.tokenizer == 'ltp':
    #     from ltp import LTP
    #     ltp = LTP(device=cfg.device)
    #
    #     def _ltp_tok(text):
    #         tokens, _ = ltp.seg(text)
    #         return tokens
    #
    #     return _ltp_tok, None, None

    if cfg.tokenizer == '':

        def _space_tok(text):
            tokens = [t.split(" ") for t in text]
            return tokens

        return _space_tok, None, None

    else:
        raise Exception("unsupported tokenizer")


def pre_clean(string):
    string = re.sub(r"\s\s+", " ", string)
    string = re.sub(r"[?]", "", string)
    string = re.sub(r">\\n(\s+)?", " ", string)
    string = re.sub(r"\d+(\.\d+)?元", r"x元", string)
    string = re.sub(r"\d+(\.\d+)?万", r"x万", string)

    string = re.sub(r"[^A-Za-z0-9(),;!?.\'`\u4E00-\u9FA5。，；？、 ]", "", string)

    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"\.", " \. ", string)
    string = re.sub(r"\"", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)

    return string.strip().lower()


def tokenize_file(tokenize, in_file, out_file):
    csv.field_size_limit(500 * 1024 * 1024)
    with open(in_file, 'r', encoding='utf-8') as f_in:
        with open(out_file, 'w', encoding='utf-8') as f_out:
            csv_reader = csv.reader(f_in, delimiter=',')
            csv_writer = csv.writer(f_out, quoting=csv.QUOTE_ALL)

            for row in tqdm(csv_reader):
                label = row[0].strip()
                text = "。".join(row[1:])
                text = pre_clean(text)
                sents = re.sub(r'(([;?；？。,!])|(\. ))', r'\1\n', text)
                sents = sents.split('\n')
                sents = [t.strip()[0:500] for t in sents if len(t) > 0]
                sents = sents[:10]
                # word only counts once in a document
                sents = tokenize(sents)
                tokens = []
                for s in sents:
                    s = post_clean(s)
                    tokens.extend(s)

                csv_writer.writerow([label, " ".join(tokens)])


def max_pooling(inp, mask, batch_first=False):
    pool_dim = 1 if batch_first else 0
    max_pool_mask = ((1 - mask) * 1e10).unsqueeze(-1)
    outp, _ = torch.max(inp - max_pool_mask, dim=pool_dim)
    return outp


def mean_pooling(inp, inp_len, mask, batch_first=False):
    pool_dim = 1 if batch_first else 0
    mean_pool_mask = mask.unsqueeze(-1)

    inp_sum = torch.sum(inp * mean_pool_mask, dim=pool_dim)
    outp = inp_sum / inp_len.unsqueeze(-1)
    return outp


