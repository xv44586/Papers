# -*- coding: utf-8 -*-
# @Date    : 2019/12/11
# @Author  : mingming.xu
# @Email   : mingming.xu@zhaopin.com
# @File    : load_data.py
import os
import pickle

import itertools

from utils.loader import load_sentences, char_mapping, update_tag_scheme, augment_with_pretrained, tag_mapping, \
    prepare_dataset


def get_data_path(file_name):
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(cur_dir, 'data', file_name)


# load data sets
def load(train_file='example.train', dev_file='example.dev', test_file='example.test', lower=True,
         zeros=True, tag_schema='iobes', map_file='map.pkl', pre_emb=True, emb_file='wiki_100.utf8'):

    train_file = get_data_path(train_file)
    dev_file = get_data_path(dev_file)
    test_file = get_data_path(test_file)
    map_file = get_data_path(map_file)
    emb_file = get_data_path(emb_file)

    train_sentences = load_sentences(train_file, lower, zeros)
    dev_sentences = load_sentences(dev_file, lower, zeros)
    test_sentences = load_sentences(test_file, lower, zeros)

    # Use selected tagging scheme (IOB / IOBES)
    update_tag_scheme(train_sentences, tag_schema)
    update_tag_scheme(test_sentences, tag_schema)

    # create maps if not exist
    if not os.path.isfile(map_file):
        # create dictionary for word
        if pre_emb:
            dico_chars_train = char_mapping(train_sentences, lower)[0]
            dico_chars, char_to_id, id_to_char = augment_with_pretrained(
                dico_chars_train.copy(),
                emb_file,
                list(itertools.chain.from_iterable(
                    [[w[0] for w in s] for s in test_sentences])
                )
            )
        else:
            _c, char_to_id, id_to_char = char_mapping(train_sentences, lower)

        # Create a dictionary and a mapping for tags
        _t, tag_to_id, id_to_tag = tag_mapping(train_sentences)
        with open(map_file, "wb") as f:
            pickle.dump([char_to_id, id_to_char, tag_to_id, id_to_tag], f)
    else:
        with open(map_file, "rb") as f:
            char_to_id, id_to_char, tag_to_id, id_to_tag = pickle.load(f)

    # prepare data, get a collection of list containing index
    train_data = prepare_dataset(
        train_sentences, char_to_id, tag_to_id, lower
    )
    dev_data = prepare_dataset(
        dev_sentences, char_to_id, tag_to_id, lower
    )
    test_data = prepare_dataset(
        test_sentences, char_to_id, tag_to_id, lower
    )
    print("%i / %i / %i sentences in train / dev / test." % (
        len(train_data), 0, len(test_data)))

    return train_data, dev_data, test_data, char_to_id, tag_to_id, id_to_char, id_to_tag


if __name__ == '__main__':

    train,_,_ = load()
    print(train[:1])