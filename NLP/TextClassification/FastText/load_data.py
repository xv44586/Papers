# -*- coding: utf-8 -*-
# @Date    : 2019/9/12
# @Author  : mingming.xu
# @File    : load_data.py
'''
fix np.load bug
'''
import json

import numpy as np
from six.moves import zip  # pylint: disable=redefined-builtin

from tensorflow.python.keras._impl.keras.utils.data_utils import get_file


def load_data(path='imdb.npz',
              num_words=None,
              skip_top=0,
              maxlen=None,
              seed=113,
              start_char=1,
              oov_char=2,
              index_from=3):
    """Loads the IMDB dataset.

  Arguments:
      path: where to cache the data (relative to `~/.keras/dataset`).
      num_words: max number of words to include. Words are ranked
          by how often they occur (in the training set) and only
          the most frequent words are kept
      skip_top: skip the top N most frequently occurring words
          (which may not be informative).
      maxlen: truncate sequences after this length.
      seed: random seed for sample shuffling.
      start_char: The start of a sequence will be marked with this character.
          Set to 1 because 0 is usually the padding character.
      oov_char: words that were cut out because of the `num_words`
          or `skip_top` limit will be replaced with this character.
      index_from: index actual words with this index and higher.

  Returns:
      Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.

  Raises:
      ValueError: in case `maxlen` is so low
          that no input sequence could be kept.

  Note that the 'out of vocabulary' character is only used for
  words that were present in the training set but are not included
  because they're not making the `num_words` cut here.
  Words that were not seen in the training set but are in the test set
  have simply been skipped.
  """
    path = get_file(
        path, origin='https://s3.amazonaws.com/text-datasets/imdb.npz')
    f = np.load(path, allow_pickle=True)
    x_train = f['x_train']
    labels_train = f['y_train']
    x_test = f['x_test']
    labels_test = f['y_test']
    f.close()

    np.random.seed(seed)
    np.random.shuffle(x_train)
    np.random.seed(seed)
    np.random.shuffle(labels_train)

    np.random.seed(seed * 2)
    np.random.shuffle(x_test)
    np.random.seed(seed * 2)
    np.random.shuffle(labels_test)

    xs = np.concatenate([x_train, x_test])
    labels = np.concatenate([labels_train, labels_test])

    if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        xs = [[w + index_from for w in x] for x in xs]

    if maxlen:
        new_xs = []
        new_labels = []
        for x, y in zip(xs, labels):
            if len(x) < maxlen:
                new_xs.append(x)
                new_labels.append(y)
        xs = new_xs
        labels = new_labels
        if not xs:
            raise ValueError('After filtering for sequences shorter than maxlen=' +
                             str(maxlen) + ', no sequence was kept. '
                                           'Increase maxlen.')
    if not num_words:
        num_words = max([max(x) for x in xs])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        xs = [[oov_char if (w >= num_words or w < skip_top) else w for w in x]
              for x in xs]
    else:
        new_xs = []
        for x in xs:
            nx = []
            for w in x:
                if skip_top <= w < num_words:
                    nx.append(w)
            new_xs.append(nx)
        xs = new_xs

    x_train = np.array(xs[:len(x_train)])
    y_train = np.array(labels[:len(x_train)])

    x_test = np.array(xs[len(x_train):])
    y_test = np.array(labels[len(x_train):])

    return (x_train, y_train), (x_test, y_test)


def load_data_ngram():
    return (['Access the notebooks in the notebooks directory in this repo',
            'You can run these notebook on Google Colab (recommended) or on your local machine']*20, [1, 0]*20),(['Rename this new notebook by removing the Copy of part in the title'], [1])