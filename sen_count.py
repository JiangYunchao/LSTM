#统计description中的句子数目分布

import sys
sys.path.insert(0, '/home/jiang/polyvore')
from utils.data_utils import load_pkl
from utils.data_utils import DataFile
import numpy as np
import nltk
import nltk.data
import matplotlib.pyplot as plt
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

def load_data(state):
    datafile = DataFile('/data/polyvore/processed/tuples',
                        '/data/polyvore/processed/image_list')
    image_list = datafile.image_list
    fashion_sets, fashion_items = load_pkl('/data/polyvore/processed/pickles')
    positive_tuple, negative_tuples = datafile.get_tuples(state,repeated=False)
    return image_list, positive_tuple, negative_tuples, fashion_items

def set_dic(set_tuple, image_list, fashion_items):
    all_items = set()
    text_list = []
    for i, tpl in enumerate(set_tuple):
        for j in range(1, 4):
            image_name = image_list[j-1][tpl[j]]
            all_items.add(image_name)
    for name in all_items:
        item_info = fashion_items[name]
        text_list.append(tokenizer.tokenize(item_info['text']))

    return all_items, text_list
    
    
if __name__ == '__main__':
  image_list, positive_tuple, negative_tuples, fashion_items = load_data('train')
  all_items, text_list = set_dic(positive_tuple, image_list, fashion_items)
  %matplotlib inline 
  count_list = [0]*10
  for text in text_list:
      if len(text) >=9:
          count_list[9] += 1
      else:
          count_list[len(text)] += 1
  x = [i for i in range(0,10)]
  y = count_list
  plt.plot(x,y)
  plt.xlabel('sentences num in a text')
  plt.ylabel('item number')
  print(count_list)
  plt.show()

#result
#[16722, 57512, 34495, 29876, 24175, 24459, 16833, 15044, 12902, 31639]
