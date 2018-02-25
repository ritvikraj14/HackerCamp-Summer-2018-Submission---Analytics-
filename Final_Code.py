
# coding: utf-8

# In[194]:



# import different relevant libraries

import numpy as np
import pandas as pd

# pd.set_option('max_columns',200)

pd.set_option('display.height', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
import difflib
from collections import defaultdict
import jellyfish
import re
import fuzzy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer


# In[195]:


# read csv file

data = pd.read_csv('Deduplication Problem - Sample Dataset.csv')


# In[196]:


data.head()


# In[197]:


data.describe()


# In[198]:


df = pd.DataFrame()


# In[199]:


data = data[['ln','fn']]


# In[200]:


data


# In[201]:


df['Full_Name'] = data.ln + ' ' + data.fn


# In[202]:


# this function is used to clean string 

def string_clean(input_str, normalized=False, ignore_list=[]):

    for ignore_str in ignore_list:
        input_str = re.sub(r'{0}'.format(ignore_str), '', input_str,
                           flags=re.IGNORECASE)

    if normalized is True:
        input_str = input_str.strip().lower()

        # clean special chars and extra whitespace

        input_str = re.sub("\W", '', input_str).strip()

    return input_str


# this function is used to find the similarity between two names

def find_similarity_score(
    first_str,
    second_str,
    normalized=False,
    ignore_list=[],
    ):

    first_str = string_clean(first_str, normalized=normalized,
                             ignore_list=ignore_list)
    second_str = string_clean(second_str, normalized=normalized,
                              ignore_list=ignore_list)
    match_ratio = (difflib.SequenceMatcher(None, first_str,
                   second_str).ratio()
                   + jellyfish.jaro_winkler(unicode(first_str),
                   unicode(second_str))) / 2.0
    return match_ratio


# In[203]:


find_similarity_score('JAMES MICHAELSON JR', 'ROBERT MICHAELSON JR',
                      normalized=True, ignore_list=[])


# In[204]:


find_similarity_score('JAMES MICHAELSON JR', 'GALETICH JR ADDISON',
                      normalized=True, ignore_list=[])


# In[205]:


matrix = pd.DataFrame(columns=df.Full_Name)
matrix


# In[206]:


matrix['Full_Name'] = df.Full_Name
matrix


# In[207]:


matrix.set_index('Full_Name', drop=True, inplace=True)


# In[208]:


matrix


# In[209]:


# make a similarity matrix with similarity score

for i in matrix.index:
    for j in matrix.columns:
        matrix.loc[i, j] = find_similarity_score(i, j, ignore_list=['JR'
                ])


# In[210]:


matrix


# In[211]:


# set threshold

t_matrix = matrix.iloc[:, :] > 0.85


# In[212]:


t_matrix


# In[213]:


t_matrix.describe()


# In[214]:


# remove absolute same strings

rt_matrix = t_matrix[~t_matrix.index.duplicated(keep='last')]


# In[215]:


rt_matrix.head()


# In[216]:


rt_matrix.describe()


# In[217]:


# find similar names for each patient name

dic = defaultdict(list)
for i in rt_matrix.index:
    for j in rt_matrix.columns:
        if rt_matrix.loc[i, j].all():
            if j != i and j not in dic[i]:
                dic[i].append(j)


# In[218]:


p = []
for i in dic.keys():
    p.append([i, dic[i]])


# In[219]:


f = pd.DataFrame(p)


# In[220]:


f = f.rename(columns={0: 'Name', 1: 'Similar Names'})


# In[221]:


f = f.set_index('Name')


# In[222]:


f['Similar Names'] = f['Similar Names'].apply(lambda x: ' , '.join(x))


# In[223]:


# Ambiguos patients names with their similars names

f


# In[224]:


f.to_csv('submission.csv', index=True)

