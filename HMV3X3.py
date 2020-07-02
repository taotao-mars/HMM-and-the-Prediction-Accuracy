#!/usr/bin/env python
# coding: utf-8

# In[9]:


import random
import numpy
import numpy as np
import pandas as pd

def rand_pick(seq , probabilities):
    x = random.uniform(0 ,1)
    cumprob = 0.0
    for item , item_pro in zip(seq , probabilities):
        cumprob += item_pro
        if x < cumprob:
            break
    return item

state_list = [[0]*15 for i in range(10)]
emission_list = [[0]*15 for i in range(10)]

first_probabilities=[0.333,0.333,0.333]

value_list = [0,1,2]
probabilities = [0,0.9,0.1]

states = ['Cold', 'Hot','Temp']
hidden_states = ['Snow', 'Rain', 'Sunshine']
pi = [0, 0.2, 0.8]
state_space = pd.Series(pi, index=hidden_states, name='states')
a_df = pd.DataFrame(columns=hidden_states, index=hidden_states)
# transition matrices
a_df.loc[hidden_states[0]] = [0.3, 0.3, 0.4]
a_df.loc[hidden_states[1]] = [0.1, 0.45, 0.45]
a_df.loc[hidden_states[2]] = [0.2, 0.3, 0.5]
print("\n HMM transition matrix:\n", a_df)
a = a_df.values

observable_states = states
b_df = pd.DataFrame(columns=observable_states, index=hidden_states)

b_df.loc[hidden_states[0]] = [0.9,0,0.1]
b_df.loc[hidden_states[1]] = [0.6,0.1,0.3]
b_df.loc[hidden_states[2]] = [0.3,0.5,0.2]
print("\n Observable layer  matrix:\n",b_df)
b = b_df.values


def viterbi(pi, a, b, obs):

    nStates = np.shape(b)[0]
    T = np.shape(obs)[0]

    # init blank path
    path = path = np.zeros(T,dtype=int)
    # delta --> highest probability of any path that reaches state i
    delta = np.zeros((nStates, T))
    # phi --> argmax by time step for each state
    phi = np.zeros((nStates, T))

    # init delta and phi
    delta[:, 0] = pi * b[:, obs[0]]
    phi[:, 0] = 0

    #print('\nStart Walk Forward\n')
    # the forward algorithm extension
    for t in range(1, T):
        for s in range(nStates):
            delta[s, t] = np.max(delta[:, t-1] * a[:, s]) * b[s, obs[t]]
            phi[s, t] = np.argmax(delta[:, t-1] * a[:, s])
           # print('s={s} and t={t}: phi[{s}, {t}] = {phi}'.format(s=s, t=t, phi=phi[s, t]))

    # find optimal path
   # print('-'*50)
   # print('Start Backtrace\n')
    path[T-1] = np.argmax(delta[:, T-1])
    for t in range(T-2, -1, -1):
        path[t] = phi[path[t+1], [t+1]]
       # print('path[{}] = {}'.format(t, path[t]))

    return path, delta, phi


count_list=0
while count_list<10:
    fist_state=rand_pick(value_list, first_probabilities)
    state_list[count_list][0]=fist_state
    count_list_element=1
    while count_list_element<15:
        state_list[count_list][count_list_element]=rand_pick(value_list,a_df.values[state_list[count_list][count_list_element-1]])
        count_list_element=count_list_element+1
    count_list=count_list+1

count_list=0
while count_list<10:
    count_list_element=0
    while count_list_element<15:
        emission_list[count_list][count_list_element]=rand_pick(value_list,b_df.values[state_list[count_list][count_list_element]])
        count_list_element=count_list_element+1
    count_list=count_list+1


def calculate_precision(list1,list2):
    length=15
    count=0;
    a=0;
    while a<length:
        if list1[a]==list2[a]:
            count=count+1
        a=a+1
    return count/length

test_index=0
while test_index<10:
    path, delta, phi = viterbi(pi, a, b, np.array(emission_list[test_index]))
    print(calculate_precision(path,state_list[test_index]))
    test_index=test_index+1



