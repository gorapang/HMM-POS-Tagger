from collections import defaultdict, Counter
from math import log
import numpy as np
import random
from itertools import tee

EPSILON = 1e-5


def baseline(train, test):
    '''
    Implementation for the baseline tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words, use utils.strip_tags to remove tags from data)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    word_tag = {} # 각 단어의 품사 빈도를 저장하는 딕셔너리
    output = []
    tag_cnt = Counter() # 품사의 전체 빈도를 저장하는 counter 객체

    # training data를 사용해 각 단어의 품사 빈도 계산
    for sentence in train:
        for word, tag in sentence:
            if word not in word_tag:
                word_tag[word] = Counter()
            word_tag[word][tag] += 1
            tag_cnt[tag] += 1
    
    # 각 단어에 대해 가장 높은 빈도로 등장한 태그를 저장하는 딕셔너리
    most_common_tags_for_words = {} 
    for word, tag_cnt in word_tag.items():
         most_common_tag = tag_cnt.most_common(1)[0][0]
         most_common_tags_for_words[word] = [most_common_tag]
         for tag, cnt in tag_cnt.items():
            if tag != most_common_tag and cnt == tag_cnt[most_common_tag]:
                most_common_tags_for_words[word].append(tag)

    # test data에 대한 품사 태깅
    for sentence in test:
        tag_pred = []
        for word in sentence:
            if word in word_tag:
                # 각 단어에 대해 가장 높은 빈도로 등장한 tag를 선택
                chosen_tag = random.choice(most_common_tags_for_words[word])
                tag_pred.append((word, chosen_tag))
            else: 
                # unseen data는 training data에서 가장 높은 빈도를 보인 tag를 선택
                max_tags = [key for key, value in tag_cnt.items() if value == max(tag_cnt.values())]
                chosen_tag = random.choice(max_tags)
                tag_pred.append((word, chosen_tag))
        output.append(tag_pred)

    return output



class node: # 최적 경로를 찾는 데 사용할 노드
    def __init__(self, p, parent, word_, tag_):
        self.probability = p
        self.backpointer = parent
        self.word = word_
        self.tag = tag_



def count_tags_and_pairs(train):
    tag_cnt = Counter()
    word_tag_cnt = defaultdict(Counter)
    tag_initial_cnt = Counter()

    # tag_cnt, word_tag_cnt, tag_initial_cnt 계산 
    for sentence in train:
        prev_tag = 'START'
        for index, (word, tag) in enumerate(sentence):
            tag_cnt[tag] += 1
            word_tag_cnt[word][tag] += 1
            prev_tag = tag
            if index == 0:
                tag_initial_cnt[tag] += 1

    # tag_pair_cnt 계산
    def pairwise(iterable):
        a, b = tee(iterable)
        next(b, None)
        return zip(a, b)

    tag_pair_cnt = {prev: Counter() for prev in tag_cnt}  # 초기화
    temp_tag_counter = Counter()

    for sentence in train:
        tag_list = [tag for _, tag in sentence]  # 각 문장에서 태그만 리스트로 저장
        # 인접한 태그 둘 간의 전이 횟수 계산
        for prev, next_tag in pairwise(tag_list):
            tag_pair_cnt[prev].update([next_tag])
            temp_tag_counter.update([prev])

    return tag_cnt, word_tag_cnt, tag_initial_cnt, tag_pair_cnt, temp_tag_counter


#laplace smoothing과 log scaling를 적용한 initial, transition, emission probability를 계산해 반환하는 함수
def calculate_probabilities(tag_initial_cnt, tag_pair_cnt, temp_tag_counter, word_tag_cnt, tag_cnt, alpha, train, hapax_probabilities):
    initial_prob = dict()
    transition_prob = dict()
    emission_prob = dict()

    # Initial probability 계산
    for tag in tag_pair_cnt:
        initial_prob[tag] = np.log((tag_initial_cnt[tag] + alpha) / (len(train) + alpha * len(tag_cnt)))

    # Transition probability 계산
    for prev_tag in tag_pair_cnt:
        transition_prob[prev_tag] = dict()
        for next_tag in tag_pair_cnt:
            transition_prob[prev_tag][next_tag] = np.log((tag_pair_cnt[prev_tag][next_tag] + alpha) / (temp_tag_counter[prev_tag] + alpha * hapax_probabilities[tag] * len(tag_cnt)))

    # Emission probability 계산
    for word in word_tag_cnt:
        emission_prob[word] = dict()
        for tag in tag_pair_cnt:
            # 성능 향상을 위해 hapax probability를 곱하도록 수정함
            emission_prob[word][tag] = np.log((word_tag_cnt[word][tag] + alpha * hapax_probabilities[tag]) / (tag_cnt[tag] + (alpha * (len(word_tag_cnt) + 1))))

    return initial_prob, transition_prob, emission_prob


def viterbi(train, test):
    '''
    Implementation for the viterbi tagger.
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    output = []    

    #--------------- 1. Count tag, tag pairs, (word, tag) pairs ---------------#

    tag_cnt = Counter() # training data에서 각 태그의 등장 횟수를 셈
    tag_pair_cnt = defaultdict(Counter) # 한 태그 뒤에 다른 태그가 나오는 횟수를 저장 (transition)
    word_tag_cnt = defaultdict(Counter) # 특정 단어가 특정 품사로 등장한 횟수를 저장
    tag_initial_cnt = Counter() # 각 태그가 문장의 시작에 등장한 횟수를 셈

    tag_cnt, word_tag_cnt, tag_initial_cnt, tag_pair_cnt, temp_tag_counter = count_tags_and_pairs(train)
    word_tag_cnt['UNKNOWN'] = Counter()

    # viterbi 성능 향상을 위한 변수들 
    onetime_cnt = Counter() # 1번만 나타난 단어를 셈

    for tag in tag_cnt:
        for word in word_tag_cnt:    
            if word_tag_cnt[word][tag] == 1:
                onetime_cnt[tag] += 1



    #--------------- 2. smoothed probabilities 계산 & 3. log scaling 적용 ---------------#
    initial_prob = dict() # 각 태그가 문장의 시작에 등장할 확률을 저장하는 딕셔너리
    transition_prob = dict() # 한 태그에서 다른 태그로 전이할 확률을 저장하는 2d 딕셔너리
    emission_prob = dict() # 한 태그가 특정 단어에 매칭될 확률을 저장하는 2d 딕셔너리

    alpha = EPSILON   

    # 성능 향상을 위해 hapax prob 추가
    onetime_prob = dict() # 1번만 등장한 단어가 특정 태그에 매칭될 확률 저장
    for tag in tag_cnt:
        probability = (onetime_cnt[tag] + alpha) / (len(onetime_cnt) + alpha * len(tag_cnt))
        onetime_prob[tag] = probability

    initial_prob, transition_prob, emission_prob = calculate_probabilities(tag_initial_cnt, tag_pair_cnt, temp_tag_counter, word_tag_cnt, tag_cnt, alpha, train, onetime_prob)


    #--------------- 4. 최적의 경로를 역추적 ---------------#
    for sentence in test:
        previous_node_list = []
        for index, word in enumerate(sentence):
            current_node_list = []

            if word in word_tag_cnt: # training data에 존재하는 단어
                if index == 0: # 문장의 첫 단어인 경우
                    for tag in word_tag_cnt[word]:
                        # initial 확률과 emission 확률 사용
                        p = initial_prob[tag] + emission_prob[word][tag]
                        current_node_list.append(node(p, None, word, tag)) # 노드를 리스트에 추가
                else:
                    for tag in word_tag_cnt[word]:
                        # emission, transition 확률을 사용해 노드의 확률 계산
                        p = emission_prob[word][tag] + max([node.probability + transition_prob[node.tag][tag] for node in previous_node_list])
                        # 이전 단계의 노드들 중 최대 확률을 가진 노드를 찾아 리스트에 추가
                        max_node = max(previous_node_list, key=lambda node_: node_.probability + transition_prob[node_.tag][tag]) 
                        current_node_list.append(node(p, max_node, word, tag)) 

            else: # unseen word인 경우
                if index == 0: # 문장의 첫 단어인 경우
                    for tag in transition_prob:
                        #UNKNOWN에 대한 emission 확률과 intial 확률을 사용
                        p = initial_prob[tag] + emission_prob['UNKNOWN'][tag]
                        current_node_list.append(node(p, None, word, tag)) # 노드를 리스트에 추가
                else:
                    for tag in transition_prob:
                        # emission, transition 확률을 사용해 노드의 확률 계산
                        p = emission_prob['UNKNOWN'][tag] + max([node.probability + transition_prob[node.tag][tag] for node in previous_node_list])
                        # 이전 단계의 노드들 중 최대 확률을 가진 노드를 찾아 리스트에 추가
                        max_node = max(previous_node_list, key=lambda node_: node_.probability + transition_prob[node_.tag][tag]) 
                        current_node_list.append(node(p, max_node, word, tag)) 
 
            previous_node_list = current_node_list


        # 역추적
        tmp_list = []
        current_node = max(previous_node_list, key=lambda node_: node_.probability)
        
        while current_node != None:
            tmp_list.append((current_node.word, current_node.tag))
            current_node = current_node.backpointer

        tmp_list.reverse()
        output.append(tmp_list)

    return output




