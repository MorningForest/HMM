from itertools import chain
import pickle
import os
import math
MIN_FLOAT = -3.14e100

class HMM:
    def __init__(self):
        self.A_dict = {}    #状态转移概率矩阵
        self.B_dict = {}    #发射概率矩阵
        self.pi_dict = {}   #初始状态矩阵
        self.state_list = {'S','B','M','E'} #状态集合
        self.isTraining = False
        self.model_file = r"data/hmm_models.pkl"

    def _init_parameter(self):
        '''
        Returns # 初始化参数
        -------
        '''

        for state in self.state_list:
            self.A_dict[state] = {s:0 for s in self.state_list}
            self.B_dict[state] = {}
            self.pi_dict[state] = 0.0

    def _try_load_model(self, trained):
        '''
        Parameters
        ----------
        trained

        Returns
        -------
        #load model
        '''
        if trained:
            with open(self.model_file, 'rb') as f:
                self.pi_dict = pickle.load(f)
                self.A_dict = pickle.load(f)
                self.B_dict = pickle.load(f)
                self.isTraining = True

    def _make_label(self, words):
        '''
        Parameters
        ----------
        words

        Returns label
        -------

        '''
        output=[]
        if len(words)==1:
            output += ['S']
        else:
            output += ['B']+['M']*(len(words)-2)+['E']
        return output


    def _train(self, path):
        '''
        Parameters
        ----------
        path

        Returns A,B,pi
        -------
        '''
        self._init_parameter()
        count_dict={s: 0 for s in self.state_list}
        words = set()
        lineNum = 0
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                lineNum += 1
                words |= set(line)
                lineList = line.split("  ")
                line = list(chain.from_iterable(lineList))
                lineState=[]
                for item in lineList:
                    lineState.extend(self._make_label(item))
                for k,v in enumerate(lineState):
                    count_dict[v] += 1
                    if k == 0:
                        self.pi_dict[v] += 1  #'
                    else:
                        self.A_dict[lineState[k-1]][v] += 1
                        self.B_dict[v][line[k]] = self.B_dict[v].get(line[k], 0) + 1

        self.pi_dict = {s: math.log2(v/lineNum) if v>0 else 0  for s,v in self.pi_dict.items()}
        self.A_dict = {k: {k1: math.log2((v1+1)/count_dict[k]) for k1,v1 in v.items()} for k,v in self.A_dict.items()}
        self.B_dict = {k: {k1: math.log2((v1+1)/count_dict[k]) for k1,v1 in v.items()} for k,v in self.B_dict.items()}
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.pi_dict, f)
            pickle.dump(self.A_dict, f)
            pickle.dump(self.B_dict, f)

    def veterbi(self, text, states, start_p, trans_p, emit_p):
        '''
        Parameters
        ----------
        text
        states
        start_p
        trans_p
        emit_p

        Returns prob, pos_list
        -------

        '''
        V = [{}]
        path={}
        for s in states:
            V[0][s] = start_p[s]+emit_p[s].get(text[0], MIN_FLOAT)
            path[s]=[s]
        for t in range(1, len(text)):
            newpath = {}
            V.append({})
            neverSeen = text [t] not in emit_p['S'].keys() and text[t] not in emit_p['M'].keys() and \
                        text[t] not in emit_p['E'].keys() and text[t] not in emit_p['B'].keys()
            for s in states:
                empt = emit_p[s].get(text[t], MIN_FLOAT) if not neverSeen else 1.0
                (prob, state) = max(
                    (V[t-1][y]+trans_p[y].get(s, MIN_FLOAT)+empt, y) for y in states
                )
                V[t][s] = prob
                newpath[s] = path[state]+[s]
            path = newpath
        if emit_p['M'].get(text[-1], 0) > emit_p['S'].get(text[-1], 0):
            (prob, state) = max([(V[len(text) - 1][y], y) for y in ('E', 'M')])
        else:
            (prob, state) = max([(V[len(text) - 1][y], y) for y in states])
        return prob, path[state]

    def cut(self, text):
        '''
        Parameters
        ----------
        text

        Returns
        -------
        cut_word
        '''
        if not self.isTraining:
            self._try_load_model(os.path.exists(self.model_file))
        prob, pos_list = self.veterbi(text, self.state_list, self.pi_dict, self.A_dict, self.B_dict)
        print(pos_list)
        begin, next = 0, 0
        for i, char in enumerate(text):
            pos = pos_list[i]
            if pos == 'B':
                begin = i
            elif pos == 'E':
                yield text[begin: i + 1]
                next = i + 1
            elif pos == 'S':
                yield char
                next = i + 1
        if next < len(text):
            yield text[next:]

if __name__ == '__main__':
    path = r"HMM/pku_training.utf8"
    h=HMM()
    text = '''1987年11月，中国肯德基第1家餐厅落户北京；
　　2001年10月，中国肯德基第500家餐厅落户上海；
　　2002年09月，中国肯德基第700家餐厅落户深圳；
　　2004年01月，中国肯德基第1000家餐厅落户北京；
　　2004年12月，中国肯德基第1200家餐厅落户三亚；
　　2007年11月，中国肯德基第2000家餐厅落户成都；
　　2009年06月，中国肯德基第2600家餐厅落户郑州；
　　2010年06月，中国肯德基第3000家餐厅落户上海……
    '''
    print(text.strip().split('\n'))
    res = h.cut(text)
    print(str(list(res)))
