from itertools import chain
import pickle
import os
class HMM:
    def __init__(self):
        self.A_dict={}  #状态转移概率矩阵
        self.B_dict={}  #发射概率矩阵
        self.pi_dict={} #初始状态矩阵
        self.state_list={'B', 'E', 'S', 'M'}  #状态集合
        self.model_file = r"data/hmm_model.pkl"
        self.load_para=False

    def _try_load_model(self, trained):
        if trained:
            with open(self.model_file, 'rb') as f:
                self.pi_dict=pickle.load(f)
                self.A_dict=pickle.load(f)
                self.B_dict=pickle.load(f)
                self.load_para=True

    def _init_parameter(self):
        #初始化参数
        for state in self.state_list:
            self.A_dict[state]={s:0.0 for s in self.state_list} #状态值X状态值
            self.B_dict[state]={} #状态值X观测值
            self.pi_dict[state]=0.0 #初始状态概率

    def _make_label(self, sentence):
        #为词语打标记
        out_text = []
        if len(sentence)==1:
            out_text.append('S')
        else:
            out_text += ['B'] + ['M'] * (len(sentence) - 2) + ['E']
        return out_text

    def _train(self, path):
      #训练数据集，得到矩阵
        self._init_parameter()
        count_dic={s:0 for s in self.state_list}
        words = set() #观察者集合
        with open(path, encoding="utf-8") as f:
            line_num=0
            for line in f:
                line_num += 1
                line=line.strip()
                if not line:
                    continue
                linelist=list(chain.from_iterable(line.split(" ")))
                words |= set(linelist)
                linestate=[]
                for word in line.split(" "):
                    linestate.extend(self._make_label(word))
                for k,v in enumerate(linestate):
                    count_dic[v] += 1 #状态出现加一
                    if k==0:
                        self.pi_dict[v] += 1  #计算初始状态值
                    else:
                        #计算转移概率矩阵
                        self.A_dict[linestate[k-1]][v] += 1
                        #计算发射概率
                        self.B_dict[v][linelist[k]]=self.B_dict[v].get(linelist[k], 0) + 1
            f.close()
        self.pi_dict={k:v/line_num for k,v in self.pi_dict.items()}   #计算初始状态概率矩阵 
        self.A_dict={k:{k1:v1/count_dic[k] for k1,v1 in v.items()} for k,v in self.A_dict.items()}#计算状态转移矩阵
        self.B_dict={k:{k1:(v1+1)/count_dic[k] for k1,v1 in v.items()} for k,v in self.B_dict.items()}#计算发射概率矩阵
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.pi_dict, f)
            pickle.dump(self.A_dict, f)
            pickle.dump(self.B_dict, f)
    
    def viterbi(self, text, states, start_p, trans_p, emit_p):
        V = [{}]
        path = {}
        for y in states:
            V[0][y] = start_p[y] * emit_p[y].get(text[0], 0)
            path[y] = [y]
        for t in range(1, len(text)):
            V.append({})
            newpath = {}
            # 检验训练的发射概率矩阵中是否有该字
            neverSeen = text [t] not in emit_p['S'].keys() and text[t] not in emit_p['M'].keys() and \
                        text[t] not in emit_p['E'].keys() and text[t] not in emit_p['B'].keys()
            for y in states:
                # 设置未知字单独成词\n",
                emitP = emit_p[y].get(text[t], 0) if not neverSeen else 1.0
                (prob, state) = max([(V[t - 1][y0] * trans_p[y0].get(y, 0) * emitP, y0)
                                     for y0 in states if V[t - 1][y0] > 0])
                V[t][y] = prob
                newpath[y] = path[state] + [y]
            path = newpath
        if emit_p['M'].get(text[-1], 0) > emit_p['S'].get(text[-1], 0):
            (prob, state) = max([(V[len(text) - 1][y], y) for y in ('E', 'M')])
        else:
            (prob, state) = max([(V[len(text) - 1][y], y) for y in states])

        return prob, path[state]
  
    def cut(self, text):
        if not self.load_para:
            self._try_load_model(os.path.exists(self.model_file))
        prob, pos_list = self.viterbi(text, self.state_list, self.pi_dict, self.A_dict, self.B_dict)
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

h =HMM()
res=h.cut("实验室中任何可控的、有稳定特征能态的量子系统都可以看作是一种电池。")
print(str(list(res)))

