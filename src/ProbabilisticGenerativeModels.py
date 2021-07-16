import numpy as np
import sys
from abc import ABC

class Constant:
    FMAX = np.finfo(np.float16).max


class IInputMaker(ABC):
    '''入力信号生成インターフェース
    '''
    def _make_signal(self):
        '''信号xを生成
        '''
        pass

    def _make_observed_data(self, x):
        '''観測データyを生成
        '''
        pass

    def make_dataset(self, sigma=1):
        '''信号xと観測データyを生成
        '''
        pass



class DefaultInputMaker(IInputMaker):
    '''一番最初に問題に記されている入力生成
    '''
    def __init__(self, seed=0):
        self._rng = np.random.default_rng(seed)
        self._x_possible_values = [[0, 0], [0, 1], [1, 0], [1, 1]]
        self._probability = [0.6, 0.1, 0.1, 0.2]

    def probability(self, x):
        '''xが出現する確率を返す関数
        '''
        for i in range(len(self._x_possible_values)):
            if((x == np.array(self._x_possible_values[i])).all()):
                return self._probability[i]
        
        print('Error:不正な引数', file=sys.stderr)

    def _make_signal(self):
        '''信号xを生成
        '''
        x = self._rng.choice(self._x_possible_values, p=self._probability)
        return np.array(x)
        
    def _make_observed_data(self, x, n=[3, 3], sigma=1):
        '''観測データyを生成
            Arg:
                x: 大元の信号x
                n: n[i]は x[i]から作られるyの数
        '''
        
        y = np.full((len(n), max(n)), Constant.FMAX)
        
        # yをxのデータをもとにガウス分布で作成
        for i in range(len(n)):
            for j in range(n[i]):
                y[i][j] = self._rng.normal(x[i], sigma)
        
        return y

    def make_dataset(self, n=[3, 3], sigma=1):
        '''信号xと観測データyを生成
        '''
        x = self._make_signal()
        y = self._make_observed_data(x, n, sigma)
        return (x, y)


class ProbabilisticGenerativeModelsForX2:
    '''確率的生成モデル
    観測データyを与えられた時、大元の信号xを推測する
    xの生成分布は既知である
    現在はxが二次元の場合のみ
    '''
    def __init__(self):
        pass

    def calc_S_G(self, y, n, sigma, probability_func):
        ''' 
        統計量、事後確率を計算
        xが二次元の場合のみしか使えません
        '''
        x_possible_value_num = 2
        
        hG_numerator = 0 # 分子
        for x1_value in range(x_possible_value_num):
            for x2_value in range(x_possible_value_num):
                superstring = 0
                x_tilde = np.array([x1_value, x2_value])

                for i in range(len(n)):
                    for j in range(n[i]):
                        superstring += ((1 - 2*y[i][j] + 2*y[i][j]*x_tilde[i] - x_tilde[i]**2) / (2 * sigma**2))

                hG_numerator += probability_func(x_tilde) * np.exp(superstring)

        return probability_func(np.array([1, 1])) / hG_numerator

    def calc_S_T(self, y, n, sigma, probability_func):
        '''
        xが2つの時しか使えません
        '''
        superscript = 0 # expの肩
        for i in range(len(n)):
            for j in range(n[i]):
                superscript += (1 - 2*y[i][j])/(2 * sigma**2)

        h_T = 1 + probability_func(np.array([0, 0]))/probability_func(np.array([1, 1])) * np.exp(superscript)
        
        return 1/h_T

    def calc_S_P_12(self, y, n, sigma, probability_func):
        superscript_1 = 0 # expの肩

        i=0
        for j in range(n[i]):
            superscript_1 += (1 - 2*y[i][j]) / (2 * sigma**2)

        term1_inverse = 1 + \
            (probability_func(np.array([0, 0])) + probability_func(np.array([0, 1]))) / \
            (probability_func(np.array([1, 0])) + probability_func(np.array([1, 1]))) * \
            np.exp(superscript_1)
        
        superscript_2 = 0
        i = 1
        for j in range(n[i]):
            superscript_2 += (1 - 2*y[i][j]) / (2 * sigma**2)

        term2_inverse = 1 + probability_func(np.array([1, 0])) / probability_func(np.array([1, 1])) * np.exp(superscript_2)

        return 1 / (term1_inverse * term2_inverse)

        

if __name__ == '__main__':
    np.set_printoptions(precision=20, suppress=True, floatmode='unique')

    x, y = DefaultInputMaker().make_dataset(n=[3, 5])
    print(x)
    print(y)
    print(DefaultInputMaker().probability([1, 1]))