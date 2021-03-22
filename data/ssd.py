from scipy.stats import pearsonr,spearmanr

import numpy as np
class Helper(object):
    def __init__(self,grade_file = "grade.txt", timespans= ([ i for i in range(1810,1861,1)],[ i for i in range(1960,2011,1)])):
        self.data = self.read(grade_file)
        self.words = self.data.keys()
        self.timespans = timespans

    def read(self,filename):
        data = dict()
        with open(filename) as f:
            for line in f:
                token,score = line.split()
                if "_" in token:
                    token = token.split("_")[0]
                # print(token,score)
                data[token] = float(score)
        return data

    def evaluate(self,scores):
        return pearsonr(scores,list(self.data.values())),spearmanr(scores,list(self.data.values()))

    def adapt(self, in_words):
        word_not_found = [word for word in self.words if word not in in_words  ]
        print("words not found in semantically-shifted dataset: {} " + " ".join(word_not_found))
        self.words = [word for word in self.words if word in in_words  ]
        self.data =  {word:score  for word,score in self.data.items() if word in in_words  }

if __name__ == "__main__":
    helper = Helper()
    print(helper.timespans)
    print(helper.words)
    scores = np.random.rand(len(helper.words))
    print(helper.evaluate(scores))

