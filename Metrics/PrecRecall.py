class Metrics:
    def __init__(self, y_true, y_pred):
        self.pred_1 = {i for i,elem in enumerate(y_pred) if elem == 1}
        self.true_1 = {i for i,elem in enumerate(y_true) if elem == 1}
        self.true_pos = self.pred_1 & self.true_1

    def precision(self):
        return len(self.true_pos)/len(self.true_1)

    def recall(self):
        return len(self.true_pos)/len(self.pred_1)

    def f1(self):
      return 2*self.precision()*self.recall() / (self.precision() + self.recall())
