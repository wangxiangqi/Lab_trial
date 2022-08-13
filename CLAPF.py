import random
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
import scores
class CLAPF:
    user_count = 943
    item_count = 1682
    latent_factors = 20
    lr = 0.01
    reg = 0.01
    lambnda=0.1
    eta=0.01
    alpha_u=0.01
    alpha_v=0.01
    beta_v = 0.002
    train_count = 1000
    train_data_path = 'train.txt'
    test_data_path = 'test.txt'
    size_u_i = user_count * item_count
    # latent_factors of U & V
    U = np.random.rand(user_count, latent_factors) * 0.01
    V = np.random.rand(item_count, latent_factors) * 0.01
    biasV = np.random.rand(item_count) * 0.01
    test_data = np.zeros((user_count, item_count))
    test = np.zeros(size_u_i)
    predict_ = np.zeros(size_u_i)
    def load_data(self, path):
        user_ratings = defaultdict(set)
        with open(path, 'r') as f:
            for line in f.readlines():
                u, i = line.split(" ")
                u = int(u)
                i = int(i)
                user_ratings[u].add(i)
        return user_ratings

    def load_test_data(self, path):
        file = open(path, 'r')
        for line in file:
            line = line.split(' ')
            user = int(line[0])
            item = int(line[1])
            self.test_data[user - 1][item - 1] = 1
    def train(self, user_ratings_train):
        for user in range(self.user_count):
            # sample a user
            u = random.randint(1, self.user_count)
            if u not in user_ratings_train.keys():
                continue
            i= random.sample(user_ratings_train[u], 1)[0]
            uo = random.randint(1, self.user_count)
            if uo not in user_ratings_train.keys():
                continue
            h=random.sample(user_ratings_train[uo], 1)[0]
            j = random.randint(1, self.item_count)
            while j in user_ratings_train[u]:
                j = random.randint(1, self.item_count)
            u -= 1
            uo -=1
            i -= 1
            h -= 1
            j -= 1
            r_ui = np.dot(self.U[u], self.V[i].T) + self.biasV[i]
            r_uj = np.dot(self.U[u], self.V[j].T) + self.biasV[j]
            r_uh = np.dot(self.U[u], self.V[h].T) + self.biasV[h]
            r_uiuj = self.lambnda * (r_uh - r_ui) + (1-self.lambnda) * (r_ui - r_uj)
            loss_func = -1.0 / (1 + np.exp(r_uiuj))
            # now is updating section
            for f in range(self.latent_factors):
                self.U[u][f] = self.U[u][f] - self.eta * ( loss_func * ( self.lambnda * (self.V[h][f] - self.V[i][f] ) + (1-self.lambnda) * (self.V[i][f] - self.V[j][f])) + self.alpha_u * self.U[u][f] )
            for f1 in range(self.latent_factors):
                self.V[i][f1] = self.V[i][f1] - self.eta * ( loss_func * (1-2*self.lambnda) * self.U[u][f1] + self.alpha_v * self.V[i][f1] )
            for f2 in range(self.latent_factors):
                self.V[j][f2] = self.V[j][f2] - self.eta * ( loss_func * (self.lambnda-1) * self.U[u][f] + self.alpha_v * self.V[j][f] )
            for f3 in range(self.latent_factors):
                self.V[h][f3] = self.V[h][f3] - self.eta * ( loss_func * self.lambnda *self.U[u][f3] + self.alpha_v * self.V[h][f3] )
            self.biasV[i] = self.biasV[i] - self.eta * ( loss_func * (1-2*self.lambnda) + self.beta_v * self.biasV[i] )
            self.biasV[j] = self.biasV[j] - self.eta * ( loss_func * (self.lambnda-1) + self.beta_v * self.biasV[j] )
            self.biasV[h] = self.biasV[h] - self.eta * ( loss_func * self.lambnda + self.beta_v * self.biasV[h] )
    def predict(self, user, item):
        predict = np.mat(user) * np.mat(item.T)
        return predict

    def main(self):
        user_ratings_train = self.load_data(self.train_data_path)
        self.load_test_data(self.test_data_path)
        for u in range(self.user_count):
            for item in range(self.item_count):
                if int(self.test_data[u][item]) == 1:
                    self.test[u * self.item_count + item] = 1
                else:
                    self.test[u * self.item_count + item] = 0
        # training
        for i in range(self.train_count):
            print(i)
            self.train(user_ratings_train)
        predict_matrix = self.predict(self.U, self.V)
        # prediction
        self.predict_ = predict_matrix.getA().reshape(-1)
        self.predict_ = pre_handel(user_ratings_train, self.predict_, self.item_count)
        auc_score = roc_auc_score(self.test, self.predict_)
        print('AUC:', auc_score)
        # Top-K evaluation
        scores.topK_scores(self.test, self.predict_, 5, self.user_count, self.item_count)
def pre_handel(set, predict, item_count):
    # Ensure the recommendation cannot be positive items in the training set.
    for u in set.keys():
        for j in set[u]:
            predict[(u - 1) * item_count + j - 1] = 0
    return predict
if __name__ == '__main__':
    clapf = CLAPF()
    clapf.main()
    