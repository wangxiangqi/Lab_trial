import random
from collections import defaultdict
import numpy as np
from regex import D
from sklearn.metrics import roc_auc_score
import scores
#找到bug 
class GBPR:
    user_count = 943
    item_count = 1682
    latent_factors = 20
    lr = 0.01
    reg = 0.01
    alpha_u = 0.01
    alpha_v = 0.01
    beta_v = 0.01
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
    setsizeG = 3
    rho=0.1
    eta=0.01
    #注意定义的实现

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
    def train(self,user_ratings_train):
        for user in range(self.user_count):
            #print(user)
            # sample a user
            u = random.randint(1, self.user_count)
            if u not in user_ratings_train.keys():
                continue
            # sample a positive item from the observed items
            if(user_ratings_train[u]):
                pass
            else:
                break
            i = random.sample(user_ratings_train[u], 1)[0]
            # sample a negative item from the unobserved items
            j = random.randint(1, self.item_count)
            #print("llllll")
            while j in user_ratings_train[u]:
                j = random.randint(1, self.item_count)
            #这里相对BPR应该加上setsizeGuser矩阵
            userSet=[]
            #print(type(user_ratings_train))
            #print(user_ratings_train.items())
            #这里该怎么实现Item2User的流程
            for llp in range(self.user_count):
                #print(user_ratings_train[llp])
                if i in user_ratings_train[llp]:
                    userSet.append(llp)
            userSet=list(set(userSet))
            #print("userSet:",userSet.__len__())
            userSetsize=userSet.__len__()
            userList=np.array(userSet)
            userSetG=[]
            userSetG.append(u)
            k=1#已经有了一个用户
            #print("point")
            #print(userSetsize)
            #print(self.setsizeG)
            while k<userSetsize and k<self.setsizeG and userSetsize>3:
                #print("k is",k)
                random_w=random.randint(1, userSetsize-1)
                w=userList[random_w]
                #print("w is",w)
                if w not in userSetG:
                    userSetG.append(w)
                    k=k+1
                else:
                    continue
            #现在进行calculating
            userSetSizeG=userSetG.__len__()
            #print("userSetSizeG:",userSetSizeG)
            U_G=[0]*self.latent_factors
            #print(userSetG)
            #print("w",w)
            w=0
            wo=w
            #print("shape of U_G",U_G)
            #print("w",w)
            while wo<userSetG.__len__():
                for f in range(self.latent_factors):
                    U_G[f]+=self.U[wo][f]
                wo+=1
                w+=1
            #计算U_G矩阵
            for f in range(self.latent_factors):
                U_G[f] = U_G[f] / userSetSizeG
            #计算rG_i矩阵
            rG_i=self.biasV[i]
            #print("rG_i:",rG_i)
            for f in range(self.latent_factors):
                rG_i+=U_G[f] * self.V[i][f]
            userSetSizeG=userSetG.__len__()
            u -= 1
            i -= 1
            j -= 1
            r_ui = np.dot(self.U[u], self.V[i].T) + self.biasV[i]
            r_uj = np.dot(self.U[u], self.V[j].T) + self.biasV[j]
            r_Guiuj = self.rho*rG_i + (1-self.rho) * r_ui - r_uj
            loss_func = -1.0 / (1 + np.exp(r_Guiuj))
            #print("loss_func:",loss_func)
            # update U and V
            #update U_w
            while w < userSetG.__len__():
                if w==u:
                    for f in range(self.latent_factors):
                        self.U[u][f] = self.U[u][f] - self.eta * ( loss_func * ( self.V[i][f] * ( 1-self.rho + self.rho/userSetSizeG ) - self.V[j][f] ) + self.alpha_u * self.U[u][f] )
                else:
                    for f in range(self.latent_factors):
                        self.U[w][f] = self.U[w][f] - self.eta * ( loss_func * self.V[i][f] * self.rho/userSetSizeG + self.alpha_u * self.U[w][f] / userSetSizeG )
            for f in range(self.latent_factors):
                self.V[i][f] = self.V[i][f] - self.eta * ( loss_func * ( (1-self.rho) * self.U[u][f] + self.rho * U_G[f] ) + self.alpha_v * self.V[i][f] )
            for f in range(self.latent_factors):
                self.V[j][f] = self.V[j][f] - self.eta * ( loss_func * (-self.U[u][f]) + self.alpha_v * self.V[j][f] )
            self.biasV[i] = self.biasV[i] - self.eta * ( loss_func + self.beta_v * self.biasV[i] )
            self.biasV[j] = self.biasV[j] - self.eta * ( loss_func * (-1) + self.beta_v * self.biasV[j] )
            #print("next goes on")
        #print("self.user_count",self.user_count)
    def predict_(self,user,item):
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
            print("training:",i)
            self.train(user_ratings_train)
        predict_matrix = self.predict_(self.U, self.V)
        # prediction
        print("now is prediction")
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
if __name__=="__main__":
    gbpr=GBPR()
    gbpr.main()
