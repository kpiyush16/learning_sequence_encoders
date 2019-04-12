# coding: utf-8
# In[1]:
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
from torch.autograd import Variable
from sklearn.metrics.pairwise import pairwise_distances
import warnings
from data_generate import *
from utils import *

class Config(object):
    def __init__(self):
        self.train_model = "DistMult"
        self.save_path = "./saved_res1/"
        self.testFlag = True
        self.save_epoch = 10
        self.hidden_size = 100
        self.nbatches = 0
        self.batch_size = 512
        self.entity_nums = 0
        self.relation_nums = 0
        self.trainTimes = 201
        self.margin = 1.0
        self.lmbda = 0.01
        self.lr = 1e-3
        self.p_norm = 2
        self.vocab = True
        self.valid_epoch = 201
        self.dropout = 0.4
        self.criterion = "MRandSoftPlus" # "CE" or "MRandSoftPlus"
        self.load = "./saved_res1/DistMult_200.pt"

class TransE(nn.Module):
    def __init__(self, config):
        super(TransE, self).__init__()
        self.config = config
        self.entity_embeddings = nn.Embedding(config.entity_nums, config.hidden_size)
        self.relation_embeddings = nn.Embedding(config.relation_nums, config.hidden_size)
        if self.config.criterion == "CE":
            self.criterion = nn.BCEWithLogitsLoss(size_average=True, reduce=True).cuda()
        elif self.config.criterion == "MRandSoftPlus":
            self.criterion = nn.MarginRankingLoss(self.config.margin, False).cuda()
        else:
            print("Enter correct criterion");exit(0)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform(self.relation_embeddings.weight.data)
    
    def loss_calc(self, pos, neg):
        if self.config.criterion == "CE":
            y = torch.tensor([1]*len(pos)+[0]*len(neg)).cuda().float()
            return self.criterion(torch.cat((pos, neg)).float(), y)
        else:
            y = Variable(torch.Tensor([-1.0]).cuda())
            return self.criterion(pos, neg, y)

    def _calc(self, h, r, t):
        return torch.norm(h + r - t, self.config.p_norm, -1)

    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        pos_h_e = self.entity_embeddings(pos_h)
        pos_r_e = self.relation_embeddings(pos_r[:,0])
        pos_t_e = self.entity_embeddings(pos_t)
        
        neg_h_e = self.entity_embeddings(neg_h)
        neg_r_e = self.relation_embeddings(neg_r[:,0])
        neg_t_e = self.entity_embeddings(neg_t)

        p_score = self._calc(pos_h_e, pos_r_e, pos_t_e)
        n_score = self._calc(neg_h_e, neg_r_e, neg_t_e)
        
        return self.loss_calc(p_score, n_score)

    def predict(self, h, r, t):
        ph_e = self.entity_embeddings(h)
        pt_e = self.entity_embeddings(t)
        pr_e = self.relation_embeddings(r[:,0])
        # pr_e = self.relation_embeddings(Variable(torch.from_numpy(r)))
        # predict = torch.sum(torch.abs(ph_e + pr_e - pt_e), 1)
        return (ph_e, pr_e, pt_e)

class TA_TransE(nn.Module):
    def __init__(self, config):
        super(TA_TransE, self).__init__()
        self.config = config
        self.entity_embeddings = nn.Embedding(config.entity_nums, config.hidden_size)
        self.relation_embeddings = nn.Embedding(config.relation_nums, config.hidden_size)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True, dropout=self.config.dropout)
        self.hidden = self.init_hidden()
        if self.config.criterion == "CE":
            self.criterion = nn.BCEWithLogitsLoss(size_average=True, reduce=True).cuda()
        elif self.config.criterion == "MRandSoftPlus":
            self.criterion = nn.MarginRankingLoss(self.config.margin, False).cuda()
        else:
            print("Enter correct criterion");exit(0)
        self.init_weights()
        
    def init_hidden(self):
        return (Variable(torch.zeros(1, self.config.batch_size, self.config.hidden_size)).cuda(),
                Variable(torch.zeros(1, self.config.batch_size, self.config.hidden_size)).cuda())

    def init_weights(self):
        nn.init.xavier_uniform(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform(self.relation_embeddings.weight.data)
    
    def _calc(self, h, r, t):
        return torch.norm(h + r - t, self.config.p_norm, -1)

    def loss_calc(self, pos, neg):
        if self.config.criterion == "CE":
            y = torch.tensor([1]*len(pos)+[0]*len(neg)).cuda().float()
            return self.criterion(torch.cat((pos, neg)).float(), y)
        else:
            y = Variable(torch.Tensor([-1.0]).cuda())
            return self.criterion(pos, neg, y)
        

    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        pos_h_e = self.entity_embeddings(pos_h)
        # pos_r_e = self.lstm(self.relation_embeddings(pos_r), self.hidden)[0][:, -1, :]
        pos_r_e = self.lstm(self.relation_embeddings(pos_r))[1][0][0]
        pos_t_e = self.entity_embeddings(pos_t)
        
        neg_h_e = self.entity_embeddings(neg_h)
        # neg_r_e = self.lstm(self.relation_embeddings(neg_r), self.hidden)[0][:, -1, :]
        neg_r_e = self.lstm(self.relation_embeddings(neg_r))[1][0][0]
        neg_t_e = self.entity_embeddings(neg_t)

        p_score = self._calc(pos_h_e, pos_r_e, pos_t_e)
        n_score = self._calc(neg_h_e, neg_r_e, neg_t_e)
        
        return self.loss_calc(p_score, n_score)

    def predict(self, h, r, t):
        ph_e = self.entity_embeddings(h)
        pt_e = self.entity_embeddings(t)
        # pr_e = self.lstm(self.relation_embeddings(r))[0][:,-1,:]
        pr_e = self.lstm(self.relation_embeddings(r))[1][0][0]
        # pr_e = self.relation_embeddings(Variable(torch.from_numpy(r)))
        # predict = torch.sum(torch.abs(ph_e + pr_e - pt_e), 1)
        return (ph_e, pr_e, pt_e)

class DistMult(nn.Module):
    def __init__(self, config):
        super(DistMult, self).__init__()
        self.config = config
        self.entity_embeddings = nn.Embedding(config.entity_nums, config.hidden_size)
        self.relation_embeddings = nn.Embedding(config.relation_nums, config.hidden_size)
        if self.config.criterion == "CE":
            self.criterion = nn.BCEWithLogitsLoss(size_average=True, reduce=True).cuda()
        elif self.config.criterion == "MRandSoftPlus":
            self.criterion=nn.Softplus().cuda()
        else:
            print("Enter correct criterion");exit(0)        
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform(self.relation_embeddings.weight.data)
    
    def _calc(self, h, t, r):
        return - torch.sum(h * t * r, -1)

    def loss(self, score, regul, batch_y):
        if self.config.criterion == "CE":
            # return torch.mean(self.criterion(score, batch_y)) + self.config.lmbda * regul
            return self.criterion(score, batch_y)
        else:
            return torch.mean(self.criterion(score * batch_y)) + self.config.lmbda * regul 
    
    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        h, r, t = torch.cat((pos_h, neg_h)), torch.cat((pos_r, neg_r)), torch.cat((pos_t, neg_t))
        batch_y = torch.Tensor([1]*len(pos_h)+[-1]*len(neg_h)).cuda()
        h, t = self.entity_embeddings(h), self.entity_embeddings(t)
        r = self.relation_embeddings(r[:,0])
        score = self._calc(h ,t, r)
        regul = torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)
        return self.loss(score, regul, batch_y)
    
    def predict(self, h, r, t):
        # p_e_h=self.entity_embeddings(Variable(torch.from_numpy(predict_h)))
        p_e_h=self.entity_embeddings(h)
        p_e_t=self.entity_embeddings(t)
        p_e_r = self.relation_embeddings(r[:,0])
        # p_e_r=self.relation_embeddings(Variable(torch.from_numpy(predict_r)))
        # p_score=-self.loss_calc(p_e_h,p_e_t,p_e_r)
        return (p_e_h, p_e_r, p_e_t)

class TA_DistMult(nn.Module):
    def __init__(self, config):
        super(TA_DistMult, self).__init__()
        self.config = config
        self.entity_embeddings = nn.Embedding(config.entity_nums, config.hidden_size)
        self.relation_embeddings = nn.Embedding(config.relation_nums, config.hidden_size)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True, dropout=self.config.dropout)
        
        if self.config.criterion == "CE":
            self.criterion = nn.BCEWithLogitsLoss(size_average=True, reduce=True).cuda()
        elif self.config.criterion == "MRandSoftPlus":
            self.criterion=nn.Softplus().cuda()
        else:
            print("Enter correct criterion");exit(0)
        self.hidden = self.init_hidden()
        self.init_weights()
        
    def init_hidden(self):
        return (Variable(torch.zeros(1, 2*self.config.batch_size, self.config.hidden_size)).cuda(),
                Variable(torch.zeros(1, 2*self.config.batch_size, self.config.hidden_size)).cuda())

    def init_weights(self):
        nn.init.xavier_uniform(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform(self.relation_embeddings.weight.data)
    
    def _calc(self, h, t, r):
        return - torch.sum(h * t * r, -1)

    def loss(self, score, regul, batch_y):
        if self.config.criterion == "CE":
            # return torch.mean(self.criterion(score, batch_y)) + self.config.lmbda * regul
            return self.criterion(score, batch_y)
        else:
            return torch.mean(self.criterion(score * batch_y)) + self.config.lmbda * regul 
    
    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        h, r, t = torch.cat((pos_h, neg_h)), torch.cat((pos_r, neg_r)), torch.cat((pos_t, neg_t))
        batch_y = torch.tensor([1]*len(pos_h)+[-1]*len(neg_h)).cuda().float()

        h, t = self.entity_embeddings(h), self.entity_embeddings(t)
        r = self.lstm(self.relation_embeddings(r))[1][0][0]
        score = self._calc(h, t, r)

        regul = torch.mean(h ** 2) + torch.mean(t ** 2) + torch.mean(r ** 2)
        return self.loss(score, regul, batch_y)
    
    def predict(self, h, r, t):
        p_e_h=self.entity_embeddings(h)
        p_e_t=self.entity_embeddings(t)
        p_e_r = self.lstm(self.relation_embeddings(r))[1][0][0]
        return (p_e_h, p_e_r, p_e_t)

def evaluation(model, entity_embeddings, testList, tripleDict, filter=True, L1_flag=False, bs = 500, cuda = False, res_file = None):
    totalRank, hit10Count, tripleCount, reciRank = 0, 0, 0, 0
    print("Stating Test----------------->")
    print("Test Triple Count", len(testList))

    for i in range(0, len(testList), bs):
        model.eval()
        if cuda:
            headList = torch.tensor([triple.h for triple in testList[i:i+bs]], dtype=torch.int64).cuda()
            tailList = torch.tensor([triple.t for triple in testList[i:i+bs]], dtype=torch.int64).cuda()
            relList = torch.tensor([triple.r for triple in testList[i:i+bs]], dtype=torch.int64).cuda()
        else:
            headList = torch.tensor([triple.h for triple in testList[i:i+bs]], dtype=torch.int64)
            tailList = torch.tensor([triple.t for triple in testList[i:i+bs]], dtype=torch.int64)
            relList = torch.tensor([triple.r for triple in testList[i:i+bs]], dtype=torch.int64)
        h_e, r_e, t_e = model.predict(headList, relList, tailList)
        if cuda:
            h_e, r_e, t_e, headList, tailList = h_e.cpu().numpy(), r_e.cpu().numpy(), t_e.cpu().numpy(), headList.cpu().numpy(), tailList.cpu().numpy()
        if("TransE" in model.config.train_model):
            c_t_e = h_e + r_e
            c_h_e = t_e - r_e
            if L1_flag == True:
                dist = pairwise_distances(c_t_e, entity_embeddings, metric='manhattan')
            else:
                dist = pairwise_distances(c_t_e, entity_embeddings, metric='euclidean')

            rankArrayTail = np.argsort(dist, axis=1)
            # print(type(tailList), type(rankArrayTail))
            if filter == False:
                rankListTail = np.array([int(np.argwhere(elem[1]==elem[0]))+1 for elem in zip(tailList, rankArrayTail)])
            else:
                rankListTail = np.array([argwhereTail(elem[0], elem[1], elem[2], elem[3], tripleDict)+1
                                for elem in zip(headList, tailList, relList, rankArrayTail)])
            
            isHit10ListTail = [x for x in rankListTail if x < 10]

            if L1_flag == True:
                dist = pairwise_distances(c_h_e, entity_embeddings, metric='manhattan')
            else:
                dist = pairwise_distances(c_h_e, entity_embeddings, metric='euclidean')
            
            rankArrayHead = np.argsort(dist, axis=1)

            if filter == False:
                rankListHead = np.array([int(np.argwhere(elem[1]==elem[0]))+1 for elem in zip(headList, rankArrayHead)])
            # Check whether it is false negative
            else:
                rankListHead = np.array([argwhereHead(elem[0], elem[1], elem[2], elem[3], tripleDict)+1
							for elem in zip(headList, tailList, relList, rankArrayHead)])
            
            isHit10ListHead = [x for x in rankListHead if x < 10]

            totalRank += float(sum(rankListTail) + sum(rankListHead))/2.0
            hit10Count += (len(isHit10ListTail) + len(isHit10ListHead))/2
            tripleCount += (len(rankListTail) + len(rankListHead))/2
            reciRank += (0.5)*(sum(1/rankListHead)+sum(1/rankListTail))
            print(totalRank/(i+bs), hit10Count*100/(i+bs), reciRank/(i+bs))
        else:
            rankListHead, rankListTail = [], []
            score = -np.sum(h_e * t_e * r_e, 1)
            score_list_tail = np.transpose([-np.sum(h_e * x * r_e, 1) for x in entity_embeddings])
            score_list_head = np.transpose([-np.sum(x * t_e * r_e, 1) for x in entity_embeddings])
            for i in range(len(score)):
                for j in range(len(score_list_tail[i])):
                    if(score_list_tail[i][j] == score[i]):
                        rankListTail.append(np.where(np.argsort(score_list_tail[i]) == j)[0] + 1)

            for i in range(len(score)):
                for j in range(len(score_list_head[i])):
                    if(score_list_head[i][j] == score[i]):
                        rankListHead.append(np.where(np.argsort(score_list_head[i]) == j)[0] + 1)
            isHit10ListTail = [x for x in rankListTail if x < 10]
            isHit10ListHead = [x for x in rankListHead if x < 10]
            # print(rankListHead)
            totalRank += float(sum(rankListTail) + sum(rankListHead))/2.0
            hit10Count += (len(isHit10ListTail) + len(isHit10ListHead))/2
            tripleCount += (len(rankListTail) + len(rankListHead))/2
            reciRank += (0.5)*(sum(1/np.array(rankListHead))+sum(1/np.array(rankListTail)))
        
    print("Hits@10, MR, MRR")
    s = "{:.0f}, {:.1f}, {:.4f}".format(float(hit10Count)*100/len(testList), float(totalRank)/len(testList), float(reciRank)/float(len(testList)))
    print(s)
    if(res_file):
        with open(res_file, "a") as f:
            f.write(s+"\n")


def test():
    config = Config()
    te_set, vocab_e, vocab_r = test_data_prepare("./TemporalKGs/icews05-15/icews_2005-2015_test.txt", vocab_e_path="ent.vocab", vocab_r_path="rel.vocab")
    t_set, v_set, _, _ = data_prepare("./TemporalKGs/icews05-15/icews_2005-2015_train.txt", "./TemporalKGs/icews05-15/icews_2005-2015_valid.txt", config.vocab)
    _, _, tripleDict = loadTriple(test_data=t_set+v_set+te_set)
    config.relation_nums = len(vocab_r)
    config.entity_nums = len(vocab_e)

    print("Evaluating %s"%(config.train_model))
    testTotal, testList, _ = loadTriple(test_data=te_set)

    print("Total number of triples: ", testTotal)
    print("Total number of Entities:", config.entity_nums)
    print("Total number of Relations:", config.relation_nums-32)
    device = torch.device("cuda")
    if(config.train_model == "TA_DistMult"):
        model = TA_DistMult(config).to(device)
    elif(config.train_model == "TA_TransE"):
        model = TA_TransE(config).to(device)
    elif(config.train_model == "TransE"):
        model = TransE(config).to(device)
    elif(config.train_model == "DistMult"):
        model = DistMult(config).to(device)
    else:
        print("Please Enter Valid Model Name");exit(1)

    ckpt = torch.load(config.load)
    print("Loaded ckpt from %s"%(config.load))
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    with torch.no_grad():
        evaluation(model, model.entity_embeddings.weight.data.cpu().numpy(), testList, tripleDict, cuda=True, res_file="results_%s.log"%(config.train_model))

    return

def train():
    config = Config()
    t_set, v_set, vocab_e, vocab_r = data_prepare("./TemporalKGs/icews05-15/icews_2005-2015_train.txt", "./TemporalKGs/icews05-15/icews_2005-2015_valid.txt", vocab = config.vocab)
    _, _, tripleDict = loadTriple(test_data=t_set+v_set)
    config.relation_nums = len(vocab_r)
    config.entity_nums = len(vocab_e)
    print("Total number of triples: ",len(t_set+v_set))
    print("Total number of Entities:", config.entity_nums)
    print("Total number of Relations:", config.relation_nums-32)

    if(config.nbatches):
        config.batch_size = len(t_set) // config.nbatches
    else:
        config.nbatches = len(t_set)//config.batch_size
    
    device = torch.device("cuda")

    if(config.train_model == "TA_DistMult"):
        model = TA_DistMult(config).to(device)
    elif(config.train_model == "TA_TransE"):
        model = TA_TransE(config).to(device)
    elif(config.train_model == "TransE"):
        model = TransE(config).to(device)
    elif(config.train_model == "DistMult"):
        model = DistMult(config).to(device)
    else:
        print("Please Enter Valid Model Name");exit(1)
    # optimizer = optim.SGD(model.parameters(), lr=1e-2)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    print("Starting Train %s----------->"%(config.train_model))
    log = open("train_%s.log"%(config.train_model), "a")
    log.flush()
    for epoch in range(config.trainTimes):
        model.train()
        train_loss = 0
        for bn in range(config.nbatches):
            ph, pr, pt = tuple(get_batch(t_set, config.batch_size, bn*config.batch_size))
            nh, nr, nt = tuple(get_nbatch(t_set, config.batch_size, bn*config.batch_size))
            ph_t = torch.tensor(ph, dtype=torch.int64).cuda()
            pr_t = torch.tensor(pr, dtype=torch.int64).cuda()
            pt_t = torch.tensor(pt, dtype=torch.int64).cuda()
            nh_t = torch.tensor(nh, dtype=torch.int64).cuda()
            nr_t = torch.tensor(nr, dtype=torch.int64).cuda()
            nt_t = torch.tensor(nt, dtype=torch.int64).cuda()
            optimizer.zero_grad()
            loss = model(ph_t, pr_t, pt_t, nh_t, nr_t, nt_t)
            train_loss = train_loss + loss.item()
            loss.backward()
            optimizer.step()
            # model.hidden[0].detach_();model.hidden[1].detach_()
            # if(bn%100 == 0): print("Steps completed: ", bn)

        test_loss = 0
        sz = len(v_set)
        v_nbatches = sz//config.batch_size
        for bn in range(v_nbatches):
            vph, vpr, vpt = tuple(get_batch(v_set, config.batch_size, bn*config.batch_size))
            vnh, vnr, vnt = tuple(get_nbatch(v_set, config.batch_size, bn*config.batch_size))
            hp = torch.tensor(vph, dtype=torch.int64).cuda()
            rp = torch.tensor(vpr, dtype=torch.int64).cuda()
            tp = torch.tensor(vpt, dtype=torch.int64).cuda()
            hn = torch.tensor(vnh, dtype=torch.int64).cuda()
            rn = torch.tensor(vnr, dtype=torch.int64).cuda()
            tn = torch.tensor(vnt, dtype=torch.int64).cuda()
            optimizer.zero_grad()
            loss = model(hp, rp, tp, hn, rn, tn)
            test_loss = test_loss + loss

        scheduler.step(train_loss)

        if(epoch%config.save_epoch == 0 and epoch):
            torch.save({'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'loss':train_loss, 'batch_size':config.batch_size}, os.path.join(config.save_path,'{}_{}.pt'.format(config.train_model, epoch)))
        
        lin = 'Epoch:{:.0f}\t Train_loss:{:.2f}\t Valid_loss:{:.2f}\t Avg train_loss:{:.6f}\tAvg valid_loss:{:.6f}'.format(epoch,(train_loss),(test_loss),(train_loss/config.nbatches),(test_loss/v_nbatches))
        print(lin)
        log.write(lin+"\n")
        if(epoch%config.valid_epoch == 0 and epoch):
            testTotal, testList, _ = loadTriple(test_data=v_set)
            with torch.no_grad():
                evaluation(model, model.entity_embeddings.weight.data.cpu().numpy(), testList, tripleDict, cuda=True, res_file="results_%s.log"%(config.train_model))
        
    log.close()
    return


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    print("CUDA Available: ", torch.cuda.is_available())
    train()
    test()
