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
import calendar

def JulianDate_to_MMDDYYY(y,jd):
    month = 1
    day = 0
    while jd - calendar.monthrange(y,month)[1] > 0 and month <= 12:
        jd = jd - calendar.monthrange(y,month)[1]
        month = month + 1
    d = jd
    if jd//10 == 0:
        d = '0'+str(jd)
    return ([x+'y' for x in list(str(y))] +[str(month)+'m' if month//10==1 else '0'+str(month)+'m'] 
            +[x+'d' for x in list(str(d))])

def get_batch(data_lst, bs, ptr):
    h, r, t = [], [], []
    for x in data_lst[ptr:bs+ptr]:
        h.append(x[0])
        # r.append(x[1])
        r.append([x[1]]+[vocab_r[i] for i in JulianDate_to_MMDDYYY(2014,x[3]//24+1)])
        t.append(x[2])
    return (h, r, t)

# Returns a negative batch with either head or tail corruption for entire batch
def get_nbatch(data_lst, bs, ptr):
    triples = [[],[],[]]
    for x in data_lst[ptr:bs+ptr]:
        triples[0].append(x[0])
        triples[1].append([x[1]]+[vocab_r[i] for i in JulianDate_to_MMDDYYY(2014,x[3]//24+1)])
        # triples[1].append(x[1])
        triples[2].append(x[2])
    np.random.shuffle(triples[np.random.randint(2)*2])
    return(triples[0], triples[1], triples[2])

class Triple(object):
	def __init__(self, head, tail, relation):
		self.h = head
		self.t = tail
		self.r = relation

def loadTriple(inPath, fileName, test_data=None):
    tripleList = []
    if test_data is not None:
        for x in test_data:
            head = x[0]
            tail = x[2]
            rel = [x[1]]+[vocab_r[i] for i in JulianDate_to_MMDDYYY(2014,x[3]//24+1)]
            tripleList.append(Triple(head, tail, rel))
    else:
        with open(os.path.join(inPath, fileName), 'r') as fr:
            for line in fr:
                x = list(map(int, line.strip().split("\t")))
                head = x[0]
                tail = x[2]
                rel = [x[1]]+[vocab_r[i] for i in JulianDate_to_MMDDYYY(2014,x[3]//24+1)]
                tripleList.append(Triple(head, tail, rel))

    # tripleDict = {}
    # for triple in tripleList:
    #     tripleDict[(triple.h, triple.t, triple.r)] = True

    return len(tripleList), tripleList

class Config(object):
    def __init__(self):
        self.train_model = "TA_TransE"
        self.testFlag = True
        self.hidden_size = 100
        self.nbatches = 0
        self.batch_size = 512
        self.entity_nums = 0
        self.relation_nums = 0
        self.trainTimes = 501
        self.margin = 1.0
        self.lmbda = 0
        self.lr = 1e-3

class TA_TransE(nn.Module):
    def __init__(self, config):
        super(TA_TransE, self).__init__()
        self.config = config
        self.entity_embeddings = nn.Embedding(config.entity_nums, config.hidden_size)
        self.relation_embeddings = nn.Embedding(config.relation_nums, config.hidden_size)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True)
        self.hidden = self.init_hidden()
        self.init_weights()
        
    def init_hidden(self):
        return (Variable(torch.zeros(1, self.config.batch_size, self.config.hidden_size)).cuda(),
                Variable(torch.zeros(1, self.config.batch_size, self.config.hidden_size)).cuda())

    def init_weights(self):
        nn.init.xavier_uniform(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform(self.relation_embeddings.weight.data)
        F.normalize(self.entity_embeddings.weight.data, p = 2)
        F.normalize(self.relation_embeddings.weight.data, p = 2)
    
    def loss_calc(self, pos, neg):
        criterion = nn.MarginRankingLoss(self.config.margin, False).cuda()
        y = Variable(torch.Tensor([-1])).cuda()
        return criterion(pos, neg, y)

        # criterion = nn.CrossEntropyLoss().cuda()
        # y = Variable(torch.Tensor([0, 1])).cuda()
        # return criterion(torch.Tensor([neg, pos]).cuda())

    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        pos_h_e = self.entity_embeddings(pos_h)
        temp = self.relation_embeddings(pos_r)
        out, _ = self.lstm(temp, self.hidden)
        pos_r_e = out[:, -1, :]
        pos_t_e = self.entity_embeddings(pos_t)
        
        neg_h_e = self.entity_embeddings(neg_h)
        temp1 = self.relation_embeddings(neg_r)
        out1, _ = self.lstm(temp1, self.hidden)
        neg_r_e = out1[:, -1, :]
        neg_t_e = self.entity_embeddings(neg_t)

        p_score = torch.abs(pos_h_e + pos_r_e - pos_t_e)
        n_score = torch.abs(neg_h_e + neg_r_e - neg_t_e)
        p_score = p_score.view(-1, 1, 100)
        n_score = n_score.view(-1, 1, 100)

        pos = torch.sum(torch.mean(p_score, 1), 1)
        neg = torch.sum(torch.mean(n_score, 1), 1)

        return self.loss_calc(pos, neg)

    def predict(self, h, r, t):
        ph_e = self.entity_embeddings(h)
        pt_e = self.entity_embeddings(t)
        pr_e = self.lstm(self.relation_embeddings(r))[0][:,-1,:]
        # pr_e = self.relation_embeddings(Variable(torch.from_numpy(r)))
        # predict = torch.sum(torch.abs(ph_e + pr_e - pt_e), 1)
        return (ph_e, pr_e, pt_e)

class TA_DistMult(nn.Module):
    def __init__(self, config):
        super(TA_DistMult, self).__init__()
        self.config = config
        self.entity_embeddings = nn.Embedding(config.entity_nums, config.hidden_size)
        self.relation_embeddings = nn.Embedding(config.relation_nums, config.hidden_size)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, batch_first=True)
        self.softplus=nn.Softplus().cuda()
        self.hidden = self.init_hidden()
        self.init_weights()
        
    def init_hidden(self):
        return (Variable(torch.zeros(1, 2*self.config.batch_size, self.config.hidden_size)).cuda(),
                Variable(torch.zeros(1, 2*self.config.batch_size, self.config.hidden_size)).cuda())

    def init_weights(self):
        nn.init.xavier_uniform(self.entity_embeddings.weight.data)
        nn.init.xavier_uniform(self.relation_embeddings.weight.data)
        F.normalize(self.entity_embeddings.weight.data, p = 2)
        F.normalize(self.relation_embeddings.weight.data, p = 2)
    
    def loss_calc(self,h,t,r):
        return torch.sum(h*t*r,1,False)
    
    def loss_func(self,loss,regul):
        return loss+self.config.lmbda*regul
    
    def forward(self, pos_h, pos_r, pos_t, neg_h, neg_r, neg_t):
        h, r, t = torch.cat((pos_h, neg_h)), torch.cat((pos_r, neg_r)), torch.cat((pos_t, neg_t))
        y = torch.Tensor([1]*len(pos_h)+[-1]*len(neg_h)).cuda()
        e_h, e_t = self.entity_embeddings(h), self.entity_embeddings(t)
        temp = self.relation_embeddings(r)
        out,_ = self.lstm(temp)
        e_r = out[:,-1,:]
        res = self.loss_calc(e_h,e_t,e_r)
        tmp = self.softplus(- y*res)
        loss = torch.mean(tmp)
        regul = torch.mean(e_h ** 2) + torch.mean(e_t ** 2) + torch.mean(e_r ** 2)
        loss = self.loss_func(loss,regul)
        return loss
    
    def predict(self, predict_h, predict_t, predict_r):
        p_e_h=self.entity_embeddings(Variable(torch.from_numpy(predict_h)))
        p_e_t=self.entity_embeddings(Variable(torch.from_numpy(predict_t)))
        p_e_r = self.lstm(self.relation_embeddings(r))[0][:,-1,:]
        # p_e_r=self.relation_embeddings(Variable(torch.from_numpy(predict_r)))
        p_score=-self.loss_calc(p_e_h,p_e_t,p_e_r)
        return p_score.cpu().data.numpy()	

def evaluation_transE(model, entity_embeddings, testList, L1_flag=False, bs = 500):
    totalRank, hit10Count, tripleCount = 0, 0, 0
    print("Stating Test----------------->")
    print("Test Triple Count", len(testList))
    print("Hits@10  Average Rank  Triples Processed")
    for i in range(0, len(testList), bs):
        headList = torch.tensor([triple.h for triple in testList[i:i+bs]], dtype=torch.int64)
        tailList = torch.tensor([triple.t for triple in testList[i:i+bs]], dtype=torch.int64)
        relList = torch.tensor([triple.r for triple in testList[i:i+bs]], dtype=torch.int64)
        h_e, r_e, t_e = model.predict(headList, relList, tailList)
        c_t_e = h_e + r_e
        c_h_e = t_e - r_e

        if L1_flag == True:
            dist = pairwise_distances(c_t_e, entity_embeddings, metric='manhattan')
        else:
            dist = pairwise_distances(c_t_e, entity_embeddings, metric='euclidean')

        rankArrayTail = np.argsort(dist, axis=1)
        rankListTail = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(tailList, rankArrayTail)]
        isHit10ListTail = [x for x in rankListTail if x < 10]

        if L1_flag == True:
            dist = pairwise_distances(c_h_e, entity_embeddings, metric='manhattan')
        else:
            dist = pairwise_distances(c_h_e, entity_embeddings, metric='euclidean')
        rankArrayHead = np.argsort(dist, axis=1)
        rankListHead = [int(np.argwhere(elem[1]==elem[0])) for elem in zip(headList, rankArrayHead)]
        # print(rankArrayHead)
        # print(rankListHead)
        # break
        isHit10ListHead = [x for x in rankListHead if x < 10]

        totalRank += (sum(rankListTail) + sum(rankListHead))/2
        hit10Count += (len(isHit10ListTail) + len(isHit10ListHead))/2
        tripleCount += (len(rankListTail) + len(rankListHead))/2
        if(i%100 == 0):
            print("{:.6f}, {:.6f}, {:.0f}".format(hit10Count*100/(i+bs), totalRank/(i+bs), tripleCount))

    print((hit10Count, totalRank, tripleCount))
    return hit10Count, totalRank, tripleCount


with open("icews/combined.txt", "r") as f1:
    data = [list(map(int, line.strip().split("\t")[:-1])) for line in f1.readlines()]
    

    lst = ([str(x)+'y' for x in range(10)]+[str(x)+'m' if len(str(x))==2 else '0'+str(x)+'m' for x in range(1,13)]
        +[str(x)+'d' for x in range(10)])
    vocab_, vocab_r = set([x[1] for x in data]), {}
    for i in lst:
        vocab_r[i] = len(vocab_r)
    for i in vocab_:
        vocab_r[str(i)] = len(vocab_r)
    np.random.shuffle(data)
    ent_nums = len(set([x[0] for x in data]+[x[2] for x in data]))
    rel_nums = len(vocab_r)
    with open("./icews/train_mod.txt", "r") as f2, open("./icews/test_mod.txt", "r") as f3:
        data = [list(map(int, line.strip().split("\t"))) for line in f2.readlines()]
        test_data = [list(map(int, line.strip().split("\t"))) for line in f3.readlines()]


warnings.filterwarnings("ignore")
print("CUDA Available: ", torch.cuda.is_available())

def test():
    config = Config()
    config.relation_nums = rel_nums
    config.entity_nums = ent_nums
    print("Total number of Entities:", config.entity_nums)
    print("Total number of Relations:", config.relation_nums-32)

    print("Evaluating %s"%(config.train_model))
    testTotal, testList = loadTriple('./icews/', 'test_mod.txt')

    train_model = TA_TransE(config)
    ckpt = torch.load("./icews/TA_TransE_495.pt")
    train_model.load_state_dict(ckpt['model_state_dict'])
    train_model.eval()
    with torch.no_grad():
        evaluation_transE(train_model, train_model.entity_embeddings.weight.data.cpu().numpy(), testList)

    return

def train():
    config = Config()
    config.relation_nums = rel_nums
    config.entity_nums = ent_nums
    print("Total number of triples: ",len(data))
    print("Total number of Entities:", config.entity_nums)
    print("Total number of Relations:", config.relation_nums-32)
    sz = int(0.1*(len(data)))
    t_set, v_set = data[:-sz], data[-sz:]
    if(config.nbatches):
        config.batch_size = len(t_set) // config.nbatches
    else:
        config.nbatches = len(t_set)//config.batch_size
    
    device = torch.device("cuda")

    if(config.train_model == "TA_DistMult"):
        model = TA_DistMult(config).to(device)
    elif(config.train_model == "TA_TransE"):
        model = TA_TransE(config).to(device)
    else:
        print("Please Enter Valid Model Name");exit(1)
    # optimizer = optim.SGD(model.parameters(), lr=1e-2)
    optimizer = optim.Adam(model.parameters(), lr=config.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, verbose=True)

    print("Starting Train ----------->")
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
        for bn in range(sz//config.batch_size):
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
        print('Epoch:{:.0f}\t Train_loss:{:.2f}\t Valid_loss:{:.2f}\t Avg train_loss:{:.6f}\tAvg valid_loss:{:.6f}'.format(epoch,(train_loss),(test_loss),(train_loss/len(t_set)),(test_loss/len(v_set))))
        if(epoch%5 == 0):
            torch.save({'epoch':epoch,
            'model_state_dict':model.state_dict(),
            'optimizer_state_dict':optimizer.state_dict(),
            'loss':train_loss, 'batch_size':config.batch_size}, os.path.join("./icews/",'{}_{}.pt'.format(config.train_model, epoch)))

    # print(model.state_dict())

# train()
test()
