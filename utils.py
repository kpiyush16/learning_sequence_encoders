import numpy as np
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
        r.append(x[1])
        t.append(x[2])
    return (h, r, t)

# Returns a negative batch with either head or tail corruption for entire batch
def get_nbatch(data_lst, bs, ptr):
    triples = [[],[],[]]
    for x in data_lst[ptr:bs+ptr]:
        triples[0].append(x[0])
        triples[1].append(x[1])
        triples[2].append(x[2])
    np.random.shuffle(triples[np.random.randint(2)*2])
    return(triples[0], triples[1], triples[2])

class Triple(object):
	def __init__(self, head, tail, relation):
		self.h = head
		self.t = tail
		self.r = relation

# Find the rank of ground truth tail in the distance array,
# If (head, num, rel) in tripleDict,
# skip without counting.
def argwhereTail(head, tail, rel, array, tripleDict):
	wrongAnswer = 0
	for num in array:
		if num == tail:
			return wrongAnswer
		elif (head, num, rel[0]) in tripleDict:
			continue
		else:
			wrongAnswer += 1
	return wrongAnswer

def argwhereHead(head, tail, rel, array, tripleDict):
	wrongAnswer = 0
	for num in array:
		if num == head:
			return wrongAnswer
		elif (num, tail, rel[0]) in tripleDict:
			continue
		else:
			wrongAnswer += 1
	return wrongAnswer


def loadTriple(inPath=None, test_data=None, vocab_r = None):
    tripleList = []
    if test_data is not None:
        for x in test_data:
            head = x[0]
            tail = x[2]
            rel = x[1]
            tripleList.append(Triple(head, tail, rel))
    else:
        with open(inPath, 'r') as fr:
            for line in fr:
                x = list(map(int, line.strip().split("\t")))
                head = x[0]
                tail = x[2]
                rel = [x[1]]+[vocab_r[i] for i in JulianDate_to_MMDDYYY(2014,x[3]//24+1)]
                tripleList.append(Triple(head, tail, rel))

    tripleDict = {}
    for triple in tripleList:
        tripleDict[(triple.h, triple.t, triple.r[0])] = True

    return len(tripleList), tripleList, tripleDict
