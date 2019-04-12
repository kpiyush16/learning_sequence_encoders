import numpy as np
import io

def test_data_prepare(test_file, vocab_e_path="ent.vocab", vocab_r_path="rel.vocab"):
    with io.open(vocab_e_path, "r", encoding="utf8") as f1, io.open(vocab_r_path, "r", encoding="utf8") as f2, io.open(test_file, "r", encoding="utf8") as f3:
        test, test_t = [line.strip().split("\t") for line in f3.readlines()], []
        for x in test:
            [y, m, d] = x[-1].split("-")
            y = [y[0]+"y", y[1]+"y", y[2]+"y", y[3]+"y"]
            m, d = [m+"m"], [d[0]+"d", d[1]+"d"]
            test_t.append(y+m+d)
        vocab_e = {i[0]:int(i[1]) for i in [line.strip().split("\t") for line in f1.readlines()]}
        vocab_r = {i[0]:int(i[1]) for i in [line.strip().split("\t") for line in f2.readlines()]}
        te_set = []
        for (x, y) in zip(test, test_t):
            try:
                te_set.append([vocab_e[x[0]]]+[[vocab_r[x[1]]]+[vocab_r[i] for i in y]]+[vocab_e[x[2]]])
            except:
                continue
        return(te_set, vocab_e, vocab_r)

def data_prepare(train_file, valid_file, vocab=False, vocab_e_path="ent.vocab", vocab_r_path="rel.vocab"):
    with io.open(train_file, "r", encoding="utf8") as f1, io.open(valid_file, "r", encoding="utf8") as f2:
        train, train_t = [line.strip().split("\t") for line in f1.readlines()], []
        valid, valid_t = [line.strip().split("\t") for line in f2.readlines()], []
        for x in train:
            [y, m, d] = x[-1].split("-")
            y = [y[0]+"y", y[1]+"y", y[2]+"y", y[3]+"y"]
            m, d = [m+"m"], [d[0]+"d", d[1]+"d"]
            train_t.append(y+m+d)
        for x in valid:
            [y, m, d] = x[-1].split("-")
            y = [y[0]+"y", y[1]+"y", y[2]+"y", y[3]+"y"]
            m, d = [m+"m"], [d[0]+"d", d[1]+"d"]
            valid_t.append(y+m+d)

        #When vocab is not present
        if (vocab):
            print("Vocabulary is already present")
            with io.open(vocab_e_path, "r", encoding="utf8") as f3, io.open(vocab_r_path, "r", encoding="utf8") as f4:
                vocab_e = {i[0]:int(i[1]) for i in [line.strip().split("\t") for line in f3.readlines()]}
                vocab_r = {i[0]:int(i[1]) for i in [line.strip().split("\t") for line in f4.readlines()]}
        else:
            with io.open(vocab_e_path, "w", encoding="utf8") as f3, io.open(vocab_r_path, "w", encoding="utf8") as f4:
                lst = ([str(x)+'y' for x in range(10)]+[str(x)+'m' if len(str(x))==2 else '0'+str(x)+'m' for x in range(1,13)]
                +[str(x)+'d' for x in range(10)])
                vocab_e, vocab_r = {}, {}
                for i in set([x[0] for x in train+valid]+[x[2] for x in train+valid]):
                    vocab_e[i] = len(vocab_e)
                for i in lst:
                    vocab_r[i] = len(vocab_r)
                for i in set([x[1] for x in train]):
                    vocab_r[i] = len(vocab_r)
                f3.write("\n".join(["{}\t{}".format(key, vocab_e[key]) for key in vocab_e])+"\n")
                f4.write("\n".join(["{}\t{}".format(key, vocab_r[key]) for key in vocab_r])+"\n")
                # print(vocab_e)
        
        t_set = [[vocab_e[x[0]]]+[[vocab_r[x[1]]]+[vocab_r[i] for i in y]]+[vocab_e[x[2]]] for (x, y) in zip(train, train_t)]
        v_set = [[vocab_e[x[0]]]+[[vocab_r[x[1]]]+[vocab_r[i] for i in y]]+[vocab_e[x[2]]] for (x, y) in zip(valid, valid_t)]
        
        return(t_set, v_set, vocab_e, vocab_r)

    