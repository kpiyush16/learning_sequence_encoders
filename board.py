import matplotlib.pyplot as plt

with open("train_log_TA_TransE.txt", "r") as f:
    a = [x.strip().split("\t") for x in f.readlines()]
    eps, trn, vld = [], [], []
    # print(a)
    for i in a[:100]:
        eps.append(int(i[0][6:])); trn.append(float(i[3][16:])); vld.append(float(i[4][16:]))
    # print(eps, trn, vld)
    plt.title("TA_TransE")
    plt.plot(eps, trn)
    plt.plot(eps, vld)
    plt.legend(["Avg_train_loss", "Avg_valid_loss"], loc='upper right')
    plt.savefig("TA_TransE.png")
    plt.show()