import numpy as np
import glob
import sys

sys.path.append("..")


def main():

    N = 256;
    LOOP = 100000;

    Data_Num = N*LOOP;
    data_pool = [-1,1]

    q1 = np.random.randint(0,2,(1,Data_Num));
    q2 = np.random.randint(0,2,(1,Data_Num));

    for i in range(0,N*LOOP):
        q1[0][i] = data_pool[q1[0][i]]
        q2[0][i] = data_pool[q2[0][i]]

    d = q1[0] + 1j*q2[0]
    d = d.reshape(LOOP,N)
    print(d)
    modChunks = d

    np.savez('data/modChunks.npz', data=modChunks)

if __name__ == "__main__":
    print("start...")
    main()






