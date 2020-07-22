import twitter
import preprocessing
import lda
import os.path
from multiprocessing import process,freeze_support


def main():
    # proses pengambilan data
    twitter.IAmbilData()
    # cek kesiapan file
    if os.path.exists('data/raw.txt'):
        print("proses berlanjut ke preprosesing")
    else:
        print("data input blm ada, silakan melakukan pengambilan data")
        exit()

    preprocessing.IPreProcessing()

    if os.path.exists('data/PreStemingText.txt'):
        print("proses berlanjut ke lda")
        lda.ILdaProses()

    else:
        print("data input blm ada, silakan melakukan pengambilan data")
        exit()


if __name__ == "__main__":
    freeze_support()
    print("selamat datang di program ini")
    main()

