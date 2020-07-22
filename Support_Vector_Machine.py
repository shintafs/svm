__author__ = 'fakhriyani'


dataIPK = [3.25 , 3.00 , 2.74 , 1.90, 3.6 ]
dataGaji = [3000000 ,2000000 , 5000000, 5000000 , 1000000 ]
dataHasil = ['lulus' , 'lulus' , 'galulus' , 'galulus' , 'lulus']
dataTestIPK = [2.75]
dataTestGaji = [2500000]
k = ['']
k1 = ['']


jumlah_data = 5
lulus = 3
tidak_lulus = 2

for x in dataIPK :
    print(x)

for x in range (0,5) :
    if dataIPK[x] < 2.75 :
        dataIPK[x] = 'rendah'
    elif dataIPK[x] <= 3.25 :
        dataIPK[x] = 'cukup'
    else:
        dataIPK[x] = 'tinggi'


for x in dataIPK :
    print(x)


for x in dataGaji :
    print(x)

for x in range (0,5) :
    if dataGaji[x] <= 2000000 :
        dataGaji[x] = 'kecil'
    elif dataGaji[x] <= 4000000 :
        dataGaji[x] = 'sedang'
    else:
        dataGaji[x] = 'besar'


for x in dataGaji :
    print(x)

prior_lulus = lulus / jumlah_data
prior_galulus = tidak_lulus/ jumlah_data

print(prior_lulus , prior_galulus)


counter = 0

for x in dataTestIPK :
    if x < 2.75 :
        k[counter] = 'rendah'
        counter+=1
    elif x <= 3.25 :
         k[counter] = 'cukup'
         counter+=1
    else:
         k[counter] = 'tinggi'
         counter+=1

counter = 0
for x in dataTestGaji :
    if x <= 2000000 :
        k1[counter] = 'kecil'
        counter+=1
    elif x<= 4000000 :
         k1[counter] = 'sedang'
         counter+=1
    else:
        k1[counter] = 'besar'
        counter+=1

print(k, k1)


rendahL = 0
rendahTL = 0
cukupL = 0
cukupTL = 0
tinggiL = 0
tinggiTL = 0

Plulus = 0
Ptidaklulus = 0

for x in dataHasil :
    if x is 'lulus' :
        Plulus += 1
        print(Plulus)
    else :
        Ptidaklulus +=1
        print(Ptidaklulus)


print(Plulus, Ptidaklulus)

m = len(k)

print(k)
print(dataIPK)

for i in range(0,m) :
    for x in range(0, jumlah_data):
        print(dataIPK[x])
        if k[i] is 'cukup' :
            if (dataIPK[x] is 'cukup' and dataHasil[x] is 'lulus') :
                print(dataIPK[x])
                cukupL +=1
            elif (dataIPK[x] is 'cukup' and dataHasil[x] is 'galulus'):
                cukupTL +=1
        elif k[i] is 'rendah' :
            if (dataIPK[x] is 'rendah' and dataHasil[x] is 'lulus') :
                print(dataIPK[x])
                rendahL +=1
            elif (dataIPK[x] is 'rendah' and dataHasil[x] is 'galulus'):
                rendahTL +=1
        else :
            if ( dataIPK[x] is 'tinggi' and dataHasil[x] is 'lulus') :
                print(dataIPK[x])
                tinggiL +=1
            elif (dataIPK[x] is 'tinggi' and dataHasil[x] is 'galulus'):
                tinggiTL +=1
        print(x)

print(rendahL, rendahTL , cukupL , cukupTL , tinggiL , tinggiTL)


#probability

probrendahL = rendahL / Plulus
probrendahTL = rendahTL / Ptidaklulus
probcukupL = cukupL / Plulus
probcukupTL = cukupTL / Ptidaklulus
probtinggiL = tinggiL / Plulus
probtinggiTL = tinggiTL / Ptidaklulus

print(probrendahL, probrendahTL , probcukupL , probcukupTL, probtinggiL , probtinggiTL)