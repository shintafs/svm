import os, sys
j = 581
for i in range(1975,1994):
    os.renames("doc/z/y/tweet%d.txt" %i , "doc/z/y/tweet%d.txt" %j)
    j+=1
    print(j)