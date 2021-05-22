import random
#############SET THESE PARAMETERS BEFORE RUNNING########
INPUT_FILEPATH=
OUTPUT_NOISED_FILEPATH=no_ES1.txt
OUTPUT_ORIGINAL_FILEPATH=es_XX.txt
CHAR_RANGE=[97,122]#ascii range of character
CHAR_THRESHOLD=0.4
REPEAT=10

#########################
def argsort(seq):
    return sorted(range(len(seq)), key=seq.__getitem__)
f=open(INPUT_FILEPATH,'r')
fw=open(OUTPUT_NOISED_FILEPATH,'w')
fw2=open(OUTPUT_ORIGINAL_FILEPATH,'w')
z=0
k=0
for line in f:
    line=line.strip().split()
    c=[]
    sa=''.join(line)
    b=0
    for ca in sa:

        if ord(ca)<CHAR_RANGE[0] or ord(ca)>CHAR_RANGE[1]:
            b=b+1
    if b/len(sa)<CHAR_THRESHOLD:
        k=k+1
        for iag in range(REPEAT):
            c=[]
            for a in line:
                if random.random()<0.9:
                    c.append(a)
                else:
                    c.append('<mask>')
            p=[]
            for v in range(len(c)):
                p.append(v+random.random()*4)
            p=argsort(p)
    
            e=[]
            for v in p:
                e.append(c[v])
        
            fw.write(' '.join(e)+'\n')
            fw2.write(' '.join(line)+'\n')
            z=z+1
#            if z%1000000==0:
#                print(z)
           
        