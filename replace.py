f=open('/data/denoise/train.spm.no_PT1','r')
fw=open('/data/denoise/train.spm.no_PT','r')
for line in f:
    line=line.replace('▁< mas k >','<mask>')
    fw.write(line)

f.close()
fw.close()

f=open('/data/denoise/train.spm.no_ES1','r')
fw=open('/data/denoise/train.spm.no_ES','r')
for line in f:
    line=line.replace('▁< mas k >','<mask>')
    fw.write(line)
    
f.close()
fw.close()