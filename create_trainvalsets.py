import glob, os, sys
from shutil import copyfile

dtype = sys.argv[1] # train , test
rs_id = sys.argv[2]

fsrc = 'LeHou_GBMLGG_300x300_adaptive_ppi_gridwise_random_'+dtype+'_'+rs_id+'.txt'
f = open(fsrc)
content = f.readlines()
f.close()

dst = '/largeDataVolume/LeHou_GBMLGG_300x300_adaptive_ppi_gridwise_random_' + rs_id + '/' + dtype

fail_cnt = 0
for line in content:
    fpath, label = line.strip().split(',')
    filename = os.path.basename(fpath)
    if not os.path.isdir(os.path.join(dst, label)):
        os.makedirs(os.path.join(dst, label))
    try:
        copyfile(fpath, os.path.join(dst, label, filename))
    except:
        print('Failed to copy {}'.format(filename))
        fail_cnt +=1

print('No. of failed copies: {}'.format(fail_cnt))    
