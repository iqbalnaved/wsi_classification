import matplotlib.pyplot as plt
import sys

cnn = sys.argv[1]

src = 'log/'+cnn+'_patch_selection.log'

f = open(src)
content = f.readlines()
f.close()

comb_id_list, ps_list, data_sz_list, ps_percent_list = [], [], [],[]
for line in content:
    if 'iter=' in line:
        parts = line.split(' ')
        comb_id = int(parts[0].strip(':').split('=')[1])
        ps = int(parts[1])
        dsz = int(parts[5])
        comb_id_list.append(comb_id)
        ps_list.append(ps)
        data_sz_list.append(dsz)
        ps_per = (ps / dsz) * 100.
        ps_percent_list.append(ps_per)
        
        
plt.plot(comb_id_list, ps_percent_list)
plt.xlim([1, 125])
plt.xlabel('Threshold combination ID')
plt.ylabel('Percentage of patches selected')
plt.title(cnn+' feature based patch selection')
plt.savefig('plots/'+cnn+'_patch_selection_plot.png')
plt.show()
        
        
