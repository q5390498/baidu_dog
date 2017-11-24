import os
import glob

files = []#glob.glob('./submit/17*.txt')
# files.append('./submit/173.txt')
files.append('./submit/1696.txt')
files.append('./submit/1689_revise.txt')
files.append('./submit/revise_1673.txt')

print len(files)
print files
label_count = {}
for f in files:
    f = open(f).readlines()
    for line in f:
        filename = line.strip('\n').split('\t')[-1]
        label = line.split('\t')[0]
        #print filename, label
        if filename not in label_count:
            label_count[filename] = {}
            label_count[filename][label] = 1
        else:
            if label not in label_count[filename]:
                label_count[filename][label] = 1
            else:
                label_count[filename][label] += 1
print label_count

vote_label = {}
for key in label_count:
    min_num = 0
    for k in label_count[key]:
        if min_num < label_count[key][k]:
            vote_label[key] = k
            min_num = label_count[key][k]

print vote_label
file = files[0]
file = open(file).readlines()
vote_pred = open('vote_submit.txt', 'w')
for line in file:
    key = line.strip('\n').split('\t')[-1]
    vote_pred.write('15' + '\t' + key + '\n')

