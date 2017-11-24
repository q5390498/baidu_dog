txt = open('./diff1.txt').readlines()
new_dict = open('./new_dict.txt', 'w')
for line in txt:
    line = line.strip('\n')
    id = line.split(' ')[0]
    print line
    label_1673 = line.split(' ')[1]
    label_1689 = line.split(' ')[2]
    if(len(line.split(' ')) < 4):
        break
    new_label = line.split(' ')[3].strip('\r')
    if(new_label == '?'):
        new_label = label_1673
    new_dict.write(id + ' ' + new_label + '\n')