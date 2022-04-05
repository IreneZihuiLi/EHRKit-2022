import json
import sys
f = open(sys.argv[1], 'rb')
lines = f.readlines()
data = []
for line in lines:
    try:
        line = line.decode('utf-8')
    except:
        line = line.decode('ISO-8859-1')
    data.append(line)
lines = [line.split('|') for line in data]
d = {}
for num,line in enumerate(lines,start = 1):
    d['text'] = line[6]
    d['section_info'] = line[5]
    d['end_pos'] = line[4]
    d['start_pos'] = line[3]
    d['usage'] = line[2]
    d['long_form'] = line[1]
    d['short_form'] = line[0]
    d['id'] = num
    with open('solr_data/{}.json'.format(num), 'w') as f:
        json.dump(d,f)
f.close()
