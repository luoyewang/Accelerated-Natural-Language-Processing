#!/usr/bin/python2.7
# Mapper in: title.basics.tsv out: [genre:str   runtimeMinues:str   count:str]
import sys
SKIPVAL = '\\N'
DELIM = '\t'
from collections import defaultdict
genre_dict = defaultdict(int)
'''
genre_dict[genre] = [sum,max, min, average,count]
'''
MAX_SIZE = 100
def map_function(parts):
    runtimeMinues = parts[-2]
    # print(runtimeMinues)
    if parts[-1] != SKIPVAL and runtimeMinues != SKIPVAL:
        for genre in parts[-1].split(','):
            yield genre,int(runtimeMinues), 1

for line in sys.stdin:
    # Split each line of the table to obtain the fields of each column
    # parts = line.strip().split(DELIM)
    # runtimeMinues = parts[-2]
    #
    # # If the "genre" field in a line is empty, skip the line
    # if parts[-1] == SKIPVAL:
    #     continue
    # if runtimeMinues != SKIPVAL:
    #
    #
    # # Each title can have multiple genres, need to split
    #     for genre in parts[-1].split(','):
    #         # Yield all genres of the title, one per output line
    #         print(genre + '\t' + runtimeMinues + '\t' + str(1) )

    for genre, runtimeMinues, count in map_function(line.strip().split(DELIM)):
        if genre in genre_dict.keys():
            #runtimeMinues = int(runtimeMinues)
            genre_dict[genre][0] += runtimeMinues
            genre_dict[genre][1] = max(genre_dict[genre][1],runtimeMinues)
            genre_dict[genre][2] = min(genre_dict[genre][2], runtimeMinues)
            genre_dict[genre][3] += count
        else:
            genre_dict[genre] = []
            genre_dict[genre].append(runtimeMinues)
            genre_dict[genre].append(runtimeMinues)
            genre_dict[genre].append(runtimeMinues)
            genre_dict[genre].append(count)

        if len(genre_dict) > MAX_SIZE:
            for genre, value in genre_dict.items():
                print('%s\t%s\t%s\t%s\t%s'%(genre,str(value[0]), str(value[1]),str(value[2]),str(value[3]) ))
#                print(genre +'\t' + str(value[0]) +'\t'+str(value[1])+ str(value))
            genre_dict.clear()
for genre,value in genre_dict.items():
    print('%s\t%s\t%s\t%s\t%s'%(genre,str(value[0]), str(value[1]),str(value[2]),str(value[3]) ))
