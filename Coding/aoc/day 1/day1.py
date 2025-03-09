import sys
import re

file = sys.argv[1]
f = open(file, "r")

Lines = f.readlines()

temp = ''
total = 0

for line in Lines:
    temp = line.strip()
    temp2 = re.sub("[^0-9]","",temp)
    temp2 = (int(temp2[0])*10) + int(temp2[len(temp2)-1])
    print(temp2)
    total += temp2

print(total)

f.close