import sys
import re

# def number_to_string(argument):
#     match argument:
#         case "one":
#             return 1
#         case "two":
#             return 2
#         case 2:
#             return "two"
#         case default:
#             return "something"

file = sys.argv[1]
f = open(file, "r")

Lines = f.readlines()

temp = ''
big = 0
small = 1000000
word_to_number = {"one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9"}
numbers = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
for line in Lines:
    temp = line.strip()
    print(temp)
    #for word, number in word_to_number.items():
        #temp = temp.replace(word, number)

    #check for the lowest index word which is a number
    #check for the highset index word which is a number
    
    for x in numbers:
        txt = str(x)
        if temp.find(txt) != -1:
            small = min(small, temp.find(txt))
        print(x)
    print(small)
    print(big)
print(big * 10 + small)

f.close