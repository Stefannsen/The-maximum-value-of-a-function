ch1 = [1,2,3,4,5]
ch2 = [0,0,0,0,0]
u = 0
ch1[:u-1], ch2[:u-1] = ch2[u:], ch1[u:]
print(ch1)
print(ch2)