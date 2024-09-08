tc = int(input())
for x in range(tc):
    n, k = [int(i) for i in input().split()]
    str = input()
    goodness = 0
    for i in range(len(str)//2):
        if str[i] != str[len(str) - i - 1]:
            goodness += 1
    if goodness == k: print("Case #{}: {}".format(x + 1, 0))
    else:
        print("Case #{}: {}".format(x + 1, abs(goodness - k)))





