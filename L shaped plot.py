def get_ans(i, j):
    ans1 = 0
    if i == 1 or j == 1: return 0
    ans1 = min(i, j//2)
    ans1 -= 1
    ans1 += min(j, i//2)
    ans1 -= 1
    return ans1
t = int(input())
tc = 0
while t > 0:
    r, c = [int(i) for i in input().split()]
    matrix, up, down, left, right = [], [], [], [], []
    
    for _ in range(r):
        matrix.append(list(map(int,input().split())))
        up.append([0 for i in range(c)])
        down.append([0 for i in range(c)])
        left.append([0 for i in range(c)])
        right.append([0 for i in range(c)])
    for y in range(c):
        for x in range(r):
            if not matrix[x][y]:
                continue
            down[x][y] = 1
            if x > 0: down[x][y] += down[x - 1][y]
    for y in range(c):
        for x in range(r - 1, -1, -1):
            if not matrix[x][y]:
                continue
            up[x][y] = 1
            if x + 1 < r: up[x][y] += up[x + 1][y]
    for x in range(r):
        for y in range(c):
            if not matrix[x][y]:
                continue
            right[x][y] = 1
            if y > 0: right[x][y] += right[x][y - 1]
                    
    for x in range(r):
        for y in range(c - 1, -1, - 1):
            if not matrix[x][y]:
                continue
            left[x][y] = 1
            if y + 1 < c: left[x][y] += left[x][y + 1]
    
    ans = 0
    for x in range(r):
        for y in range(c):
            if not matrix[x][y]:
                continue
            ans += get_ans(left[x][y], down[x][y])
            ans += get_ans(left[x][y], up[x][y])
            ans += get_ans(right[x][y], down[x][y])
            ans += get_ans(right[x][y], up[x][y])
    tc += 1
    print("Case #{}: {}".format(tc, ans))
    t -= 1
    