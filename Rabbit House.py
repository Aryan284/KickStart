from heapq import heappush, heappop
X = [-1, 0, 0, 1]
Y = [0, -1, 1, 0]
tc = int(input())
t = 0
while tc > 0:
	r, c = [int(i) for i in input().split()]
	matrix = []
	for i in range(r):
		matrix.append(list(map(int, input().split())))
	heap = []
	for i in range(r):
		for j in range(c):
			heappush(heap, (-matrix[i][j], i, j))
	ans = 0
	while heap:
		h, i, j = heappop(heap)
		height = -h
		for k in range(4):
			x, y = i + X[k], j + Y[k]
			if 0 <= x < r and 0 <= y < c:
				if matrix[x][y] >= height - 1:
					continue
				diff = height - 1 - matrix[x][y]
				ans += diff
				ele = heappop(heap)
				ele = height - 1
				heappush(heap, (-ele, x, y))
	t += 1
	print("Case #{}: {}".format(t, ans))
	tc -= 1

