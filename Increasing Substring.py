tc = int(input())
t = 0
while tc > 0:
    n = int(input())
    s = input()
    dp = [1] * n
    for i in range(1, len(s)):
        if s[i] > s[i - 1]:
            dp[i] = dp[i - 1] + 1
        else: dp[i] = dp[-1]
    res = (' '.join(str(dp[i]) for i in range(len(dp))))
    t += 1
    print("Case #{}: {}".format(t, res))
    tc -= 1