# BIT Flip
from collections import deque


def flip_horizontally_in_place(image: bytearray, w: int, h: int) -> None:
    # Precompute the lookup table for reversing bits in a byte
    reverse_lookup = [int(f'{i:08b}'[::-1], 2) for i in range(256)]
    
    # Number of bytes per row
    bytes_per_row = w // 8

    for row in range(h):
        # Pointers to the beginning and end of the current row
        start = row * bytes_per_row
        end = start + bytes_per_row - 1

        # Swap bytes symmetrically, and reverse their bits during the swap
        while start < end:
            # Reverse bits in both bytes
            image[start], image[end] = reverse_lookup[image[end]], reverse_lookup[image[start]]
            start += 1
            end -= 1

        # If there's an odd number of bytes, reverse the middle one
        if start == end:
            image[start] = reverse_lookup[image[start]]

# Example usage:
image_data = bytearray([
    0b10100011, 0b00001111,  # Row 1: 10100011 00001111
    0b11110000, 0b11001010,  # Row 2: 11110000 11001010
    0b10010110, 0b01010101   # Row 3: 10010110 01010101
])
width = 16
height = 3

flip_horizontally_in_place(image_data, width, height)

# Output the flipped image for verification
for i in range(height):
    print(bin(image_data[i * (width // 8)])[2:].zfill(8), bin(image_data[i * (width // 8) + 1])[2:].zfill(8))


# Add an element from ( A ) to ( x ): ( x + A[i] )
# Subtract an element from ( A ) from ( x ): ( x - A[i] )
# Perform a bitwise XOR of an element from ( A ) with ( x ): ( x XOR A[i] )
# O(M⋅len(arr))
# M is the range of possible values of x explored.
# len(arr) is the number of elements in the array.
def func(arr, start, end):
    queue = deque([start])
    seen = set([start])
    ans = 0
    while queue:
        for _ in range(len(queue)):
            val = queue.popleft()
            if val == end: return ans
            for x in arr:
                for op in (x + val, val - x, x ^ val):
                    seen.add(op)
                    queue.append(op)
        ans += 1
    return -1

print(func([6,2,7,7], 10, 21))

# Write a **iterator** that returns favourite photos ids then photos ids (in the given order),
# Photos: [p10,p2,p3,p4,p5,p6,p7,p8,....]
# Favourite: [p8,p4,p10]

class PhotoIter:
    def __init__(self, photos, favourites):
        self.photos = photos
        self.favourites = favourites
        self.p_idx = 0
        self.f_idx = 0
    
    def __iter__(self):
        while self.f_idx < len(self.favourites):
            yield self.favourites[self.f_idx]
            self.f_idx += 1
        while self.p_idx < len(self.photos):
            curr = self.photos[self.p_idx]
            if curr not in self.favourites:
                yield curr
            self.p_idx += 1

photos = ['p10', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8']
favourites = [ 'p4','p8', 'p10']
photo = PhotoIter(photos, favourites)
res = list(photo)
print(res)

# Lock tolerance 

def countNumberCombinations(user, bypass, tolerance, numOptions):
    from itertools import product
    
    def get_valid_range(value, tolerance, numOptions):
        """ Compute the set of valid values within tolerance for a given digit. """
        return set((value + delta) % numOptions for delta in range(-tolerance, tolerance + 1))

    def get_valid_combinations(user, bypass, tolerance, numOptions):
        valid_sets = []
        
        for u_digit, b_digit in zip(user, bypass):
            u_valid_range = get_valid_range(u_digit, tolerance, numOptions)
            b_valid_range = get_valid_range(b_digit, tolerance, numOptions)
            # Union of valid ranges from user and bypass
            valid_sets.append(u_valid_range.union(b_valid_range))
        
        return valid_sets

    # Calculate the valid sets for each digit position
    valid_sets = get_valid_combinations(user, bypass, tolerance, numOptions)

    # Count valid combinations using the Cartesian product of valid sets
    num_combinations = 1
    for valid_set in valid_sets:
        num_combinations *= len(valid_set)
    
    return num_combinations

# Example usage:
user_combination = [0, 1, 2]
bypass_combination = [3, 4, 5]
tolerance = 2
num_options = 10

print(countNumberCombinations(user_combination, bypass_combination, tolerance, num_options))

# There is a binary tree (a sample tree was made with “/”, “\” ). Tree has total three kind of nodes.

# Normal nodes : with two children
# Leaf nodes : no children
# Special nodes : only one children.
# Multiple consecutive Special nodes make a Special nodes Chain. like (a → b → c) are the node chain, so length is 3 here.
# Followup: return all the unique chain lengths and its count in the tree.
#TC: SC:# O(N)

class Node:
    def __init__(self,val):
        self.val = val
        self.left = None
        self.right = None

def func(root):
    def dfs(root, leng):
        if not root: return 0 
        leng = max(leng, res[0])
        if not root.left and not root.right:
            return
        if root.left and root.right:
            dfs(root.left, 0)
            dfs(root.right, 0)
        elif root.left:
            dfs(root.left, leng + 1)
        
        elif root.right:
            dfs(root.right, leng + 1)
        return 
    res = [0]
    dfs(root, 0)
    return res[0]

def follow_func(root):
    def dfs(root, leng):
        if not root: return 0 
        leng = max(leng, res[0])
        if not root.left and not root.right:
            if leng > 0:
                mapp[leng] += 1
            return
        if root.left and root.right:
            dfs(root.left, 0)
            dfs(root.right, 0)
        elif root.left:
            dfs(root.left, leng + 1)
        
        elif root.right:
            dfs(root.right, leng + 1)
        return 
    res = [0]
    mapp = defaultdict(int)
    dfs(root, 0)
    return res[0], mapp
node = Node(1)
node.left = Node(2)
node.right = Node(3)
node.left.left = Node(4)
node.right.right = Node(5)
print(func(node))




# You're given a list of elements. Each element has a unique id and 3 properties. Two elements ar
# Please write a function that takes the input and returns all the duplicates.

# E1: id1, p1, p2, p3|
# E2: id2, p1, p4, p5
# E3: id3, p6, p7, p8
# in this example we should return {{idl, id2}, {id3}}

from collections import defaultdict
def func(accounts):
    def dfs(email):
        seen.add(email)
        arr.append(email)
        for i in emails[email]:
            if i not in seen:
                dfs(i)
                
    emails = defaultdict(list)
    names = defaultdict(str)
    for element in accounts:
        id = element[0]
        for p in element[1:]:
            if p in names:
                emails[id].append(names[p])
                emails[names[p]].append(id)
            else:
                names[p] = id
    # print(emails, names)
    seen = set()
    res = []
    for element in accounts:
        if element[0] not in seen:
            arr = []
            dfs(element[0])
            
            res.append(arr)
    # print(res)
    return res
data = [['id1', 'p1', 'p2', 'p3'],['id2', 'p1', 'p4', 'p5'],['id3', 'p6', 'p7', 'p8']]
print(func(data))

# DSU
# O(N)
from collections import defaultdict

class UnionFind:
    def __init__(self, size):
        self.parent = [i for i in range(size)]
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)

        if rootX != rootY:
            self.parent[rootY] = rootX
            

def mergeprops(elems):
    uf = UnionFind(len(elems))
    property_indices = defaultdict(list)

    # step1: merge groups
    for i, props_a in enumerate(elems):
        for prop in props_a:
            property_indices[prop].append(i)

    # Step2: find merged groups
    for prop, indices in property_indices.items():
        for i in range(1, len(indices)):
            uf.union(indices[i], indices[0])

    # Step 3: generate results
    result = defaultdict(list)
    for i, _ in enumerate(elems):
        result[uf.find(i)].append(i)
    

    return [duplicates for _, duplicates in result.items()]

print(mergeprops([[1,2,3],[1,4,6],[5,7,8]])) # output -> [[0, 1], [2]]
print(mergeprops([[1,2,3],[1,4,5],[5,7,8]])) # output -> [[0, 1, 2]]


import collections
class DSU:
    def __init__(self, size) -> None:
        self.parent = [i for i in range(size)]
        self.rank = [1]*size
        
    def find(self, node):
        if self.parent[node] != node:
            self.parent[node] = self.find(self.parent[node])
            
        return self.parent[node]
    
    def union(self, nodeA , nodeB):
        x = self.find(nodeA)
        y = self.find(nodeB)
        
        if x == y:
            return
        
        if self.rank[x] > self.rank[y]:
            self.parent[y] = x
            self.rank[x]+=self.rank[y]
        else:
            self.parent[x] = y
            self.rank[y]+=self.rank[x]
            
            
#first map ids to int
events = [['id1', 'p1', 'p2', 'p3'],['id2', 'p1', 'p4', 'p5'],['id3', 'p6', 'p7', 'p8']]
mp = collections.defaultdict(int)

for event in events:
    _id = event.split(" ")[0]
    
    _events = set(event.split(" ")[1:])
    
    mp[int(_id)] = _events
    
dsu = DSU(len(mp))
for k1, v1 in mp.items():
    for k2 , v2 in mp.items():
        if k1 != k2:
            
            if v1.intersection(v2):
                dsu.union(k1, k2)
                
agg = collections.defaultdict(list)

for i in range(len(mp)):
    agg[dsu.find(i)].append(i)
    
print(list(agg.values())) 

# O(1) per enqueue
# O(K) SC
# Mean of data stream of last k lelment 
from collections import deque
class MKAverage:

    def __init__(self,k: int):
        self.queue = deque()
        self.k = k
        self.curr_Sum = 0

    def addElement(self, num: int) -> None:
        self.queue.append(num)
        self.curr_Sum += num
        if len(self.queue) > self.k:
            old_sum = self.queue.popleft()
            self.curr_Sum -= old_sum
        
    def calculateMKAverage(self) -> int:
        
        return self.curr_Sum/len(self.queue)
        
arr = [50, 60, 70, 50, 100, 120, 80, 140]
K = 5
stream_calculator = MKAverage(5)
for num in arr:
    stream_calculator.addElement(num)
    mean = stream_calculator.calculateMKAverage()
    print(f"Mean after adding {num}: {mean}")

# Follow up: 
from collections import deque
from sortedcontainers import SortedList

class StreamMean:
    def __init__(self, k, x):
        self.great = SortedList()
        self.small = SortedList()
        self.gSum = 0
        self.sSum = 0
        self.gSz = x
        self.sSz = k - x
        self.nums = deque()
    #  // Add a new integer val to the stream
    # // This is to maintain the largest x elements in great and the rest in small
    def add_helper(self, val):
        self.great.add(val)
        self.gSum += val
        if len(self.great) <= self.gSz:
            return
        cur = self.great[0]
        self.great.remove(cur)
        self.gSum -= cur
        self.small.add(cur)
        self.sSum += cur
    # Remove the oldest element in the stream in O(log(k))
    # // This is to maintain the size of the stream k
    def remove_helper(self, val):
        if val in self.small:
            self.small.remove(val)
            self.sSum -= val
            return
        self.great.remove(val)
        self.gSum -= val
        if not self.small:
            return
        nval = self.small[-1]
        self.small.remove(nval)
        self.sSum -= nval
        self.add_helper(nval)
    # // O(log(k))
    def add(self, val):
        self.nums.append(val)
        if len(self.nums) > self.gSz + self.sSz:
            self.remove_helper(self.nums.popleft())
        self.add_helper(val)

    def get_mean(self):
        if len(self.small) == 0:
            return -1
        return self.sSum / len(self.small)
        

# %String manipulate
env = {
        'b': '%a%src',
        'a': 'data%c%',
        'c': 'base'
}

def replace_in_pattern(pattern):
    pattern = list(pattern)
    left = 0
    right = 0

    while left < len(pattern):
        while left < len(pattern) and pattern[left] != "%":
            left += 1
        right = left + 1
        while right < len(pattern) and pattern[right] != "%":
            right += 1
        if right < len(pattern) and pattern[right] == pattern[left] == "%":
            key = ''.join(pattern[left + 1:right])
            replacement = list(env[key])
            pattern[left:right + 1] = replacement
            left = left - 1
            right = left - 1
        left += 1

    return ''.join(pattern)

print(replace_in_pattern("hello%b% "))


# Bank Transaction

from collections import defaultdict
from itertools import combinations
def max_customers_served(initial_money, transactions):
    prev = 0
    curr_dpst = initial_money
    ans = 0
    for i in range(len(transactions)):
        curr_dpst = curr_dpst - transactions[i]
        while curr_dpst < 0 and prev <= i:
            curr_dpst += transactions[prev]
            prev += 1
        ans = max(ans, i - prev + 1)
    return ans
print(max_customers_served(5,  [-2, 5, 1, 3, 2, -3, -1, 4, 1]))

# Max PDt length
# O(n*s) + O(n^2), where s is the average length of word and n is number of words. Space complexity is O(n).
def maxProduct(words):
    bitdict = defaultdict(int)
    ans = 0
    for word in words:
        for l in word:
            bitdict[word] |= 1 << (ord(l) - 97)
    for w1, w2 in combinations(bitdict.keys(), 2):
        if bitdict[w1] & bitdict[w2] == 0:
            pdt = len(w1) * len(w2)
            if ans < pdt:
                ans = pdt
                wordres = [w1, w2]
    print(wordres)
    return ans
maxProduct(['dog', 'cat', 'pull', 'space', 'feed'] )


class Solution:
    def maxProduct(self, words: List[str]) -> int:
        n = len(words)
        best = 0
        trie = {}

        # Build a trie
		# O(N * U * logU) where U is the number of unique letters (at most 26), simplified to O(N)
        for word in words:
            node = trie
            letters = sorted(set(word))

            for char in letters:
                if char not in node:
                    node[char] = {}
                node = node[char]

            # The "None" node will store the length of the word
            node[None] = max(node.get(None, 0), len(word))

        # Loop through each word
		# O(N)
        for word in words:
            letters = set(word)
            word_len = len(word)

            # With BFS find the longest word inside the trie that does not have any common letters with current word
			# O(2^26 - 1) => O(1)
            queue = collections.deque([trie])

            while queue:
                node = queue.popleft()

                if None in node:
                    best = max(best, node[None] * word_len)

                # Explore the neighbors
                for char in node.keys():
                    if char is not None and char not in letters:
                        queue.append(node[char])

        return best
    
# You are given a binary tree and cost of each edge is given

# A-B=3, A-C=4
# B-D=1,B-E=1

# We can remove any edges and we have to calculate the minimum cost to remove the path from root to leaf node.

# Building the cost_m and deg arrays: O(E), where E is the number of edges.
# Processing each leaf node: Each node is processed once, and each edge is considered twice
#  (once for each endpoint), so O(V + E), where V is the number of nodes.
# Thus, the overall time complexity is O(V + E).

# Storing the cost matrix cost_m: O(E) for edges.
# Storing degree and minimum cost for each node: O(V).
# Thus, the overall space complexity is O(V + E).

def solve(edges, n):
    cost_m = [dict() for _ in range(n)]
    deg = [0 for _ in range(n)]
    deg[0] = 1
    for i, j, cost in edges:
        cost_m[i][j] = cost
        deg[i] += 1
        cost_m[j][i] = cost
        deg[j] += 1
    mincost = [0 for _ in range(n)]
    queue_leave = [node for node in range(n) if deg[node] == 1]
    for leaf in queue_leave:
        mincost[leaf] = float("inf")
    while queue_leave:
        node = queue_leave.popleft()
        costbelow = mincost[node]
        for parent in cost_m[node]:
            if deg[parent] > 1:
                cost = cost_m[node][parent]
                mincost[parent] += min(cost, costbelow)
                deg[parent] -= 1
                if deg[parent] == 1:
                    queue_leave.append(parent)
    return mincost[0]


# Question - Given a formula of letters with parentheses, remove all parentheses from the formula.
# Examples:
# a-(b+c) -> a-b-c
# a-(a-b) -> b
from collections import defaultdict

def simplify(Str):
    Len = len(Str)
    res = []
    i = 0
    sign_stack = [1]  # This will store the sign; 1 represents '+', -1 represents '-'
    current_sign = 1  # Overall sign

    while i < Len:
        if Str[i] == ' ':
            i += 1
            continue

        if Str[i] == '+':
            current_sign = sign_stack[-1]  # Keep the current sign
            i += 1
        elif Str[i] == '-':
            current_sign = -1 * sign_stack[-1]  # Flip the sign
            i += 1
        elif Str[i] == '(':
            # Push the current sign before '(' onto the stack
            sign_stack.append(current_sign)
            i += 1
        elif Str[i] == ')':
            # Pop the sign stack since we're done with the current parentheses
            sign_stack.pop()
            i += 1
        else:
            # Append the current character (variable or operator) with the correct sign
            if current_sign == -1:
                res.append('-' + Str[i])
            else:
                res.append('+' + Str[i])
            i += 1

    # Join the result to get a simplified form
    simplified_expr = ''.join(res)

    # Now, simplify the resulting expression by canceling out like terms
    return simplify_algebra(simplified_expr)

def simplify_algebra(expr):
    # Dictionary to store counts of variables
    var_count = defaultdict(int)

    # Tokenize and evaluate
    i = 0
    while i < len(expr):
        sign = 1
        if expr[i] == '-':
            sign = -1
            i += 1
        elif expr[i] == '+':
            i += 1

        # Assuming variable is a single character (e.g., 'a', 'b')
        if i < len(expr):
            var = expr[i]
            var_count[var] += sign
            i += 1

    # Rebuild the expression from the variable counts
    result = []
    for var, count in var_count.items():
        if count > 0:
            result.append('+' + var if count == 1 else f'+{count}{var}')
        elif count < 0:
            result.append('-' + var if count == -1 else f'{count}{var}')

    # Join the result and return
    final_result = ''.join(result)

    # Remove leading '+' if it exists
    return final_result.lstrip('+')

# Driver Code
if __name__ == '__main__':
    s1 = "a-(a-b)"
    s2 = "a-(b+c)"
    s3 = "a-(b+c)+c"

    print(simplify(s1))  # Expected output: b
    print(simplify(s2))  # Expected output: a-b-c
    print(simplify(s3))  # Expected output: a-b+c


import heapq
from collections import defaultdict, deque
# Building the frequency map: O(m)
# Building the heap: O(u log u)
# Constructing new neighborhoods: O(n * (size log u + size log size)) ≈ O(m log u + m log size) for all neighborhoods.
# Reinserting elements: O(m log u)
# Thus, the total time complexity is:O(m log u + m log size)
# Since size can be at most m, and u can also be at most m (in the worst case), 
# the overall time complexity simplifies to:O(m log m)
# Space Complexity:
# Heap: O(u), where u is the number of unique elements.
# Frequency map: O(u)
# Result list: O(m), where m is the total number of elements.
# Thus, the overall space complexity is O(m).

# Neighbour rearrange 

import heapq
from collections import defaultdict

class Pair:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __lt__(self, other):
        if self.y == other.y:
            return self.x < other.x
        return self.y > other.y

def rearrange(neighborhoods):
    n = len(neighborhoods)
    map_count = defaultdict(int)

    for lst in neighborhoods:
        for i in lst:
            map_count[i] += 1

    queue = []
    for key, value in map_count.items():
        heapq.heappush(queue, Pair(key, value))
    result = []
    for i in range(n):
        used = []
        new_neighborhood = []

        size = len(neighborhoods[i])

        while size > 0 and queue:
            pair = heapq.heappop(queue)
            new_neighborhood.append(pair.x)
            pair.y -= 1
            if pair.y > 0:
                used.append(pair)
            size -= 1

        new_neighborhood.sort()
        result.append(new_neighborhood)

        for item in used:
            heapq.heappush(queue, item)

    return result

if __name__ == "__main__":
    neighborhoods = [
        [1, 2],
        [4, 4, 7, 8],
        [4, 9, 9, 9]
    ]

    result = rearrange(neighborhoods)

    for lst in result:
        print(" ")
        for i in lst:
            print(" " + str(i), end="")


# Given a integer array 'a' with 'n' elements, find the length of the longest subsequence where a[j]-a[i]=1 given j>i . 
# # The order should remain the same.
# Create a map where the key is the last element in a subsequence and the value is the length of that subsequence.
# For each number in the arr, a[i], check if the prev element is in the map. a[i] -1 in map.
# If so add a[i] as a key to the map and map[a[i]-1] +1 as the value.

def get_longest_consecutive_subsequence(a):
    end_to_length = {}
    longest = 0
    for n in a:
        if n-1 in end_to_length:
            prev_len = end_to_length[n-1]
            end_to_length[n] = prev_len +1
        else:
            end_to_length[n] = 1

        longest = max(longest,end_to_length[n])
    
    return longest
def func(arr):
    length = [0] * len(arr)
    res = 0
    for a in arr:
          length[a] = length[a - 1] + 1
          res = max(res, length[a])
    return res
print(func([ [2, 3, 1, 4, 3, 5, 6]]))


# Given a tree having nodes with value 0 and 1. write a function to return the number of islands in tree?

# O(N)
class Node:
    def __init__(self, val):
        self.val = val
        self.children = []
        
    def addChild(self, obj):
        self.children.append(obj)
        
def count_island(root):
    def dfs(root, parent):
        if parent == None and root.val == 1 or root.val == 1 and parent.val == 0:
            cnt[0] += 1
        for nei in root.children:
            dfs(nei, root)
    cnt = [0]
    dfs(root, None)
    return cnt[0]
    
tree = Node(0)

tree.addChild(Node(0))
tree.addChild(Node(1))
tree.addChild(Node(1))

tree.children[0].addChild(Node(1))
tree.children[1].addChild(Node(0))
tree.children[2].addChild(Node(1))

tree.children[1].children[0].addChild(Node(1))
print(count_island(tree))

# There are a few nodes of the graph given where each node is associated with a specified weight. 
# To travel from node 1 to node 2, it takes some prescribed time. We need to traverse the node starting from node A,
# and return back on it within 24 time units. Find the maximum total weight that can be carried. Also, calculate the time taken.

# Assume that the connections/edges are bidirectional.


# Building the adjacency list: O(m)
# BFS exploration: O(2^n * n), where:
# 2^n represents the number of possible subsets of visited nodes.
# n represents the cost of copying the visited set and iterating over the neighbors.

from collections import deque

class Path:
    def __init__(self, visited, node, weight, time):
        self.visited = visited
        self.node = node
        self.weight = weight
        self.time = time
def gets_max(adj_list, n, weights):
    queue = deque()
    queue.append(Path(set(), 0, weights[0], 0))
    max_weight = total_time = 0
    while queue:
        curr = queue.popleft()
        # print(curr)
        node = curr.node
        time = curr.time
        weight = curr.weight
        vis = curr.visited
        if node == 0 and time <= 24:
            if weight > max_weight:
                max_weight = weight
                total_time = time
        for nei, wei in adj_list[node]:
            if time + wei <= 24:
                new_vis = vis.copy()
                new_vis.add(nei)
                new_weigh = weights[nei] + weight if nei not in vis else weight
                queue.append(Path(new_vis, nei, new_weigh, time + wei))
    return [max_weight, total_time]
    

def main():
    edges = [[0,1,2],[1,2,10],[2,3,1]]
    weight = [0,4,5,6]
    n = len(weight)
    adj_list = [[] for _ in range(n)]
    for edge in edges:
        adj_list[edge[0]].append([edge[1], edge[2]])
        adj_list[edge[1]].append([edge[0], edge[2]])
    print(gets_max(adj_list, n, weight))
    
main()

# For a stream of floating points, return any 3 points which have the distance between them less than a given value. 
# Also, remove the 3 points from memory. There can be negative points as well and the distance has to be calculated 
# between any two points. Does anyone have a solution for this?

# Example given was: 1.0,2.0,8.0,12.0,3.0
# d = 3
# If there are m numbers in the stream, the time complexity for processing the entire stream is:
# O(m * log n), where m is the number of items processed and n is the size of the SortedSet
# (which grows as more items are added).
# Float stream diff
from sortedcontainers import SortedSet

class Solution:
    def __init__(self):
        self.D = 0
        self.nums = SortedSet()

    def init(self, d):
        self.D = d
        self.nums.clear()

    def func(self, item):
        a = self.nums.bisect_left(item) - 1
        c = self.nums.bisect_right(item)

        # Case 1: Check for left and right neighbors
        if a >= 0 and c < len(self.nums):
            a_val = self.nums[a]
            c_val = self.nums[c]
            if item - a_val <= self.D and c_val - item <= self.D:
                print(f"{a_val} {item} {c_val}")
                self.nums.remove(a_val)
                self.nums.remove(c_val)
                return

        # Case 2: Check if item can form a triplet with two consecutive right neighbors
        a = self.nums.bisect_right(item)
        c = a + 1
        if a < len(self.nums) and c < len(self.nums):
            a_val = self.nums[a]
            c_val = self.nums[c]
            if a_val - item <= self.D and c_val - a_val <= self.D:
                print(f"{item} {a_val} {c_val}")
                self.nums.remove(a_val)
                self.nums.remove(c_val)
                return

        # Case 3: Check if item can form a triplet with two consecutive left neighbors
        a = self.nums.bisect_left(item) - 1
        c = a - 1
        if a >= 0 and c >= 0:
            a_val = self.nums[a]
            c_val = self.nums[c]
            if item - c_val <= self.D and a_val - c_val <= self.D:
                print(f"{c_val} {a_val} {item}")
                self.nums.remove(a_val)
                self.nums.remove(c_val)
                return

        # If no triplet is found, add the item to the set
        self.nums.add(item)

# Example usage
if __name__ == "__main__":
    solution = Solution()
    D = 5  # Set a value for D
    solution.init(D)

    # Stream of numbers to process
    stream = [1, 10, 7, -2, 8, 3, 4,15,16,18]

    for num in stream:
        solution.func(num)


# I have been given an array of trains, which has startcity, startTime and endCity, EndTime.
# and current City and currentTime, destination city and destination time was give. And I was
# supposed to answer can I reach from current city to the destination city on time.


def func(trains):
    adj = defaultdict(list)
    for startcity, endcity, startime, endtime in trains:
        adj[startcity].append((endtime - startime, endcity))
    minheap = [[currTime, startcity]]
    while minheap:
        time, city = heappop(minheap)
        if time > destinationtime:
            return False
        
        if city in vis: continue
        vis.add(city)
        for nei_t, nei_city in adj[city]:
            if nei_city not in vis:
                newtriptime = nei_t + time
            heappush(minheap, (newtriptime, nei_city))
        

# IP Belong to which country

def ip_to_int(ip):
    parts = list(map(int, ip.split('.')))
    s = (int(parts[0]) * 256**3 + int(parts[1]) * 256**2 + int(parts[2]) * 256 + int(parts[3]))
    print(s)
    return s
    
def binary_Search(ip_ranges, ip):
    low = 0
    high = len(ip_ranges) - 1
    while low <= high:
        mid = low + high >> 1
        start, end, _ = ip_ranges[mid]
        if start <= ip<= end:
            return mid
        elif ip < start:
            high = mid - 1
        else:
            low = mid + 1
    return -1
    
def func(ip_ranges, ip):
    int_ip = ip_to_int(ip)
    idx = binary_Search(ip_ranges, int_ip)
    if idx != -1:
        return ip_ranges[idx][2]
    return 'not found'
        

ip_ranges = [
        (ip_to_int("1.1.0.1"), ip_to_int("1.1.0.10"), "IND"),
        (ip_to_int("1.1.0.20"), ip_to_int("1.1.0.30"), "FR"),
        # Add more ranges here
]
a= ["1.1.0.5", "1.1.0.25", "1.1.0.35", "1.1.0.50"]
ip = "1.1.0.19"
country = func(ip_ranges, ip)
print(f"IP {ip} belongs to: {country}")


# Time taken to door
from collections import deque
from typing import List

class Solution:
    def timeTaken(self, arrival: List[int], state: List[int]) -> List[int]:
        n = len(arrival)
        ans = [0] * n
        # qs[0] := enter, qs[1] := exit
        qs = [deque(), deque()]
        time = 0
        d = 1

        for i in range(n):
            self.popQueues(time, d, arrival[i], qs, ans)
            # If the door was not used in the previous second, then the person who
            # wants to exit goes first.
            if arrival[i] > time:
                time = arrival[i]  # Forward `time` to now.
                d = 1
            qs[state[i]].append(i)

        self.popQueues(time, d, 200000, qs, ans)
        return ans

    def popQueues(self, time: int, d: int, arrivalTime: int, qs: List[deque], ans: List[int]):
        while arrivalTime > time and (qs[0] or qs[1]):
            if not qs[d]:
                d ^= 1
            ans[qs[d][0]] = time
            qs[d].popleft()
            time += 1

# -Given a list of reservations for a day, comprised of pickup and return times(from 00:00 to 23:59) assign
# a car to each reservation, using as little cars as possible, and return such list
import heapq

def time_to_minutes(time_str):
    """Convert HH:MM time format to minutes."""
    hours, minutes = map(int, time_str.split(':'))
    return hours * 60 + minutes

def car_assignment(reservations):
    # Convert the times to minutes and sort by pickup time
    reservations = [(time_to_minutes(pickup), time_to_minutes(return_)) for pickup, return_ in reservations]
    reservations.sort()
    print(reservations)
    heap = []  # min-heap to track the earliest return times
    car_assignment = []  # list to keep track of the car assignments
    car_id_counter = 1  # to keep track of the car IDs
    
    # To store the car ID assigned to each reservation
    cars_used = []
    
    for i, (pickup, return_) in enumerate(reservations):
        # If a car is available (i.e., its earliest return time is <= current pickup time)
        if heap and heap[0][0] <= pickup:
            # Reuse the car and update the heap with the new return time
            earliest_return_time, car_id = heapq.heappop(heap)
            heapq.heappush(heap, (return_, car_id))
            cars_used.append((pickup, return_, car_id))
        else:
            # Allocate a new car
            heapq.heappush(heap, (return_, car_id_counter))
            cars_used.append((pickup, return_, car_id_counter))
            car_id_counter += 1

    return cars_used

def main():
    reservations = [
        ("08:00", "09:45"),
        ("07:00", "11:00"),
        ("09:00", "10:30"),
        ("10:55", "12:30"),
        ("09:30", "11:30"),
        ("12:00", "13:00")
        
    ]
    
    assigned_cars = car_assignment(reservations)
    
    for pickup, return_, car_id in assigned_cars:
        pickup_time = f"{pickup // 60:02}:{pickup % 60:02}"
        return_time = f"{return_ // 60:02}:{return_ % 60:02}"
        print(f"Reservation from {pickup_time} to {return_time} assigned to Car {car_id}")

if __name__ == "__main__":
    main()


# Talks
# Stream word counting: O(M)
# Bucket sorting: O(U)
# Collecting top N users: O(5000 + N)
# Thus, the total time complexity is: 
# 𝑂(𝑀+𝑈+5000+𝑁)O(M+U+5000+N)

# Since 5000 is a constant, it can be omitted: 
# 𝑂(𝑀+𝑈+𝑁)
# O(M+U+N)
from collections import defaultdict
import heapq

def most_talkative_users(chat_logs, N):
    user_word_count = defaultdict(int)
    
    # Step 1: Stream word counting
    for log in chat_logs:
        user, message = log.split(':', 1)
        word_count = len(message.split())
        user_word_count[user.strip()] += word_count
    
    # Step 2: Use a heap to get the top N most talkative users
    # Since max words typed is capped at 5000, we can use bucket sorting
    bucket = [[] for _ in range(5001)]
    
    for user, count in user_word_count.items():
        bucket[count].append(user)
    
    # Collect top N users from the bucket
    result = []
    for i in range(5000, -1, -1):
        for user in bucket[i]:
            result.append((user, i))
            if len(result) == N:
                return result
    
    return result

# Example usage
chat_logs = [
    "Alice: Hey, how's it going?",
    "Bob: I'm doing well, thanks for asking!",
    "Charlie: Anyone up for a game tonight?",
    "Alice: I'm up for it, let's do it!",
    "Bob: Sure, I'm in too.",
    "Alice: Great! Let's meet at 7.",
    "Charlie: See you all then!"
]

# Get the top 2 most talkative users
N = 2
print(most_talkative_users(chat_logs, N))


# Find the total number of subsets of an array such that the LCM of elements in each subset is divisible by k.

import math
def lcm(x, y):
    return x * y // math.gcd(x, y)
def count_subset(arr, k):
    n = len(arr)
    dp = [[-1 for _ in range(k + 1)] for _ in range(n + 1)]

    def helper(n, curr_lcm):
        if n== 0:
            return 1 if curr_lcm % k == 0 else 0
        if dp[n][curr_lcm % k] != -1:
            return dp[n][curr_lcm % k]
        new_lcm = lcm(curr_lcm, arr[n - 1])
        include = helper(n - 1, new_lcm)
        exclude = helper(n - 1, curr_lcm)
        dp[n][curr_lcm % k] = include + exclude
        return dp[n][curr_lcm % k]
        
        
    helper(n, 1)

print(count_subset())


# String maths
# add(1,2)
# mul(2e3, sub(4,2))
# add(2.4, pow(2,4e4.5))

def add(a, b):
    return a + b


def sub(a, b):
    return a - b


def mul(a, b):
    return a * b


def div(a, b):
    return a / b


def convert_scientific_notation(s):
    try:
        return float(s)
    except ValueError:
        base, exp = s.split('e')
        return float(base) * (10 ** float(exp))


string_to_operator_mapping = {"add": add,
                              "mul": mul,
                              "sub": sub,
                              "pow": pow,
                              "div": div
                              }


def extract_digits(string, N, index, till=','):
    digits = ""
    while index < N and string[index] != till:
        digits += string[index]
        index += 1
    digits = convert_scientific_notation(digits)
    return digits, index


def is_operators(string, N, index):
    if index + 3 < N:
        substr = string[index:index + 3]
        return substr in string_to_operator_mapping
    return False


def calculate(string):
    operands = []
    operators = []

    N = len(string)
    index = 0
    while index < N:

        # check for operators
        if is_operators(string, N, index):
            substr = string[index:index + 3]
            operation_function = string_to_operator_mapping[substr]
            operators.append(operation_function)
            index += 3

        # check for digits: first half
        if string[index] == "(":
            index += 1
            digits, index = extract_digits(string, N, index, till=',')
            operands.append(digits)

            if index < N and string[index] == ",":
                index += 1

            # check for digits: first half
        elif string[index].isdigit():
            digits, index = extract_digits(string, N, index, till=')')
            operands.append(digits)

        # get the operators function and execute and push to operands
        elif string[index] == ")":
            operators_func = operators.pop()
            operands1, operands2 = operands.pop(), operands.pop()
            operands.append(operators_func(operands1, operands2))

            index += 1
        else:
            index += 1

    return operands[0]


strings = ['add(1,2)', 'mul(2e3, sub(4,2))', 'add(2.4, pow(2,4e4.5))']
for s in strings:
    print(calculate(s))


# Robb bank graph (leet: Maximum Path Quality of a Graph)
from collections import defaultdict, Counter
def max_robbed_bank(edges, money, T):
    def dfs(curr, time, qual):
        if curr == 0:
            res[0] = max(res[0], qual)
        for nei, t in graph[curr    ]:
            if time + t > T: continue
            if nei in vis: dfs(nei, time + t, qual)
            else:
                vis.add(nei)
                dfs(nei, time + t, qual + money[nei])
                vis.remove(nei)
        
    graph = defaultdict(list)
    vis = set()
    for u, v , w in edges:
        graph[u].append((v, w))
        graph[v].append((u, w))
    res = [0]
    vis.add(0)
    dfs(0, 0, money[0])
    return res[0]
print(max_robbed_bank([[0,1,10],[1,2,10],[0,3,10]], [5,10,15,20],30))

# Unique path to reach dest from 2 node using the least number of unique edges

from collections import deque
def unique_Edge(graph, src1, src2, dest):
    def bfs(graph, src, dest):
        queue = deque([(src, [])])
        vis = set()
        vis.add(src)
        while queue:
            node, path = queue.popleft()
            for nei in graph[node]:
                if nei not in vis:
                    vis.add(nei)
                    new_path = path + [(node, nei)]
                    if nei == dest:
                        return new_path
                    queue.append((nei, new_path))
        return []
            
    path_src1 = bfs(graph, src1, dest)
    path_src2 = bfs(graph, src2, dest)
    unqieue = set(path_src1 + path_src2)
    return len(unqieue)
#     A - m - x - y - z - p - D
# ...... |
# .......B

graph = {
    'A': ['m'],
    'm': ['A', 'B', 'x'],
    'B': ['m'],
    'x': ['y', 'm'],
    'y': ['x', 'z'],
    'z': ['y', 'p'],
    'p': ['z', 'D'],
    'D': ['p'],
}

A = 'A'
B = 'B'
D = 'D'

print(unique_Edge(graph, A, B, D))
# V+E


# 0-1bfs
from collections import deque, defaultdict

def bfs_01(graph, start_a, start_b, destination):
    # Deque for 0-1 BFS
    deque_a = deque([(start_a, None)])
    deque_b = deque([(start_b, None)])
    
    # Visited sets for A and B's BFS
    visited_a = {start_a: None}
    visited_b = {start_b: None}

    # To store unique edges
    unique_edges = set()

    while deque_a or deque_b:
        if deque_a:
            node_a, parent_a = deque_a.popleft()
            if parent_a is not None:
                unique_edges.add((min(parent_a, node_a), max(parent_a, node_a)))
            if node_a == destination:
                break

            # Explore neighbors of A
            for neighbor in graph[node_a]:
                if neighbor not in visited_a:
                    visited_a[neighbor] = node_a
                    deque_a.append((neighbor, node_a))

        if deque_b:
            node_b, parent_b = deque_b.popleft()
            if parent_b is not None:
                unique_edges.add((min(parent_b, node_b), max(parent_b, node_b)))
            if node_b == destination:
                break

            # Explore neighbors of B
            for neighbor in graph[node_b]:
                if neighbor not in visited_b:
                    visited_b[neighbor] = node_b
                    deque_b.append((neighbor, node_b))

    return len(unique_edges)

graph = {
    'A': ['m'],
    'm': ['A', 'B', 'x'],
    'B': ['m'],
    'x': ['y', 'm'],
    'y': ['x', 'z'],
    'z': ['y', 'p'],
    'p': ['z', 'D'],
    'D': ['p'],
}

A = 'A'
B = 'B'
D = 'D'
print(bfs_01(graph, A, B, D))  # Expected output: 4

# remove leaf and print order

from collections import defaultdict, deque
def func(edges, V):
    def topo(indegree, graph, vis, V, curr):
        q = deque()
        for i in range(V):
            if indegree[i] == 0:
                q.append(i)
        result = []
        while q:
            node = q.popleft()
            result.append(node)
            # Decrease indegree of adjacent vertices as the current node is in topological order
            for adjacent in graph[node]:
                indegree[adjacent] -= 1
                # If indegree becomes 0, push it to the queue
                if indegree[adjacent] == 0:
                    q.append(adjacent)
    
        # Check for cycle
        if len(result) != V:
            print("Graph contains cycle!")
            return []
        return result
                    
    indegree = defaultdict(int)
    graph = defaultdict(list)
    for i, j in edges:
        graph[i].append(j)
        indegree[j] += 1
    vis = set()
    res = topo(indegree, graph, vis, n, [])
    print(res)
    
edges = [[1,0],[2,0],[4,1],[3,1]]
n = 5
func(edges, n)
# O(N!)
def findAllTopologicalOrdering(edges):

    indegree, graph = defaultdict(int), defaultdict(list) 
    result, ans = set(), set()

    for i,j in edges:
        indegree[j] += 1 
        graph[i].append(j)
        ans.add(i)
        ans.add(j)
        
    n = len(ans)

    def backtrack(path,visited):
        if len(path) == n:
            result.add(tuple(path))
            
        for node in range(n):
            if indegree[node] == 0 and node not in visited:
                
                for neighbor in graph[node]:
                    indegree[neighbor] -= 1 
                    
                path.append(node)
                visited.add(node)
                
                backtrack(path,visited)
                
                for neighbor in graph[node]:
                    indegree[neighbor] += 1 
                    
                path.pop()
                visited.remove(node)
                
    backtrack([],set())

    return result 

# AP diff 
def get_sum(arith):
    i = 0
    res = 0
    while i < len(nums):
        j = i
        while j+1 < len(nums) and nums[j+1] - nums[j] == arith:
            j+=1
        left = i-1
        right = j+1
        for mid in range(i,j+1):
            res += nums[mid]*(mid-left)*(right-mid)
        i = j+1
    return res
    
# Example
nums = [1,2, 3]
print(get_sum(1) + get_sum(-1) - sum(nums)) #counted twice``
# print(get_sum(arr))  # Expected output: 107


# Find the number of partitions of an array such that each contiguous partition consists of atleast one negative number.

N = int(1e5)
dp = [[-1 for _ in range(2)] for _ in range(N)]

def total_partitions(idx, negative_present, a):
    if idx == n:
        return negative_present
    if dp[idx][negative_present] != -1:
        return dp[idx][negative_present]
    num = a[idx]
    # Case 1: Continue previous partition
    num_ways_with_old_partition = total_partitions(idx + 1, negative_present or (num < 0), a)
    
    num_ways_with_new_partition = 0

    # Case 2: If a negative number is present, then we can start a new partition
    if negative_present:
        num_ways_with_new_partition = total_partitions(idx + 1, num < 0, a)

    dp[idx][negative_present] = num_ways_with_old_partition + num_ways_with_new_partition
    return dp[idx][negative_present]

def solve():
    global n
    n = 2
    a =  [-1, -2]
    print(total_partitions(0, 0, a))

if __name__ == "__main__":
    solve()


# Uncle Git bought

import heapq

def max_gifts(n, gifts):
    # Sort gifts by their deadlines (day d_i)
    gifts.sort()
    
    total_cost = 0
    max_gifts_bought = 0
    min_heap = []
    
    for d_i, c_i in gifts:
        print(total_cost + c_i)
        if total_cost + c_i <= d_i:  # If we can buy this gift within the available budget
            heapq.heappush(min_heap, -c_i)  # Add the cost to the heap (use negative for max-heap)
            total_cost += c_i  # Update total cost
            max_gifts_bought += 1  # Increment the count of gifts bought
        elif min_heap and -min_heap[0] > c_i:  # If we can replace a more expensive gift with a cheaper one
            total_cost += c_i + heapq.heappop(min_heap)  # Adjust the total cost
            heapq.heappush(min_heap, -c_i)  # Push the cheaper gift into the heap
    
    return max_gifts_bought

# Example usage:
n = 4
gifts = [(1,1), (50,49), (51,20),(52,20)]

print(max_gifts(n, gifts))  # Expected output: 2

Folow: # If items showed

import heapq

def max_gifts(n, gifts):
    # Sort gifts by their deadlines (day d_i) and keep track of original indices
    gifts_with_indices = sorted([(d_i, c_i, i) for i, (d_i, c_i) in enumerate(gifts)])
    
    total_cost = 0
    max_gifts_bought = 0
    min_heap = []
    bought_items = []
    
    for d_i, c_i, idx in gifts_with_indices:
        if total_cost + c_i <= d_i:  # If we can buy this gift within the available budget
            heapq.heappush(min_heap, (-c_i, idx))  # Add the cost to the heap along with the index
            total_cost += c_i  # Update total cost
            bought_items.append(idx)  # Track this gift as bought
            max_gifts_bought += 1  # Increment the count of gifts bought
        elif min_heap and -min_heap[0][0] > c_i:  # If we can replace a more expensive gift with a cheaper one
            # Replace the most expensive gift bought so far
            total_cost += c_i + heapq.heappop(min_heap)[0]  # Adjust the total cost
            heapq.heappush(min_heap, (-c_i, idx))  # Push the cheaper gift into the heap
            # Remove the last added item to bought_items and replace it with the current one
            bought_items.pop()  # Remove the most expensive item
            bought_items.append(idx)  # Track this gift as bought
    
    return max_gifts_bought, [gifts[i] for i in bought_items]

# Example usage:
n = 4
gifts =  [(1,1), (50,49), (51,20),(52,20)]

max_count, bought_gifts = max_gifts(n, gifts)

print(f"Max number of gifts bought: {max_count}")
print(f"Gifts bought: {bought_gifts}")

# Valid signal antenna
# O(N)
mp = defaultdict(int)

def fill_first_window(b, dist):
    for i in range(dist):
        mp[b[i]] += 1
def max_valid_signals(a, b, dist):
    fill_first_window(b, dist)
    i = 0
    left = 0
    right = dist
    count = 0
    while i < len(a):
        if a[i] in mp:
            count += mp[a[i]]
        i += 1
        if i - dist > 0:
            mp[b[left]] -= 1
            if mp[b[left]] == 0:
                del mp[b[left]]
        left += 1
        right += 1
        if i + dist < len(b):
            mp[b[right]] += 1
    return count

A = [1, 3, 4, 3, 4, 5, 6]
B = [4, 1, 8, 7, 6, 3, 2]
D = 2
print(max_valid_signals(A, B, D))

# Address List Algo
# Timecomplexity for processing each address: 
# 𝑂(16): O(16)=O(1)
# Overlall O(N)
from itertools import product

def preprocess_address_list(address_list):
    address_set = set()
    
    for address in address_list:
        # For each address, generate all combinations of nulls and actual values
        for mask in product([True, False], repeat=4):  # 4 fields in each address
            modified_address = tuple(
                "null" if mask[i] else address[i]
                for i in range(4)
            )
            print(modified_address)
            address_set.add(modified_address)
    
    return address_set

def query_address(address_set, query):
    return query in address_set

# Example usage:
addressList = [
    (1, "A", "AZ", "AZZ"),
    (2, "B", "AZ", "ADD"),
    (1, "B", "AZ", "AZZ"),
    (1, "A", "AZ", "ADD"),
    (2, "B", "AZ", "QAA")
]

# Preprocess the address list into a set
address_set = preprocess_address_list(addressList)

# Query example
query = (1, "null", "AZ", "BDD")
print(query_address(address_set, query))  # Output: True

query = (1, "null", "null", "null")
print(query_address(address_set, query))  # Output: True or False based on the data

# Optimise

from collections import defaultdict
from typing import List, Optional

class TrieNode:
    def __init__(self):
        self.links = {}
        self.complete_address = None
        self.end = False

    def contains_key(self, key: str) -> bool:
        return key in self.links

    def put(self, key: str, node: 'TrieNode'):
        self.links[key] = node

    def get(self, key: str) -> Optional['TrieNode']:
        return self.links.get(key)

    def set_complete_address(self, complete_address: str):
        self.complete_address = complete_address

    def is_end(self) -> bool:
        return self.end

    def set_end(self, end: bool):
        self.end = end


class AddressLookup:
    def __init__(self):
        self.root = TrieNode()

    def add_address(self, address: List[str]):
        node = self.root
        br = []
        for line in address:
            br.append(line)
            if not node.contains_key(line):
                node.put(line, TrieNode())
            node = node.get(line)
        node.set_complete_address(" ".join(br))
        node.set_end(True)

    def query_address(self, address: List[Optional[str]]) -> List[str]:
        node = self.root
        result = []

        for i in range(len(address)):
            if address[i] is None:
                result.extend(self.dfs(i, node, address))
                return result
            else:
                if node.contains_key(address[i]):
                    node = node.get(address[i])

        if node is not None and node.is_end():
            result.append(node.complete_address)

        return result

    def dfs(self, index: int, root: TrieNode, address: List[Optional[str]]) -> List[str]:
        result = []

        for node in root.links.values():
            for i in range(index + 1, len(address)):
                if address[i] is None:
                    result.extend(self.dfs(i, node, address))
                else:
                    if not node.contains_key(address[i]):
                        break
                    else:
                        node = node.get(address[i])
            if node.is_end():
                result.append(node.complete_address)

        return result


if __name__ == "__main__":
    address_lookup = AddressLookup()

    address_lookup.add_address(["1", "A", "AZ", "AZZ"])
    address_lookup.add_address(["2", "B", "AZ", "ADD"])
    address_lookup.add_address(["1", "B", "AZ", "AZZ"])
    address_lookup.add_address(["1", "A", "AZ", "ADD"])
    address_lookup.add_address(["2", "B", "AZ", "QAA"])

    result = address_lookup.query_address(["1", None, "XZ", "AZZ"])

    for address in result:
        print("Found Address " + address)


# Task Management Sys
# Adding a task: O(log n).
# Getting the next task: O(1).
# Removing the next task: O(log n).
# Removing all tasks: O(1).
from heapq import heappush,heappop
class TaskManger:
    def __init__(self):
        self.tasks = []
        
    def add_task(self, taskid, deadline):
        heappush(self.tasks, (deadline, taskid))
    
    def get_next_task(self,):
        if self.tasks:
            return self.tasks[0]
        return -1
    
    def remove_next_Task(self):
        if self.tasks:
            return heappop(self.tasks)
        return -1
    
    def remove_all(self):
        self.tasks.clear()
        
    
task_manager = TaskManger()

task_manager.add_task(1, 4)
task_manager.add_task(2, 4)
task_manager.add_task(3, 15)

print("Next task:", task_manager.get_next_task())  # Should return task with deadline 5

task_manager.remove_next_Task()
print("Next task after removal:", task_manager.get_next_task())  

# Given a list of "greater than" pairs, return a boolean with:
# Whether the list of pair comparisons given are valid.
# a > b, b > c
# a > b, c > a, b > c
# Constructing the graph and indeg (from step 1): O(m).
# Queue initialization (from step 2): O(n).
# BFS loop (from step 3): O(m).
# The overall time complexity is:
# 𝑂(𝑚+𝑛)
# where:m is the number of edges.n is the number of nodes.

def func(edges):
    graph = defaultdict(list)
    indeg = defaultdict(int)
    for x, y in edges:
        graph[x].append(y)
        indeg[y] += 1
        indeg[x] += 0
    n = len(indeg)
    queue = deque([i for i in indeg if not indeg[i]])
    while queue:
        node = queue.popleft()
        cnt += 1
        for nei in graph[node]:
            indeg[nei] -= 1
            if indeg[nei] == 0:
                queue.append(nei)
        return cnt == n
                
                
print(func([('a', 'b'), ('c', 'a'), ('b', 'c')]))

# Bomb detonate
def func(arr):
    dp = {}
    def recur(i):
        if i >= len(arr): return 0
        maxi = recur(i + 1)
        for j in range(i + arr[i] + 1, len(arr)):
           maxi = max(maxi, arr[i] + arr[j], recur(j + arr[j] + 1))
        dp[i] = maxi
        return dp[i]
    res = 0
    res =max(res, recur(0))
    print(res)
arr = []
func(arr)

# Commit offset
def greedy_commits(arr):
    ready_offset = set()
    res = [-1] * len(arr)
    start = 0
    for i, offset in enumerate(arr):
        if offset == start:
            while start + 1 in ready_offset:
                ready_offset.remove(start + 1)
                start += 1
            res[i] = start
            start += 1
        elif offset > start:
            ready_offset.add(offset)
    return res

print(greedy_commits([2, 0, 1, 4, 3, 6])) # Output: [-1, 0, 2, -1, 4, -1]
print(greedy_commits([2, 4, 5, 0, 1, 3, 6])) # Output: [-1, -1, -1, 0, 2, 5, 6]
print(greedy_commits([2, 0, 1]))  # Output: [-1, 0, 2]
print(greedy_commits([0, 1, 2]))  # Output: [0, 1, 2]
print(greedy_commits([2, 1, 0, 5, 4]))  # Output: [-1, -1, 2, -1, -1]
        

# find the largest k digit number 

def large_k(arr, K):
    stack = []
    rem_cnt = len(arr) - K
    for val in arr:
        while stack and rem_cnt and stack[-1] < val:
            stack.pop()
            rem_cnt -= 1
        stack.append(val)
    return stack


arr = [4,9,2,9]
K = 2

print(large_k(arr, K))

# You are given n persons . M pairs of persons among them meet each other every day.
# A person x is said to "know" another person y if either x meets y every day or there is a person z that both x and y "know".
from collections import defaultdict
def func(edges, n):
    graph = defaultdict(list)
    for x, y in edges:
        graph[x].append(y)
        graph[y].append(x)
        
    vis = [False] * n
    component = []
    def dfs(node):
        vis[node] = True
        size = 0
        stack = [node]
        while stack:
            node = stack.pop()
            size += 1
            for nei in graph[node]:
                if not vis[nei]:
                    vis[nei] = True
                    stack.append(nei)
        return size
                    
                
    for i in range(n):
        if not vis[i]:
            size = dfs(i)
            component.append(size)
    total_pairs = n * (n - 1) // 2
    
    # Subtract pairs that know each other
    known_pairs = 0
    for size in component:
        known_pairs += size * (size - 1) // 2
    
    # Return the number of pairs that do not know each other
    return total_pairs - known_pairs
    
    
    
n, m = 4, 3
edges = [(0, 1), (1, 2), (2, 0)]
print(func(edges, n))

# Car Interval booking 

import heapq

def allocate_cars(intervals):
    if not intervals:
        return []
    
    # Sort intervals based on start times
    intervals.sort(key=lambda x: x[0])
    
    # Min-heap to track the end times of the intervals and their corresponding car index
    heap = []
    car_allocations = []
    
    # This will store the intervals assigned to each car
    car_intervals = []
    
    for interval in intervals:
        # If the heap is not empty and the earliest ending car can accommodate this interval
        if heap and heap[0][0] <= interval[0]:
            end_time, car_index = heapq.heappop(heap)
            # Assign this interval to the car that is now available
            car_intervals[car_index].append(interval)
            heapq.heappush(heap, (interval[1], car_index))
        else:
            # Need a new car
            car_index = len(car_allocations)
            car_allocations.append(car_index)
            car_intervals.append([interval])
            heapq.heappush(heap, (interval[1], car_index))
    
    # Formatting the output
    result = []
    for i, intervals in enumerate(car_intervals):
        intervals_str = ','.join(f"{{{start},{end}}}" for start, end in intervals)
        result.append(f"car{i + 1} : {intervals_str}")
    
    return result

# Example usage
intervals = [(1, 3), (2, 5), (6, 8), (7, 10), (9, 10)]
allocations = allocate_cars(intervals)

# Print result
for allocation in allocations:
    print(allocation)

# Given array of songs and ksong. Implement Music Player class where we can randomly choose song and play it. 
# After playing that song will be unavailable for the next k iterations.


import random
from collections import deque

class MusicPlayer:
    def __init__(self, songs, k):
        self.songs = songs  # List of songs
        self.k = k  # The number of iterations a song is unavailable after being played
        self.unavailable_songs = deque()  # Queue to track unavailable songs with their re-availability time
        self.available_songs = set(songs)  # Set of available songs
    
    def play_song(self):
        # Remove songs from the unavailable queue if they are now available
        if self.unavailable_songs:
            if len(self.unavailable_songs) >= self.k:
                available_song = self.unavailable_songs.popleft()
                self.available_songs.add(available_song)
        
        # Randomly choose a song from available songs
        if not self.available_songs:
            raise Exception("No songs are available to play.")
        
        song = random.choice(list(self.available_songs))
        
        # Play the chosen song
        print(f"Playing: {song}")
        
        # Make the chosen song unavailable for the next k iterations
        self.available_songs.remove(song)
        self.unavailable_songs.append(song)
        
    def get_available_songs(self):
        return list(self.available_songs)

# Example usage:
songs = ["A", "B", "C", "D"]
k = 1
player = MusicPlayer(songs, k)

for _ in range(3):  # Play songs for 10 iterations
    player.play_song()
    print("Available songs after play:", player.get_available_songs())
    print("-" * 40)

import random
from collections import deque

class MusicPlayer:
    def __init__(self, songs, k):
        self.recent_songs = deque(maxlen=k)
        self.available_songs = list(songs)
        self.k = k

    def play_song(self):
        if not self.available_songs:
            return None  # No songs to play

        # Get a random index
        index = random.randint(0, len(self.available_songs) - 1)

        # Select the song at the random index
        selected_song = self.available_songs[index]

        # Replace the selected song with the last song in the list
        self.available_songs[index] = self.available_songs[-1]
        self.available_songs.pop()  # Remove the last song

        # Add the selected song to the recent_songs queue
        self.recent_songs.append(selected_song)

        # If the recent_songs queue is full, make the first song available again
        if len(self.recent_songs) > self.k:
            song_to_re_add = self.recent_songs.popleft()
            self.available_songs.append(song_to_re_add)

        return selected_song

# Example usage:
songs = ["A", "B", "C", "D", "E"]
k = 2
player = MusicPlayer(songs, k)

# Simulate playing songs
for _ in range(10):
    song = player.play_song()
    print(f"Playing: {song}")
    print(f"Available Songs: {player.available_songs}")


#  items grouped into sections, each of equal size.
# if total unique elements(in all segments) in input = u and total elements = n, then TC: O(u * log u + n)
from collections import defaultdict

def get_rearranged_array(input_list):
    rearranged_array = [[] for _ in range(len(input_list))]
    freq = defaultdict(int)
    
    # Count frequency of each number
    for section in input_list:
        for el in section:
            freq[el] += 1
            # If any number appears more times than the number of sections, it's not possible
            if freq[el] > len(input_list):
                return []  # return empty array indicating it's not possible

    segment_itr = 0

    # Distribute numbers across sections
    while freq:
        number = min(freq.keys())  # get the smallest number available
        
        rearranged_array[segment_itr].append(number)
        
        freq[number] -= 1
        
        # Remove number if all instances are used
        if freq[number] == 0:
            del freq[number]
        
        segment_itr = (segment_itr + 1) % len(input_list)
    
    return rearranged_array

# Example usage:
input_list = [
    [2, 2, 6],
    [1, 3, 4],
    [2, 3, 4],
    [5, 7, 5],
    [5, 7, 6],
]

result = get_rearranged_array(input_list)

if not result:
    print("Not possible to restructure without duplications in sections.")
else:
    for section in result:
        print(section)



# RPC log earliest
from collections import deque, defaultdict
from typing import List, Tuple, Optional

class Log:
    def __init__(self, id: int, time: int, tag: str):
        self.id = id
        self.time = time
        self.tag = tag

def find_timeout(Logs: List[Log], T: int) -> Tuple[Optional[int], Optional[int]]:
    q = deque()
    s = {}

    for log in Logs:
        if log.tag == "start":
            s[log.id] = log.time
            q.append(log.id)
        else:  # log.tag == "end"
            if log.id in s and (log.time - s[log.id] >= T):
                return (log.id, log.time)
            else:
                s.pop(log.id, None)

        if q:
            while q and q[0] not in s:
                q.popleft()
            if q and (log.time - s[q[0]] >= T):
                return (q[0], log.time)

    return (-1, -1)

# Example usage
logs = [
    Log(0, 0, "start"),
    Log(1, 1, "start"),
    Log(0, 2, "end"),
    Log(2, 6, "start"),
    Log(1, 7, "end")
]
timeout = 2

result = find_timeout(logs, timeout)
print(result)  # Output: (1, 7)

D = 100
K = 12
def func(arr):
    data_pack = D//K
    remaining = D % K
    packet = [data_pack + 1 if i < remaining else data_pack for i in range(K)]
    print(packet)



# A Binary tree was given where its nodes represents as the employee and every parent node as the manager 
# of its child node. i had to implement an algorithm to determine if a manager's salary is higher than the
# sum of their employees' salaries or not Then answer i hve to give as a boolean.

class TreeNode:
    def __init__(self, salary) -> None:
        self.salary = salary
        self.left = None
        self.right = None

def check_relation(root):
    if not root: return True
    salary_check = 0
    if root.left:
        salary_check += root.salary
    if root.right:
        salary_check += root.salary
    if root.salary <= salary_check: return False
    return check_relation(root.left) and check_relation(root.right)

root = TreeNode(100)
root.left = TreeNode(50)
root.right = TreeNode(30)
root.left.left = TreeNode(20)
root.left.right = TreeNode(20)
root.right.left = TreeNode(10)
print(check_relation(root))

# You are given an array now return true or false if you can split the array in 2 halves having equal 
# sum after removing an element from the array.
def func(vec):
    total_sum = sum(vec)
    prefix_sum = set()
    current_sum = 0

    for x in vec:
        if (total_sum - x) % 2 == 0:  # even sum - possible
            half_sum = (total_sum - x) // 2
            if half_sum in prefix_sum:
                return True
        current_sum += x
        prefix_sum.add(current_sum)

    return False

def main():
    # vec = [1, 2, 5, 2, 1]
    vec = [1, 8, -1, 7, 0]
    # vec = [8, -1, 7, 0, 1]
    ans = func(vec)
    if not ans:
        vec.reverse()
        ans = func(vec)
    print(ans)

if __name__ == "__main__":
    main()


# max travel Fuel 

def func(d, k):
    n = len(d)
    memo = {}

    def dp(i, fuel):
        if i == n:
            return 0
        if (i, fuel) in memo:
            return memo[(i, fuel)]
        
        # If we rest
        if fuel == 0:
            result = dp(i + 1, 1)
        else:
            # We have two options: either rest or travel
            travel = d[i] + dp(i + 1, fuel - 1)
            rest = dp(i + 1, fuel + 1)
            result = max(travel, rest)
        
        # Store the result in memo
        memo[(i, fuel)] = result
        return result

    return dp(0, k)

# Test cases
print(func([8, 7, 1, 10, 10], 0))  # Output: 28
print(func([8, 7, 1, 10, 10], 1))  # Output: 35
print(func([8, 7, 1, 10, 10], 2))  # Output: 36
print(func([8, 7, 1, 10, 10], 3))  # Output: 37
print(func([8, 7, 1, 10, 10], 4))  # Output: 38
print(func([8, 7, 1, 10, 10], 5))  # Output: 38


# Combinations lock

def unique_combinations(master, bypass, k):
    n = len(master)
    
    # Function to get the valid range of digits for a given key digit
    def get_valid_range(digit):
        start = (digit - k) % 10
        end = (digit + k) % 10
        if start <= end:
            return set(range(start, end + 1))
        else:
            return set(range(start, 10)).union(set(range(0, end + 1)))
    
    master_ranges = [get_valid_range(int(d)) for d in master]
    bypass_ranges = [get_valid_range(int(d)) for d in bypass]
    
    master_combinations = 1
    bypass_combinations = 1
    common_combinations = 1
    
    for i in range(n):
        master_combinations *= len(master_ranges[i])
        bypass_combinations *= len(bypass_ranges[i])
        common_combinations *= len(master_ranges[i].intersection(bypass_ranges[i]))
    
    # Apply the inclusion-exclusion principle
    return master_combinations + bypass_combinations - common_combinations

# Example usage
master = "786"
bypass = "197"
tolerance = 3
print(unique_combinations(master, bypass, tolerance))  # Output should be 242

# Max sum of max and min subarray
def func(arr):
    n = len(arr)
    maxSum = float("-inf")
    for i in range(len(arr)):
        if i > 0 and arr[i - 1] <= arr[i]:
            maxSum = max(maxSum, arr[i] + arr[i - 1])
        if i < n - 1 and arr[i + 1] <= arr[i]:
            maxSum = max(maxSum, arr[i] + arr[i + 1])
    return maxSum
        
arr = [1,2,3,45]
res = func(arr)
print(res)

# Given an array of size n. Find all ordered pairs (i,j) such that 0 <= i , j < n. And a[i] - a[j] = i - j

def func(arr):
    n = len(arr)
    ans = 0
    mapp = {}
    for i in range(len(arr)):
        val = arr[i] - i
        if val in mapp:
            ans += mapp[val]
            mapp[val] += 1
        else:
            mapp[val] = 1
    return 2*ans
        
arr = [1,4,3,5]
res = func(arr)
print(res)

def find_pairs(arr):
    n = len(arr)
    freq_map = {}
    pairs = []
    
    # Step 1: Compute b[i] = a[i] - i and store the frequency in a map
    for i in range(n):
        b = arr[i] - i
        if b in freq_map:
            pairs.extend([(arr[prev_i], arr[i]) for prev_i in freq_map[b]])
            freq_map[b].append(i)
        else:
            freq_map[b] = [i]
    
    return pairs

# Example Usage:
arr = [1,4,3,4,5,8]
pairs = find_pairs(arr)
print("Pairs:", pairs)

# Given 2 arrays, you need to return the sum of specialCount.
 
def func(arr1, arr2):
    i = j = ans = 0
    while i < len(arr1) and j < len(arr2):
        if arr2[j] < arr1[i]:
            j += 1
        else:
            ans += j
            i += 1
    if i != len(arr1):
        ans += (len(arr1) - i) * len(arr2)
    return ans
  
         
arr1 = []
arr2 = []
print(func(arr1, arr2))

# Create array A from given array B counting how many smaller number exist in front of that number for the index.
from sortedcontainers import SortedSet

def get_original_array(input, n):
    st = SortedSet(range(1, n + 1))
    
    original_array = []
    
    for idx in range(len(input)):
        k = input[idx]
        
        itr = st[k]
        
        original_array.append(itr)
        
        st.remove(itr)
    
    return original_array

print(get_original_array([4,3,2,1,0], 5))

# Merged Interval with restriction

def get_merged_intervals(intervals, restricted_intervals):
    n = len(intervals)
    total_restricted_intervals = len(restricted_intervals)

    start_time_restricted = []
    end_time_restricted = []

    for interval in restricted_intervals:
        start_time_restricted.append(interval[0])
        end_time_restricted.append(interval[1])

    start_time_restricted.sort()
    end_time_restricted.sort()

    intervals.sort(key=lambda a: a[0])

    merged_intervals = []

    for idx in range(n):
        number_of_restricted_after_current = total_restricted_intervals - \
            len([x for x in start_time_restricted if x <= intervals[idx][1]])
        
        number_of_restricted_before_current = \
            len([x for x in end_time_restricted if x < intervals[idx][0]])

        if number_of_restricted_after_current + number_of_restricted_before_current == total_restricted_intervals:
            if not merged_intervals or intervals[idx][0] > merged_intervals[-1][1]:
                merged_intervals.append(intervals[idx])
            else:
                last_interval = merged_intervals.pop()
                merged_intervals.append([
                    last_interval[0],
                    max(last_interval[1], intervals[idx][1])
                ])

    return merged_intervals

activities = [[1, 5], [10, 15], [20, 25]]
restrictions = [[3, 4], [12, 14], [22, 23]]
print(get_merged_intervals(activities, restrictions))


# When Everyone Become Friends.

class Solution:
    def earliestAcq(self, logs, n):
        # Create a mapping of each unique string to an integer
        name_to_index = {}
        index = 0
        
        for log in logs:
            if log[1] not in name_to_index:
                name_to_index[log[1]] = index
                index += 1
            if log[2] not in name_to_index:
                name_to_index[log[2]] = index
                index += 1

        parent = [i for i in range(n)]

        def find(x):  # finds the id/leader of a node
            if parent[x] == x:
                return x
            parent[x] = find(parent[x])
            return parent[x]

        def Union(x, y):  # merges two disjoint sets into one set
            x = find(x)
            y = find(y)
            parent[x] = y

        logs.sort(key=lambda x: x[0])  # sorts friendships by timestamp
        components = n
        for friendship in logs:
            timestamp = friendship[0]
            x = name_to_index[friendship[1]]
            y = name_to_index[friendship[2]]
            if find(x) != find(y):  # merge two disjoint sets
                Union(x, y)
                components -= 1
            if components == 1:  # reached connected graph
                return timestamp
        return -1

logs = [
    [20190101, 'A', 'B'],
    [20190104, 'D', 'E'],
    [20190107, 'C', 'D'],
    [20190211, 'B', 'F'],
    [20190224, 'C', 'E'],
    [20190301, 'A', 'D'],
    [20190312, 'B', 'C'],
    [20190322, 'E', 'F']
]
n = 6
obj = Solution()
print(obj.earliestAcq(logs, n))
# TC = O(V + log E + V * alpha(E))
# SC = O(V + E)

# remove minimum number of characters of substring such that remaining string will have only unique characters.
def solve(s: str) -> int:
    window = set()
    i, j = 0, len(s) - 1
    ans = len(s)
    
    while i <= j:
        ci = s[i]
        cj = s[j]
        if ci in window and cj in window:
            break
        if ci != cj:
            if ci not in window:
                window.add(ci)
                i += 1
            if cj not in window:
                window.add(cj)
                j -= 1
        else:
            window.add(ci)
            i += 1
        if ans > j - i + 1:
            ans = j - i + 1
            
    return ans

print(solve('aaabcdaa'))

# Stream Container number
class DS:
    def __init__(self, k):
        self.k = k
        self.last_process = float("-inf")
        self.res = []
        self.counter = 0
        
    
    def add_container(self, st_inp):
        self.counter = (self.counter + 1) % (self.k + 1)
        if self.counter==0:
            if self.res and self.res[-1] == st_inp:
                self.res.pop()
            self.counter += 1
        if st_inp > self.last_process:
            self.last_process = st_inp
            self.res.append(st_inp)
    
    def printcheck(self):
        for i in self.res:
            print(i, end = ' ')
        
ds = DS(4)
input_data = [1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 4, 5, 5, 6, 6, 7, 7, 7, 8, 8, 8, 8]
for i in input_data:
    ds.add_container(i)
ds.printcheck()

# Balance Paranthesis Post Deletion          
def is_balance(s):
    st = []
    num = 0
    n = len(s)

    # process numbers
    for i in range(n):
        if '0' <= s[i] <= '9':
            num = num * 10 + int(s[i])
        else:
            while st and num > 0:
                st.pop()
                num -= 1
            if num > 0:
                return False
            st.append(s[i])

    while st and num > 0:
        st.pop()
        num -= 1
    if num > 0:
        return False

    s = ""
    open = close = 0
    while st:
        if st[-1] == '(':
            open += 1
        else:
            close += 1
        s += st.pop()

    s = s[::-1]
    print(s)
    return open == close

arr = ['((2))', '(()1(1))', '((((2))', '(())2', '((2)())', '(()())2', '()1()']
if __name__ == "__main__":
    for s in arr:
    # Write Python code here
        print(is_balance(s))


# Design DS for  restaurant customers waitlsit

from collections import defaultdict, OrderedDict

class WaitingQueue:
    def __init__(self):
        self.id_to_size_map = {}  # Dictionary to map group_id to size
        self.size_to_id_map = defaultdict(OrderedDict)  # Default dictionary to maintain insertion order

    def add_group(self, group_id, size):
        self.id_to_size_map[group_id] = size
        self.size_to_id_map[size][group_id] = None  # Using OrderedDict to maintain insertion order

    def remove_group(self, group_id):
        size = self.id_to_size_map[group_id]
        del self.size_to_id_map[size][group_id]  # Remove group_id from the size map
        if not self.size_to_id_map[size]:  # If the size map is empty, remove the size key
            del self.size_to_id_map[size]

    def select_group(self, n):
        best_fitting_group_size = max((size for size in self.size_to_id_map if size <= n), default=None)
        set_of_best_fitting_groups = self.size_to_id_map[best_fitting_group_size]
        earliest_arrived_best_fitting_group = next(iter(set_of_best_fitting_groups))  # Get the first group
        
        # Remove this group as they will leave the waiting queue now.
        self.remove_group(earliest_arrived_best_fitting_group)
        
        return earliest_arrived_best_fitting_group

# Main execution
if __name__ == "__main__":
    wq = WaitingQueue()
    
    wq.add_group("group1", 2)
    wq.add_group("group2", 4)
    wq.add_group("group3", 4)
    wq.add_group("group4", 4)
    wq.add_group("group5", 6)
    
    wq.remove_group("group3")
    
    selected_group_id = wq.select_group(5)
    print(selected_group_id)
    
    wq.add_group("group6", 3)
    
    selected_group_id = wq.select_group(5)
    print(selected_group_id)
    
    selected_group_id = wq.select_group(5)
    print(selected_group_id)
    
    selected_group_id = wq.select_group(5)
    print(selected_group_id)

    selected_group_id = wq.select_group(8)
    print(selected_group_id)


# deck of normal playing cards. 

# Store the number of cards for each rank in a map
# if the cnt for a rank is >= 3 return True
from collections import defaultdict
def check_criteria_1(cards):
    rank_cnt = defaultdict(int)
    for card in cards:
        rank = card[:-1]
        rank_cnt[rank] +=1
        if rank_cnt[rank] >= 3:
            return True
    return False


# For each card, map each suit to its nums; suit: [cards...]
# use a converstion to method to change 'rank' to a number
# Each time rank (R) is added for a suit. See if any of the 3 combos exist.
# R, R+1, R+2 || R-1, R, R+1 || R-2, R-1, R

def check_criteria_2(cards):
    rank_to_num = {
        'A':1,
        '2':2,
        '3':3,
        '4':4,
        '5':5,
        '6':6,
        '7':7,
        '8':8,
        '9':9,
        '10':10,
        'J':11,
        'Q':12,
        'K':13
        }
    
    suit_to_nums = defaultdict(set)
    for card in cards:
        rank = card[:-1]
        suit = card[-1]

        num = rank_to_num[rank]
        num_set = suit_to_nums[suit]

        if num-2 in num_set and num-1 in num_set:
            return True

        if num-1 in num_set and num+1 in num_set:
            return True
        
        if num+1 in num_set and num+2 in num_set:
            return True

        suit_to_nums[suit].add(num)

    return False
        

input1 = ['2C', '2S', '2H']
input2 = ['2C','3C','4C']
input3 = ['2C','3C','4H']

print(check_criteria_1(input1))
print(check_criteria_2(input1))

print(check_criteria_1(input2))
print(check_criteria_2(input2))

print(check_criteria_1(input3))
print(check_criteria_2(input3))

# FollowAll posible combination of valid deck

from collections import deque
import itertools
def get_all_combs(cards):
    rank_to_num = {
        'A':1,
        '2':2,
        '3':3,
        '4':4,
        '5':5,
        '6':6,
        '7':7,
        '8':8,
        '9':9,
        '10':10,
        'J':11,
        'Q':12,
        'K':13
        }
    
    suit_to_cards = defaultdict(list) # sliding window length 3

    rank_to_cards = defaultdict(list) # all combs of length 3

    for card in cards:
        rank = card[:-1]
        suit = card[-1]
        rank_to_cards[rank].append(card)
        suit_to_cards[suit].append(card)

    all_valid_combs = []
    for same_rank_cards in rank_to_cards.values():
        # could also use a triple for loop
        for comb in itertools.combinations(same_rank_cards,3):
            all_valid_combs.append(comb)
    
    for same_suit_cards in suit_to_cards.values():
        sorted_same_suit_cards = sorted(same_suit_cards,key = lambda card: rank_to_num[card[:-1]])
        window = deque()
        for i in range(len(sorted_same_suit_cards)):
            if len(window) > 0: # ensures that window remains contiguous
                prev_num = rank_to_num[window[-1][:-1]]
                cur_num = rank_to_num[sorted_same_suit_cards[i][:-1]]
                if prev_num+1 != cur_num:
                    window = deque()

            window.append(sorted_same_suit_cards[i])

            if len(window) > 3:
                window.popleft()
            if len(window) == 3:
                all_valid_combs.append(list(window))
    
    return all_valid_combs


input = ['2C', '2S', '2H', '3H', '4H', 'AS','5H','2D','JC','7H']
print(get_all_combs(input))


# Torch Wire node graph

from collections import defaultdict, deque

def func(edges, nodes):
    graph = defaultdict(list)
    n = len(nodes)
    for u, v in edges:
        graph[u].append(v)
        graph[v].append(u)
    queue = deque()
    for i in range(n):
        if nodes[i] == 16:
            queue.append((i, 16))
    while queue:
        curr, power = queue.popleft()
        for nei in graph[curr]:
            new_pow = power - 1
            if new_pow > nodes[nei]:
                nodes[nei] = new_pow
                queue.append((nei, new_pow))
    

nodes = [16, 0, 0, 16, 0]
edges = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 4)]
func(edges, nodes)
for i, x in enumerate(nodes):
    print(f"Node {i}: Value = {x}")

# Assign question to volunteers 

def find_maximum_matching(questions, volunteers):
    # Create a bipartite graph
    graph = {}
    for question in questions:
        question_id = question['id']
        question_tags = set(question['tags'])
        graph[question_id] = []
        
        for volunteer in volunteers:
            volunteer_id = volunteer['id']
            volunteer_tags = set(volunteer['tags'])
            
            # Check if there is a match
            if question_tags & volunteer_tags:
                graph[question_id].append(volunteer_id)
    # print(graph)
    
    # Function to perform DFS and find augmenting path
    def bpm(u, match_r, seen):
        for v in graph[u]:
            if not seen[v]:
                seen[v] = True
                if v not in match_r or bpm(match_r[v], match_r, seen):
                    match_r[v] = u
                    return True
        return False
    
    # Initialize match result
    match_r = {}
    
    # Iterate over all questions and find matches
    for question_id in graph:
        seen = {volunteer['id']: False for volunteer in volunteers}
        bpm(question_id, match_r, seen)
        # print(question_id, seen, match_r )
    # print(match_r)
    
    # Prepare the result
    assigned_questions = {}
    for volunteer_id, question_id in match_r.items():
        volunteer_name = next(volunteer['name'] for volunteer in volunteers if volunteer['id'] == volunteer_id)
        assigned_questions[question_id] = volunteer_name
    
    return assigned_questions

# Example usage
questions = [
    {"id": 1, "tags": ["MAC", "VSCODE"]},
    {"id": 2, "tags": ["PY", "AI"]},
    {"id": 3, "tags": ["JAVA", "OS"]},
    {"id": 4, "tags": ["PY", "NW"]}
]

volunteers = [
    {"id": "1", "tags": ["PY", "NW"], "name": "A"},
    {"id": "2", "tags": ["AI"], "name": "B"},
    {"id": "3", "tags": ["JAVA", "NW"], "name": "C"},
    {"id": "4", "tags": ["JAVA", "NW"], "name": "D"}
]

assigned = find_maximum_matching(questions, volunteers)
for question_id, volunteer_name in assigned.items():
    print(f"Question {question_id} is assigned to volunteer {volunteer_name}")
    

# Directory and selected files compress
#                        /b (2)(F)       /x.txt (1)(F)
#                                        /p.txt (1)(F)
#     /a (5)(F)          /c (1)(F)

#                        /d (2)(F)       /y.txt (1)(F)
#                                        /z.txt (1)(F)
# STEP 2
# the selected directories e.g.
# /a/d/y.txt
# /a/d/z.txt
# /a/b/p.txt

# Our TRIE after search i.e. after STEP 2 :

#                          /b (1)(F)        /x.txt (1)(F)
#                                           /p.txt (0)(F)
#        /a (2)(F)         /c (1)(F)

#                          /d (0)(F)        /y.txt (0)(F)
#                                           /z.txt (0)(F)
# STEP 3
# We will Search

# (1) /a/d/y.txt : Since "/d" char TRIE NODE count == 0, we will add the path /a/d to output and mark node "/d" isVisited from false to true
# => output = {"/a/d"}

# (2) /a/d/z.txt : Since "/d" char TRIE NODE with count == 0 has isVisited true we do nothing here

# (3) /a/b/p.txt : We traverse the path till TRIE NODE count == 0 which is char /p.txt => we add the path up to /p.txt and make its isVisited true
# => output = {"/a/d", "/a/b/p.txt"}

# Our TRIE after search i.e. after STEP 3 :

#                           /b (1)(F)       /x.txt (1)(F)
#                                           /p.txt (0)(T)
#         /a (2)(F)         /c (1)(F)

#                           /d (0)(T)       /y.txt (0)(F)
#                                           /z.txt (0)(F)

class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_file = False
        self.counter = 0
    
class Trie:
    def __init__(self):
        self.root = TrieNode()
        
    def insert(self, words):
        ws = self.root
        for word in words:
            ws.counter += 1
            if word not in ws.children:
                ws.children[word] = TrieNode()
            ws = ws.children[word]
        ws.is_file = True
    
    def decrement_child(self, words):
        ws = self.root
        for word in words:
            if word not in ws.children:
                return
            ws.counter -= 1
            ws = ws.children[word]
            
    def get_path(self, words):
        ws = self.root
        path = []
        for word in words:
            ws = ws.children[word]
            if ws.counter == 0 or ws.is_file:
                path.append(word)
                return '/'.join(path)
            path.append(word)
        return ''
        
    
def compress(dirs, select):
    trie = Trie()
    for file in dirs:
        word = file.split('/')
        trie.insert(word)
    
    for file in select:
        word = file.split('/')
        trie.decrement_child(word)
    res = []
    for file in select:
        word = file.split('/')
        final = trie.get_path(word)
        if final and final not in res:
            res.append(final)
    return res
    
all_files = ["a/b.txt", "b/c.txt", "b/d.txt", "c/e.txt", "c/f/a.txt", "c/f/b.txt", "c/g.txt", "d/a/b.txt"]
selected_files = ["b/c.txt", "b/d.txt", "c/e.txt", "c/f/a.txt", "c/f/b.txt", "d/a/b.txt"]


print(compress(all_files, selected_files))
# Insertion
# i.e. Step - 1 O( |Given directories| )
# Count update
# i.e. Step - 2 O( |Selected directories| )
# Search
# i.e. Step - 3 O( |Selected directories| )

# Total TC = O( |Given directories| + |Selected directories| )

# Space Complexity :
# SC = O( |Given directories| )            

# shortest byte sequence 
# TC and SC N2
from collections import deque
class TrieNode:
    def __init__(self, val = None):
        self.val = val
        self.children = {}
        self.isleaf = False
    
class Trie:
    def __init__(self):
        self.root = TrieNode(None)
    
    def addword(self, word):
        root = self.root
        for c in word:
            if c not in root.children:
                root.children[c] = TrieNode(c)
            root = root.children[c]
        root.isleaf = True
    
def find_short(s):
    trie = Trie()
    for i in range(len(s)):
        trie.addword(s[i:])
        
    root = trie.root
    queue = deque()
    queue.append(((root, '')))
    
    while queue:
        root, string = queue.popleft()
        temp = 'abcdef'
        for c in temp:
            if c not in root.children:
                return string + c
            else:
                queue.append((root.children[c], string + c))
    return ''
    
print(find_short('abcdefacbeddefdaabbccddeeff'))
                
                
    
# Coin change var  find the coins array        
def func(dp):
    orignal = []
    n = len(dp)
    for i in range(1, n):
        if dp[i] == 1:
            for j in range(n -1, i + 1, -1):
                dp[j] = dp[j] - dp[j - i]
            orignal.append(i)
    return orignal

print(func( [1, 0, 1, 0, 1, 1, 2, 1, 2, 1, 3]))
            

# Validate LHS RHS
class Main:

    @staticmethod
    def validate_util(str, idx, operation, constant, braket, close_bracket, count_braket):
        if idx == len(str):
            if count_braket != 0 or constant:
                return False
            return True
        ch = str[idx]
        if ch == '+' or ch == '-':
            if not operation:
                return False
            return Main.validate_util(str, idx + 1, False, True, True, False, count_braket)
        if ch == '(':
            if not braket:
                return False
            return Main.validate_util(str, idx + 1, False, True, True, False, count_braket + 1)
        if ch == ')':
            if not close_bracket or count_braket == 0:
                return False
            return Main.validate_util(str, idx + 1, True, False, False, True, count_braket - 1)
        if 'a' <= ch <= 'z':
            # print("here")
            if not constant:
                return False
            return Main.validate_util(str, idx + 1, True, False, False, True, count_braket)
        return Main.validate_util(str, idx + 1, operation, constant, braket, close_bracket, count_braket)

    @staticmethod
    def validate(str):
        arr = str.split("=")
        if len(arr) != 2:
            return False
        print(str, end=" ")
        return Main.validate_util(arr[0], 0, False, True, True, False, 0) and Main.validate_util(arr[1], 0, False, True, True, False, 0)

    @staticmethod
    def main():
        print(Main.validate("(a + c) = ((c + d))"))

# To run the main function
Main.main()



import itertools
import operator
# Evaluate expression in bracket
def evaluate_expression(expr):
    try:
        return eval(expr)
    except (ZeroDivisionError, SyntaxError):
        return None

def generate_expressions(nums):
    operators = ['+', '-', '*', '/']
    expressions = set()
    
    # Generate all permutations of the numbers
    for num_perm in itertools.permutations(nums):
        num_count = len(num_perm)
        
        # Generate all combinations of operators
        for ops in itertools.product(operators, repeat=num_count - 1):
            # Generate expression without parentheses
            expr = ''.join(f'{num}{op}' for num, op in zip(num_perm, ops + ('',)))
            expressions.add(expr)
            
            # Generate expressions with parentheses
            for i in range(num_count):
                for j in range(i + 2, num_count + 1):
                    sub_expr = ''.join(f'{num}{op}' for num, op in zip(num_perm[i:j], ops[i:j-1] + ('',)))
                    paren_expr = f'({sub_expr})'
                    expr_with_parens = ''.join(f'{num}{op}' for num, op in zip(num_perm[:i], ops[:i])) + paren_expr + ''.join(f'{num}{op}' for num, op in zip(num_perm[j:], ops[j-1:]))
                    expressions.add(expr_with_parens)
                    
    return expressions

def find_expression(nums, target):
    expressions = generate_expressions(nums)
    
    for expr in expressions:
        if evaluate_expression(expr) == target:
            return expr
    
    return None

# Example usage
nums = [2,3,4,5]
target = 22
result = find_expression(nums, target)

if result:
    print(f"Valid expression for target {target}: {result} = {target}")
else:
    print(f"No valid expression found for target {target}.")
# O(N!×4^(N−1))

#  Balanced Parentheses with Strings
def is_valid_string(s: str) -> bool:
    stack = []
    for ch in s:
        if ch in '{[(':
            stack.append(ch)
        elif ch == '}':
            if stack and stack[-1] == '{':
                stack.pop()
            else:
                return False
        elif ch == ']':
            if stack and stack[-1] == '[':
                stack.pop()
            else:
                return False
        elif ch == ')':
            if stack and stack[-1] == '(':
                stack.pop()
            else:
                return False

    return len(stack) == 0

# tournament is knockout format. 

from collections import deque
arr =  [1,8,4,5,2,7,3,6]
from collections import deque

def is_valid(nums):
    length = len(nums)
    if length % 2 != 0:
        return False
    queue = deque(nums)
    while len(queue) > 1:
        size = len(queue)
        for i in range(0, size, 2):
            player1 = queue.popleft()
            player2 = queue.popleft()
            if player1 + player2 != size + 1:
                return False
            queue.append(min(player1, player2))
    return True


    
print(is_valid(arr))

# here are m boys and n girls in invite girl

def bipartiteMatch(grid, u, visited, girls):
    m = len(grid)
    n = len(grid[0])
    for v in range(n):
        if grid[u][v] and not visited[v]:
            visited[v] = True
            if girls[v] < 0 or bipartiteMatch(grid, girls[v], visited, girls):
                girls[v] = u
                return True
    return False

def maximumInvitations(grid):
    m = len(grid)
    n = len(grid[0])
    girls = [-1] * n
    matches = 0

    for u in range(m):
        visited = [False] * n
        if bipartiteMatch(grid, u, visited, girls):
            matches += 1
    return matches

grid =  [[1,0,1,0],
[1,0,0,0],
[0,0,1,0],
[1,1,1,0]]
print(maximumInvitations(grid))

#  maximum possible score with as many jumps allowed.
# N^2
def get_max_score(nums, index, prev_index, dp):
    if index == len(nums):
        return 0

    result = 0
    if dp[index][prev_index] is not None:
        return dp[index][prev_index]

    # Decide to land
    land = (index - prev_index) * nums[index] + get_max_score(nums, index + 1, index, dp)
    # Decide not to land
    do_not_land = get_max_score(nums, index + 1, prev_index, dp)

    result = max(land, do_not_land)

    dp[index][prev_index] = result
    return result

arr =  [1, 2, 3, 4, 5]
n = len(arr)
dp = [[None for _ in range(n + 1)] for _ in range(n + 1)]
print(get_max_score(arr, 0, 0, dp))

# TC: N
int getMaxScore(vecror<int> &nums) {
     int result = 0;
     int maxSeenSoFar = INT_MIN;
    for (int i = size(nums) - 1; i > 0; i--) {
        maxSeenSoFar = max(maxSeenSoFar, nums[i]);
        result += maxSeenSoFar;
    }
    return result;
}





# generate a list of substrings such that while appending all of the substrings in the list should give
#  back the original string.
# Input: "GOOOOOOGLE"
# Output: ["G", "O", "OO", "OOO", "GL", "E"]

def list_of_substrings(s):
    size = len(s)
    result = []
    exists = set()
    i = 0
    while i < size:
        prefix = ""
        found_substring = False
        for j in range(i, size):
            prefix += s[j]
            if prefix not in exists:
                result.append(prefix)
                exists.add(prefix)
                found_substring = True
                i = j
                break
        if not found_substring:
            backtrack(result, exists, prefix)
            i += 1
        i += 1
    return result

def backtrack(result, exists, s):
    while result and s in exists:
        prev = result.pop()
        exists.remove(prev)
        s = s + prev

    result.append(s)
    exists.add(s)

s = 'GOOOOOOGLE'
print(list_of_substrings(s))



# Given an array of integers nums, find indexes [i, j] such that the 
# subarray sum maximum and nums[i] is equal to nums[j]
# Kadane algo

# Example:
def main():
    nums = [0, 1, 0, 2, 1,-1,0]


    # Initialize a regular dictionary
    mp = {}
    prefix = 0
    res = float('-inf')
    start = end = -1
    n = len(nums)

    for i in range(n):
        sum_ = prefix + nums[i]

        # Check if nums[i] exists in the dictionary
        if nums[i] in mp:
            # Extract the previously stored prefix sum and index
            prev_prefix, prev_index = mp[nums[i]]
            # Calculate the subarray sum and update result if it's greater
            if res < sum_ - prev_prefix:
                res = sum_ - prev_prefix
                start = prev_index
                end = i
            # Update the dictionary with the new prefix if it's smaller
            if prev_prefix > prefix:
                mp[nums[i]] = (prefix, i)
        else:
            # Add new entry to the dictionary
            mp[nums[i]] = (prefix, i)

        # Update the prefix sum
        prefix = sum_

    print(res, start, end)

if __name__ == "__main__":
    main()



# Bank has 1 unit of money intially.
# Customer transactions : [1, -3, 5, -2, 1]

# answer = 3

# Bank starts with customer with deposit of 5
# 1+ 5 = 6
# 6 - 2 = 4
# 4 + 1 =5
def max_customers_served(init_amount, transactions):
    left, right = 0, 0
    cash_available =  init_amount
    cust_served, max_cust_served  = 0, 0
    while right < len(transactions):
        cash_available += transactions[right]
        if cash_available >= 0:
            cust_served  = right - left + 1
            if cust_served > max_cust_served:
                max_cust_served = cust_served
        else:
            while cash_available < 0 and left <= right:
                cash_available -= transactions[left]
                left += 1
        right += 1
    return max_cust_served
print(max_customers_served(5,  [-2, 5, 1, 3, 2, -3, -1, 4, 1]))



# maximum number using digits(1 to 9) associated cost
def solve():
    A = [4, 3, 2, 8, 7, 1, 4, 3, 4]
    N = len(A)
    S = 15
    dp = [["" for _ in range(S + 1)] for _ in range(N + 1)]

    for i in range(1, N + 1):
        for j in range(1, S + 1):
            curr = str(i)
            cost = A[i - 1]
            dp[i][j] = dp[i - 1][j]

            if j >= cost:
                x = dp[i][j]
                y = curr + dp[i][j - cost]

                if len(y) > len(x):
                    dp[i][j] = y
                elif len(x) == len(y):
                    dp[i][j] = max(dp[i][j], y)

    print(dp[N][S])

solve()

# N^2
def solve(a, x, s, d, dp):
    n = len(a)
    if dp[s][d] == 1:
        return -1
    dp[s][d] = 1
    # otherwise we will move in the direction d
    # d == 0 --> left 
    # d == 1 --> right 
    if d == 0:
        for i in range(s - 1, -1, -1):
            if a[i] == a[s] + 1:
                a[s] += x
                return solve(a, x, i, 1 - d, dp)
    else:
        for i in range(s + 1, n):
            if a[i] == a[s] + 1:
                a[s] += x
                return solve(a, x, i, 1 - d, dp)

    # if we can't jump anywhere 
    return s


# Nlogn
def solve_map(a, x, s, d, dp, mp):
    n = len(a)
    if dp[s][d] == 1:
        return -1
    dp[s][d] = 1
    # otherwise we will move in the direction d
    # d == 0 --> left 
    # d == 1 --> right 
    if (a[s] + 1) not in mp:
        return s
    nidx = -1
    if d == 0:
        for x in mp[a[s] + 1]:
            if x >= s:
                break
            if x < s:
                nidx = x
    else:
        for x in mp[a[s] + 1]:
            if x > s:
                nidx = x
                break

    if nidx == -1:
        return s

    # store the a[s]+x in map 
    if a[s] + x not in mp:
        mp[a[s] + x] = set()
    mp[a[s] + x].add(s)
    mp[a[s]].discard(s)
    a[s] = a[s] + x
    # now we have to move at index nidx 
    return solve_map(a, x, nidx, 1 - d, dp, mp)

a = [3, 4, 2, 2, 7]
n = len(a)
from collections import defaultdict
mp = defaultdict(set)
for i in range(n):
    mp[a[i]].add(i)
dp1 = [[0] * 2 for _ in range(n)]
ans1 = solve_map(a, 4, 2, 0, dp1, mp)  # Note: solve_map function needs to be defined
print(ans1)


# on-call rotation schedule
# nlogn
def get_oncall_rotations(oncall):
	oncall_start_ends = []

	# Add oncall rotation as start and end separately
	# so that we know which oncall are starting and ending later

	for person, start, end in oncall:
		oncall_start_ends.append([start, -1, person])
		oncall_start_ends.append([end, 1, person])

	# Sort the array based on timestamps
	oncall_start_ends = sorted(oncall_start_ends, key = lambda x: (x[0], x[1]))

	curr = set([])
	start_ts = -1

	result = []

	for ts, sign, person in oncall_start_ends:
		# Add the first element to set and set the star timestamp
		if start_ts == -1:
			start_ts = ts
			curr.add(person)
		# Add the people result if there are any people in set
		elif curr:
			result.append([start_ts, ts, curr.copy()])
		
		# sign == -1 corresponds to start so add the person to stack 
		# -1 corresponds to end so remove from stack
		if sign == -1:
			curr.add(person)
		else:
  			curr.remove(person)
		
		start_ts = ts

	return result


oncall = [
	["Abby", 1, 10],
	["Ben", 5, 7],
	["Carla", 6, 12],
	["David", 15, 17],
  ]

print(get_oncall_rotations(oncall))

# Stone Maximum total score 
# check at every step, should we come direct or through last stone
public int maxPoints(int[] nums){
        for(int i=2;i< nums.length; i++) {
            nums[i] = Math.max(nums[i-1]  + nums[i], nums[i] * i);
        }
        return nums[nums.length-1];
    }


# Coast graph

def is_coast(grid, point):
    m, n = len(grid), len(grid[0])
    dirs = [[1,0], [0,1], [0,-1], [-1,0]]
    surrounding_water = []
    x, y = point

    for row, col in dirs:
        nr, nc = row + x, col + y
        if 0 <= nr < m and 0 <= nc < n and grid[nr][nc] == '.':
            surrounding_water.append((nr, nc))
    
    seen = set()
    def dfs(r, c):
        if (r, c) in seen or grid[r][c] == 'X':
            return False
        
        if r + 1 >= m or r - 1 < 0 or c + 1 >= n or c - 1 < 0:
            return True
        
        seen.add((r,c))
        res = (
            dfs(r+1, c) or
            dfs(r-1, c) or
            dfs(r, c+1) or
            dfs(r, c-1)
        )
        
        return res


    for row, col in surrounding_water:
        if (row, col) not in seen and dfs(row, col):
            return True
    
    return False

grid = [
    ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'],
    ['.', '.', '.', 'X', '.', 'X', '.', '.'],
    ['.', '.', '.', 'X', 'X', '.', 'X', '.']
]
coordinate = (1, 3)
print(is_coast(grid, coordinate))


# N cities and M roads that travel between the given pair of cities and time it takes to travel
#  that road. Also we are given a list of favourite cities L 
import heapq
import math

def find_closest_fav_city(n, roads, L, S):
    # Step 1: Build the graph
    graph = {i: [] for i in range(n)}
    for u, v, time in roads:
        graph[u].append((v, time))
        graph[v].append((u, time))  # Assuming it's an undirected graph
    
    # Step 2: Dijkstra's algorithm to find shortest time from source S
    dist = [math.inf] * n
    dist[S] = 0
    pq = [(0, S)]  # (distance, city)
    
    while pq:
        curr_time, city = heapq.heappop(pq)
        
        # Skip if we've already found a shorter way to this city
        if curr_time > dist[city]:
            continue
        
        # Explore neighbors
        for neighbor, time in graph[city]:
            new_time = curr_time + time
            if new_time < dist[neighbor]:
                dist[neighbor] = new_time
                heapq.heappush(pq, (new_time, neighbor))
    
    # Step 3: Find the closest favorite city
    min_time = math.inf
    closest_city = -1
    for fav_city in L:
        if dist[fav_city] < min_time:
            min_time = dist[fav_city]
            closest_city = fav_city
    
    return closest_city if closest_city != -1 else None

# Example usage:
n = 6  # Number of cities
roads = [
    (0, 1, 10), 
    (0, 2, 5), 
    (1, 3, 2), 
    (2, 3, 1), 
    (3, 4, 4), 
    (4, 5, 3)
]
L = [3, 5]  # List of favorite cities
S = 0       # Source city

print(find_closest_fav_city(n, roads, L, S))  # Output: 3

# Early Stopppage

import heapq
import math

def find_closest_fav_city(n, roads, L, S):
    # Step 1: Build the graph
    graph = {i: [] for i in range(n)}
    for u, v, time in roads:
        graph[u].append((v, time))
        graph[v].append((u, time))  # Assuming it's an undirected graph
    
    # Convert the list of favorite cities into a set for quick lookup
    favorite_cities = set(L)
    
    # Step 2: Dijkstra's algorithm to find shortest time from source S
    dist = [math.inf] * n
    dist[S] = 0
    pq = [(0, S)]  # (distance, city)
    
    while pq:
        curr_time, city = heapq.heappop(pq)
        
        # If the current city is one of the favorite cities, we can stop early
        if city in favorite_cities:
            return city
        
        # Skip if we've already found a shorter way to this city
        if curr_time > dist[city]:
            continue
        
        # Explore neighbors
        for neighbor, time in graph[city]:
            new_time = curr_time + time
            if new_time < dist[neighbor]:
                dist[neighbor] = new_time
                heapq.heappush(pq, (new_time, neighbor))
    
    # If we never found a favorite city, return None
    return None

# Example usage:
n = 6  # Number of cities
roads = [
    (0, 1, 10), 
    (0, 2, 5), 
    (1, 3, 2), 
    (2, 3, 1), 
    (3, 4, 4), 
    (4, 5, 3)
]
L = [3, 5]  # List of favorite cities
S = 0       # Source city

print(find_closest_fav_city(n, roads, L, S))  # Output: 3

# a fix vertex V to be travelled in the way to favourite city

def dijkstra(n, graph, src):
    dist = [math.inf] * n
    dist[src] = 0
    pq = [(0, src)]  # (distance, city)
    
    while pq:
        curr_time, city = heapq.heappop(pq)
        
        # Skip if we've already found a shorter way to this city
        if curr_time > dist[city]:
            continue
        
        # Explore neighbors
        for neighbor, time in graph[city]:
            new_time = curr_time + time
            if new_time < dist[neighbor]:
                dist[neighbor] = new_time
                heapq.heappush(pq, (new_time, neighbor))
    
    return dist

def find_closest_fav_city_with_vertex(n, roads, L, S, V):
    # Step 1: Build the graph
    graph = {i: [] for i in range(n)}
    for u, v, time in roads:
        graph[u].append((v, time))
        graph[v].append((u, time))  # Assuming it's an undirected graph
    
    # Step 2: Run Dijkstra from S to V
    dist_from_S = dijkstra(n, graph, S)
    
    # Step 3: Run Dijkstra from V to all favorite cities
    dist_from_V = dijkstra(n, graph, V)
    
    # Step 4: Find the favorite city that can be reached the fastest from S via V
    min_time = math.inf
    closest_city = -1
    for fav_city in L:
        total_time = dist_from_S[V] + dist_from_V[fav_city]
        if total_time < min_time:
            min_time = total_time
            closest_city = fav_city
    
    return closest_city if closest_city != -1 else None

# Example usage:
n = 6  # Number of cities
roads = [
    (0, 1, 10), 
    (0, 2, 5), 
    (1, 3, 2), 
    (2, 3, 1), 
    (3, 4, 4), 
    (4, 5, 3)
]
L = [3, 5]  # List of favorite cities
S = 0       # Source city
V = 3       # Fixed vertex to pass through

print(find_closest_fav_city_with_vertex(n, roads, L, S, V))  # Output: 5



# minimum number of non-visitable nodes. 
import heapq

def bfs_min_non_visitable(graph, source, destination, non_visitable):
    non_visitable_set = set(non_visitable)
    
    # Priority queue stores (non_visitable_count, path_length, node)
    pq = [(0, 0, source)]
    visited = {}
    
    while pq:
        non_visitable_count, path_length, node = heapq.heappop(pq)
        
        # If we reached the destination, return the number of non-visitable nodes
        if node == destination:
            return path_length, non_visitable_count
        
        if node in visited and visited[node] <= non_visitable_count:
            continue
        
        visited[node] = non_visitable_count
        
        for neighbor in graph[node]:
            new_non_visitable_count = non_visitable_count + (1 if neighbor in non_visitable_set else 0)
            
            if neighbor not in visited or visited[neighbor] > new_non_visitable_count:
                heapq.heappush(pq, (new_non_visitable_count, path_length + 1, neighbor))
    
    return -1, -1  # No path found

# Example usage
graph = {
    1: [2, 3],
    2: [1, 4],
    3: [1, 4, 5],
    4: [2, 3, 5],
    5: [3, 4]
}

source = 1
destination = 5
non_visitable = [3]  # Node 3 is non-visitable

path_length, non_visitable_count = bfs_min_non_visitable(graph, source, destination, non_visitable)
print(f"Minimum Path Length: {path_length}, Non-Visitable Nodes Count: {non_visitable_count}")


# Unavailble guys make it available Meeting Block

def availableDays(n, calendar, D, P):
    # Initialize days array with size D + 2
    days = [0] * (D + 2)
    
    # Update the days array based on calendar
    for l, r in calendar:
        if l > D:
            continue
        days[l] += 1
        days[min(D + 1, r + 1)] -= 1
    
    # Accumulate the values to get actual busy days
    for i in range(1, D + 1):
        days[i] += days[i - 1]
    
    # List for all fully available days
    allAvailable = []
    for i in range(1, D + 1):
        if days[i] == 0:
            allAvailable.append(i)
    
    # Follow-up: List for days where at least P people are available
    pAvailable = []
    for i in range(1, D + 1):
        if n - days[i] >= P:
            pAvailable.append(i)
    
    return allAvailable, pAvailable
# 2nd Follow up
class Block:
    def __init__(self, personId, startDay, endDay):
        self.personId = personId
        self.startDay = startDay
        self.endDay = endDay

def find_available_periods(blocks, total_people, P, X):
    # Step 1: Create an availability array
    max_day = max([block.endDay for block in blocks])  # Find the max day to define array size
    availability = [total_people] * (max_day + 1)  # Initially all people are available every day

    # Step 2: Mark unavailable days in the availability array
    for block in blocks:
        for day in range(block.startDay, block.endDay + 1):
            availability[day] -= 1  # Decrease availability when someone is unavailable

    # Step 3: Apply sliding window technique to find periods of at least X days where at least P people are available
    i = 0
    for j in range(len(availability)):
        if availability[j] < P:  # If fewer than P people are available
            if j - i >= X:  # If the window length is at least X
                print(f"Period of availability: {i} to {j - 1}")
            i = j + 1  # Move the start of the window to the next day

    # Check the last window if it satisfies the conditions
    if len(availability) - i >= X:
        print(f"Period of availability: {i} to {len(availability) - 1}")

# Example usage:
blocks = [
    Block(1, 1, 5),  # Person 1 is unavailable from day 1 to 5
    Block(2, 2, 6),  # Person 2 is unavailable from day 2 to 6
    Block(3, 5, 10)  # Person 3 is unavailable from day 5 to 10
]

# Suppose we have 3 people in total, we need at least 2 people available for at least 3 consecutive days
total_people = 3
P = 2
X = 3

find_available_periods(blocks, total_people, P, X)


# Logger Rate limiter
# unique message is printed at most every 10 seconds,
# Ignore any messages that are duplicated within 10 seconds interval.
# O(1)
import time

class Logger:
    def __init__(self):
        self.message_timestamp_map = {}
    
    def should_print_message(self, timestamp: int, message: str) -> bool:
        if message not in self.message_timestamp_map:
            # If the message is new, print and store the timestamp
            self.message_timestamp_map[message] = timestamp
            print(message, timestamp)
            return True
        else:
            last_printed_timestamp = self.message_timestamp_map[message]
            if timestamp - last_printed_timestamp >= 10:
                # If 10 seconds have passed, print and update the timestamp
                self.message_timestamp_map[message] = timestamp
                print(message, timestamp)
                return True
        # Otherwise, do not print the message
        return False

# Example usage
# logger = Logger()
# logger.should_print_message(1, "foo")  # prints "foo"
# logger.should_print_message(2, "bar")  # prints "bar"
# logger.should_print_message(3, "foo")  # prints nothing
# logger.should_print_message(11, "foo") # prints "foo"
# logger.should_print_message(15, "bar") # prints nothing
# logger.should_print_message(21, "foo") # prints "foo"


# 2. rate limitor that limits requests to K times for every 30 seconds.

class Logger():
    def __init__(self, limit, rate):
        self.data = {}
        self.limit = limit
        self.rate = rate

    def shouldPrintMessage(self, timestamp, message):

        if message in self.data:
            messageTimeout, numberOfRequestsLeft = self.data[message]

            if messageTimeout > timestamp and numberOfRequestsLeft > 0:
                self.data[message] = (messageTimeout, numberOfRequestsLeft - 1)
                return True
            if messageTimeout <= timestamp:
                self.data[message] = (self.limit + messageTimeout, self.rate - 1)
                return True                
            else:
                return False
        
        else:
            self.data[message] = (self.limit + timestamp, self.rate - 1)
            return True
logger = Logger(limit = 30, rate = 3)

# print(logger.shouldPrintMessage(1, 'foo')) # True
# print(logger.shouldPrintMessage(2, 'bar')) # True
# print(logger.shouldPrintMessage(3, 'foo')) # True
# print(logger.shouldPrintMessage(8, 'bar')) # True
# print(logger.shouldPrintMessage(15, 'foo')) # True
# print(logger.shouldPrintMessage(17, 'foo')) # False
# print(logger.shouldPrintMessage(19, 'foo')) # False
# print(logger.shouldPrintMessage(19, 'bar')) # True
# print(logger.shouldPrintMessage(21, 'bar')) # False
# print(logger.shouldPrintMessage(31, 'foo')) # True
# print(logger.shouldPrintMessage(32, 'bar')) # True


# elements in the array that start with the prefix value 
# O(k log n), where n is the number of strings and k is the length of the prefix.
def binary_search_leftmost(strings, prefix):
    start = 0
    end = len(strings)

    leftmost_pos = -1  # leftmost position of string with prefix 'prefix'

    while start < end:
        mid = start + (end - start) // 2
        if strings[mid][:len(prefix)] == prefix:
            leftmost_pos = mid
            end = mid  # could there be another to the left
        elif strings[mid][:len(prefix)] < prefix:
            start = mid + 1
        else:
            end = mid

    return leftmost_pos


def binary_search_rightmost(strings, prefix):
    start = 0
    end = len(strings)

    rightmost_pos = -1  # rightmost position of string with prefix 'prefix'

    while start < end:
        mid = start + (end - start) // 2
        if strings[mid][:len(prefix)] == prefix:
            rightmost_pos = mid
            start = mid + 1  # could there be another to the right
        elif strings[mid][:len(prefix)] < prefix:
            start = mid + 1
        else:
            end = mid

    return rightmost_pos


def better_binary_search(strings, prefix):
    i = binary_search_leftmost(strings, prefix)
    j = binary_search_rightmost(strings, prefix)

    if i == -1:  # we haven't found a string with given prefix
        return 0

    return (j - i + 1)
array = ["bomb", "book", "g","gift", "go", "goal", "goat", "gum","xray","yellow","zebra"]
prefix = "go"

print(better_binary_search(array,prefix))

