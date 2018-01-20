import pulp as pl

# declare some variables
# each variable is a binary variable that is either 0 or 1
# 1 means the item will go into the knapsack
a = pl.LpVariable("a", 0, 1, pl.LpInteger)
b = pl.LpVariable("b", 0, 1, pl.LpInteger)
c = pl.LpVariable("c", 0, 1, pl.LpInteger)
d = pl.LpVariable("d", 0, 1, pl.LpInteger)

# define the problem
prob = pl.LpProblem("knapsack", pl.LpMaximize)

# objective function - maximize value of objects in knapsack
prob += 5 * a + 7 * b + 2 * c + 10 * d

# constraint - weight of objects cannot exceed 15
prob += 2 * a + 4 * b + 7 * c + 10 * d <= 15

status = prob.solve()  # solve using the default solver, which is cbc
print(pl.LpStatus[status])  # print the human-readable status

# print the values
print("a", pl.value(a))
print("b", pl.value(b))
print("c", pl.value(c))
print("d", pl.value(d))
