import pulp as pl
import collections as cl

# data
shift_requirements = [1, 4, 3, 5, 2]
workers = {
    "Melisandre": {
        "availability": [0, 1, 4],
        "cost": 20
    },
    "Bran": {
        "availability": [1, 2, 3, 4],
        "cost": 15
    },
    "Cercei": {
        "availability": [2, 3],
        "cost": 35
    },
    "Daenerys": {
        "availability": [3, 4],
        "cost": 35
    },
    "Theon": {
        "availability": [1, 3, 4],
        "cost": 10
    },
    "Jon": {
        "availability": [0, 2, 4],
        "cost": 25
    },
    "Tyrion": {
        "availability": [1, 3, 4],
        "cost": 30
    },
    "Jaime": {
        "availability": [1, 2, 4],
        "cost": 20
    },
    "Arya": {
        "availability": [0, 1, 3],
        "cost": 20
    }
}

# define the model: we want to minimize the cost
prob = pl.LpProblem("scheduling", pl.LpMinimize)

# some model variables
cost = []
vars_by_shift = cl.defaultdict(list)

for worker, info in workers.items():
    for shift in info['availability']:
        worker_var = pl.LpVariable("%s_%s" % (worker, shift), 0, 1, pl.LpInteger)
        vars_by_shift[shift].append(worker_var)
        cost.append(worker_var * info['cost'])

# set the objective to be the sum of cost
prob += sum(cost)

# set the shift requirements
for shift, requirement in enumerate(shift_requirements):
    prob += sum(vars_by_shift[shift]) >= requirement

status = prob.solve()
print("Result:", pl.LpStatus[status])
results = []
for shift, vars in vars_by_shift.items():
    results.append({
        "shift": shift,
        "workers": [var.name for var in vars if var.varValue == 1],
    })

for result in sorted(results, key=lambda x: x['shift']):
    print("Shift:", result['shift'], 'workers:', ', '.join(result['workers']))

