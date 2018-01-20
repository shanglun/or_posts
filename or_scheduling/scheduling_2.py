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
    "Cersei": {
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

ban_list = {
    ("Daenerys", "Jaime"),
    ("Daenerys", "Cersei"),
    ("Jon", "Jaime"),
    ("Jon", "Cersei"),
    ("Arya", "Jaime"),
    ("Arya", "Cersei"),
    ("Arya", "Melisandre"),
    ("Jaime", "Cersei")
}

DOTHRAKI_MAX = 10
DOTHRAKI_COST = 45

# define the model: we want to minimize the cost
prob = pl.LpProblem("scheduling", pl.LpMinimize)

# some model variables
cost = []
vars_by_shift = cl.defaultdict(list)
vars_by_worker = cl.defaultdict(list)
dothrakis_by_shift = {}

for worker, info in workers.items():
    for shift in info['availability']:
        worker_var = pl.LpVariable("%s_%d" % (worker, shift), 0, 1, pl.LpInteger)
        # store some variable data so we can implement the ban constraint
        var_data = (worker,)
        vars_by_shift[shift].append((worker_var, var_data))
        # store vars by variable so we can implement the max shift constraint
        vars_by_worker[worker].append(worker_var)
        cost.append(worker_var * info['cost'])

# Add the Dothraki variables. Shieraki gori ha yeraan!
for shift in range(len(shift_requirements)):
    dothraki_var = pl.LpVariable('dothraki_%d' % shift, 0, DOTHRAKI_MAX, pl.LpInteger)
    cost.append(dothraki_var * DOTHRAKI_COST)
    dothrakis_by_shift[shift] = dothraki_var

# set the objective to be the sum of cost
prob += sum(cost)

# set the shift requirements
for shift, requirement in enumerate(shift_requirements):
    prob += sum([var[0] for var in vars_by_shift[shift]]) + dothrakis_by_shift[shift] >= requirement

# set the ban requirements
for shift, vars in vars_by_shift.items():
    for var1 in vars:
        for var2 in vars:
            worker_pair = var1[1][0], var2[1][0]
            if worker_pair in ban_list:
                prob += var1[0] + var2[0] <= 1

# set the labor standards:
for worker, vars in vars_by_worker.items():
    prob += sum(vars) <= 2

status = prob.solve()
print("Result:", pl.LpStatus[status])
results = []
for shift, vars in vars_by_shift.items():
    results.append({
        "shift": shift,
        "workers": [var[1][0] for var in vars if var[0].varValue == 1],
        "dothrakis": dothrakis_by_shift[shift].varValue
    })

for result in sorted(results, key=lambda x: x['shift']):
    print("Shift:", result['shift'], 'workers:', ', '.join(result['workers']), 'dothrakis hired:', int(result['dothrakis']))
