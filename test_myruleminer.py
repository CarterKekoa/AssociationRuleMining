import numpy as np
from mysklearn.myruleminer import MyAssociationRuleMiner

# note: order is actual/received student value, expected/solution
def test_association_rule_miner_fit():
    # Lab Task 4 -------------------------------

    # toy market basket analysis dataset
    transactions = [
        ["b", "c", "m"],
        ["b", "c", "e", "m", "s"],
        ["b"],
        ["c", "e", "s"],
        ["c"],
        ["b", "c", "s"],
        ["c", "e", "s"],
        ["c", "e"]
    ]
    
    # fit the data
    marm = MyAssociationRuleMiner()
    rules = marm.fit(transactions)
    print()
    marm.print_association_rules()

    # Find all rules for S = {b,c,m} and compute confidence. Keep rules assuming $minconf$ = 80%.
    assert(rules[0]["lhs"] == ['c', 'm'])
    assert(rules[0]["rhs"] == ['b'])
    assert(rules[0]["confidence"] == 1.0)
    assert(rules[1]["lhs"] == ['b', 'm'])
    assert(rules[1]["rhs"] == ['c'])
    assert(rules[1]["confidence"] == 1.0)
    assert(rules[2]["lhs"] == ['m'])
    assert(rules[2]["rhs"] == ['b', 'c'])
    assert(rules[2]["confidence"] == 1.0)
    assert(rules[3]["lhs"] == ['b', 's'])
    assert(rules[3]["rhs"] == ['c'])
    assert(rules[3]["confidence"] == 1.0)
    assert(rules[4]["lhs"] == ['e', 's'])
    assert(rules[4]["rhs"] == ['c'])
    assert(rules[4]["confidence"] == 1.0)
    assert(rules[5]["lhs"] == ['m'])
    assert(rules[5]["rhs"] == ['b'])
    assert(rules[5]["confidence"] == 1.0)
    assert(rules[6]["lhs"] == ['e'])
    assert(rules[6]["rhs"] == ['c'])
    assert(rules[6]["confidence"] == 1.0)
    assert(rules[7]["lhs"] == ['m'])
    assert(rules[7]["rhs"] == ['c'])
    assert(rules[7]["confidence"] == 1.0)
    assert(rules[8]["lhs"] == ['s'])
    assert(rules[8]["rhs"] == ['c'])
    assert(rules[8]["confidence"] == 1.0)


    # Lab Task 5 (For Extra Practice) ------------------------------------

    # interview dataset
    header = ["level", "lang", "tweets", "phd", "interviewed_well"]
    table = [
        ["Senior", "Java", "no", "no", "False"],
        ["Senior", "Java", "no", "yes", "False"],
        ["Mid", "Python", "no", "no", "True"],
        ["Junior", "Python", "no", "no", "True"],
        ["Junior", "R", "yes", "no", "True"],
        ["Junior", "R", "yes", "yes", "False"],
        ["Mid", "R", "yes", "yes", "True"],
        ["Senior", "Python", "no", "no", "False"],
        ["Senior", "R", "yes", "no", "True"],
        ["Junior", "Python", "yes", "no", "True"],
        ["Senior", "Python", "yes", "yes", "True"],
        ["Mid", "Python", "no", "yes", "True"],
        ["Mid", "Java", "yes", "no", "True"],
        ["Junior", "Python", "no", "yes", "False"]
    ]

    # fit the data
    rules = marm.fit(table)
    print()
    marm.print_association_rules()

    # Repeat the previous task for the remaining itemsets S in $L_{2} \cup ... \cup L_k$ 
    #  where $k$ is the last non-empty set (e.g. {b, c, m}, {b, c, s}, {c, e, s}, {b, c}, 
    #  {b, m}, {b, s}, {c, e}, {c, m}, {c, s}, {e, s}). The set of all rules left is your 
    #  final apriori set of association rules :)
    assert(rules[0]["lhs"] == ['att2=yes', 'att3=no'])
    assert(rules[0]["rhs"] == ['att4=True'])
    assert(rules[0]["confidence"] == 1.0)
    assert(rules[1]["lhs"] == ['att0=Mid'])
    assert(rules[1]["rhs"] == ['att4=True'])
    assert(rules[1]["confidence"] == 1.0)
    assert(rules[2]["lhs"] == ['att1=R'])
    assert(rules[2]["rhs"] == ['att2=yes'])
    assert(rules[2]["confidence"] == 1.0)
    assert(rules[3]["lhs"] == ['att4=False'])
    assert(rules[3]["rhs"] == ['att2=no'])
    assert(rules[3]["confidence"] == 0.8)
    assert(rules[4]["lhs"] == ['att2=yes'])
    assert(rules[4]["rhs"] == ['att4=True'])
    assert(round((rules[4]["confidence"]), 2) == 0.86)
