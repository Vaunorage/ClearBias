from methods.individual.verifair.benchmarks.fairsquare.helper import *

def sample(flag):
    age = gaussian(38.5816, 186.0614)
    education = step([(0,1,0.1644), (1,2,0.2239), (2,3,0.0361), (3,4,0.3225), (4,5,0.0177), (5,6,0.0328), (6,7,0.0424), (7,8,0.0158), (8,9,0.0198), (9,10,0.0133), (10,11,0.0530), (11,12,0.0051), (12,13,0.0287), (13,14,0.0127), (14,15,0.0102), (15,16,0.0016)])
    relationship = step([(0,1,0.0481), (1,2,0.1557), (2,3,0.4051), (3,4,0.2551), (4,5,0.0301), (5,6,0.1059)])
    sex = step([(0,1,0.3307), (1,2,0.6693)])
    capital_gain = gaussian(1077.6488, 54542539.1784)
    sensitiveAttribute(sex < 1, flag)
    qualified(age > 18)

    if relationship < 1:
        if education < 1:
            t = 0
        elif education < 2:
            t = 1
        elif education < 3:
            t = 1
        elif education < 4:
            t = 1
        elif education < 5:
            t = 0
        elif education < 6:
            t = 0
        elif education < 7:
            t = 0
        elif education < 8:
            t = 1
        elif education < 9:
            t = 1
        elif education < 10:
            t = 1
        elif education < 11:
            t = 0
        elif education < 12:
            t = 1
        elif education < 13:
            t = 1
        elif education < 14:
            t = 0
        elif education < 15:
            t = 1
        else:
            t = 1
    elif relationship < 2:
        if capital_gain < 4718.5:
            t = 1
        else:
            t = 0
    elif relationship < 3:
        if education < 1:
            t = 0
        elif education < 2:
            t = 1
        elif education < 3:
            t = 1
        elif education < 4:
            t = 1
        elif education < 5:
            t = 0
        elif education < 6:
            t = 1
        elif education < 7:
            t = 1
        elif education < 8:
            t = 1
        elif education < 9:
            t = 1
        elif education < 10:
            t = 1
        elif education < 11:
            t = 0
        elif education < 12:
            t = 1
        elif education < 13:
            t = 1
        elif education < 14:
            t = 0
        elif education < 15:
            t = 1
        else:
            t = 1
    elif relationship < 4:
        if capital_gain < 8296:
            t = 1
        else:
            t = 0
    elif relationship < 5:
        t = 1
    else:
        if capital_gain < 4668.5:
            t = 1
        else:
            t = 0
    return int(t < 0.5)
    fairnessTarget(t < 0.5)

