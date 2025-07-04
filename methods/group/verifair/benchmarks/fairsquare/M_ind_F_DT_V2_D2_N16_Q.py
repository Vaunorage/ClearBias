from methods.individual.verifair.benchmarks.fairsquare.helper import *

def sample(flag):
    age = gaussian(38.5816, 186.0614)
    relationship = step([(0,1,0.0481), (1,2,0.1557), (2,3,0.4051), (3,4,0.2551), (4,5,0.0301), (5,6,0.1059)])
    sex = step([(0,1,0.3307), (1,2,0.6693)])
    capital_gain = gaussian(1077.6488, 54542539.1784)
    sensitiveAttribute(sex < 1, flag)
    qualified(age > 18)

    if relationship < 1:
        if capital_gain < 5095.5:
            t = 1
        else:
            t = 0
    elif relationship < 2:
        if capital_gain < 4718.5:
            t = 1
        else:
            t = 0
    elif relationship < 3:
        if capital_gain < 5095.5:
            t = 1
        else:
            t = 0
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

