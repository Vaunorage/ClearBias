from methods.individual.verifair.benchmarks.fairsquare.helper import *

def sample(flag):
    age = gaussian(38.5816, 186.0614)
    relationship = step([(0,1,0.0481), (1,2,0.1557), (2,3,0.4051), (3,4,0.2551), (4,5,0.0301), (5,6,0.1059)])
    sex = step([(0,1,0.3307), (1,2,0.6693)])
    sensitiveAttribute(sex < 1, flag)
    qualified(age > 18)

    if relationship < 1:
       t = 1 
    elif relationship < 2:
        if age < 21.5:
            t = 1
        else:
            if age < 47.5:
                t = 1
            else:
                t = 0
    elif relationship < 3:
        t = 1
    elif relationship < 4:
        if age < 50.5:
            t = 1
        else:
            t = 0
    elif relationship < 5:
        if age < 49.5:
            t = 1
        else:
            t = 0
    else:
        t = 1
    return int(t < 0.5)
    fairnessTarget(t < 0.5)

