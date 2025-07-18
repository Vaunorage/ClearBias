from methods.individual.verifair.benchmarks.fairsquare.helper import *

def sample(flag):
    sex = step([(0,1,0.3307), (1,2,0.6693)])
    if sex < 1:
        capital_gain = gaussian(568.4105, 24248365.5428)
        if capital_gain < 7298.0000:
            age = gaussian(38.4208, 184.9151)
            relationship = step([(0,1,0.0491), (1,2,0.1556), (2,3,0.4012), (3,4,0.2589), (4,5,0.0294), (5,6,0.1058)])
        else:
            age = gaussian(38.8125, 193.4918)
            relationship = step([(0,1,0.0416), (1,2,0.1667), (2,3,0.4583), (3,4,0.2292), (4,5,0.0166), (5,6,0.0876)])
    else:
        capital_gain = gaussian(1329.3700, 69327473.1006)
        if capital_gain < 5178.0000:
            age = gaussian(38.6361, 187.2435)
            relationship = step([(0,1,0.0497), (1,2,0.1545), (2,3,0.4021), (3,4,0.2590), (4,5,0.0294), (5,6,0.1053)])
        else:
            age = gaussian(38.2668, 187.2747)
            relationship = step([(0,1,0.0417), (1,2,0.1624), (2,3,0.3976), (3,4,0.2606), (4,5,0.0356), (5,6,0.1021)])
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

