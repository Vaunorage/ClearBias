from methods.individual.verifair.benchmarks.fairsquare.helper import *

def sample(flag):
    sex = step([(0,1,0.3307), (1,2,0.6693)])
    if sex < 1:
        capital_gain = gaussian(568.4105, 24248365.5428)
        if capital_gain < 7298.0000:
            age = gaussian(38.4208, 184.9151)
        else:
            age = gaussian(38.8125, 193.4918)
    else:
        capital_gain = gaussian(1329.3700, 69327473.1006)
        if capital_gain < 5178.0000:
            age = gaussian(38.6361, 187.2435)
        else:
            age = gaussian(38.2668, 187.2747)
    sensitiveAttribute(sex < 1, flag)
    qualified(age > 18)

    N_age = (age - 17.0) / 62.0
    N_capital_gain = (capital_gain - 0.0) / 22040.0
    t = -0.0008 * N_age + -5.7337 * N_capital_gain + 1.0003
    if sex > 1:
        t = t + 0.0001
    return int(t < 0)
    fairnessTarget(t < 0)

