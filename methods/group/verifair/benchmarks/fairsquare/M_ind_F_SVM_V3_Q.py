from methods.individual.verifair.benchmarks.fairsquare.helper import *

def sample(flag):
    age = gaussian(38.5816, 186.0614)
    sex = step([(0,1,0.3307), (1,2,0.6693)])
    capital_gain = gaussian(1077.6488, 54542539.1784)
    sensitiveAttribute(sex < 1, flag)
    qualified(age > 18)

    N_age = (age - 17.0) / 62.0
    N_capital_gain = (capital_gain - 0.0) / 22040.0
    t = -0.0008 * N_age + -5.7337 * N_capital_gain + 1.0003
    if sex > 1:
        t = t + 0.0001
    return int(t < 0)
    fairnessTarget(t < 0)

