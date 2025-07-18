from methods.individual.verifair.benchmarks.fairsquare.helper import *

def sample(flag):
    age = gaussian(38.5816, 186.0614)
    sex = step([(0,1,0.3307), (1,2,0.6693)])
    capital_gain = gaussian(1077.6488, 54542539.1784)
    capital_loss = gaussian(87.3038, 162376.9378)
    sensitiveAttribute(sex < 1, flag)
    qualified(age > 18)

    N_age = (age - 17.0) / 62.0
    N_capital_gain = (capital_gain - 0.0) / 22040.0
    N_capital_loss = (capital_loss - 0.0) / 1258.0
    t = 0.0006 * N_age + -5.7363 * N_capital_gain + -0.0002 * N_capital_loss + 1.0003
    if sex > 1:
        t = t + -0.0003
    if sex < 1:
        t = t - 0.5
    return int(t < 0)
    fairnessTarget(t < 0)

