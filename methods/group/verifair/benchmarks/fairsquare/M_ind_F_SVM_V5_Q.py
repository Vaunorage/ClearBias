from methods.individual.verifair.benchmarks.fairsquare.helper import *

def sample(flag):
    age = gaussian(38.5816, 186.0614)
    sex = step([(0,1,0.3307), (1,2,0.6693)])
    capital_gain = gaussian(1077.6488, 54542539.1784)
    capital_loss = gaussian(87.3038, 162376.9378)
    hours_per_week = gaussian(40.4374, 152.4589)
    sensitiveAttribute(sex < 1, flag)
    qualified(age > 18)

    N_age = (age - 17.0) / 62.0
    N_capital_gain = (capital_gain - 0.0) / 22040.0
    N_capital_loss = (capital_loss - 0.0) / 1258.0
    N_hours_per_week = (hours_per_week - 4.0) / 73.0
    t = 0.0001 * N_age + -5.7368 * N_capital_gain + 0.0002 * N_capital_loss + 0.0003 * N_hours_per_week + 1     
    if sex > 1:
        t = t + 0.0005
    return int(t < 0)
    fairnessTarget(t < 0)

