from methods.individual.verifair.benchmarks.fairsquare.helper import *

def sample(flag):
    age = gaussian(38.5816, 186.0614)
    education_num = gaussian(10.0806, 6.6188)
    sex = step([(0,1,0.3307), (1,2,0.6693)])
    capital_gain = gaussian(1077.6488, 54542539.1784)
    capital_loss = gaussian(87.3038, 162376.9378)
    hours_per_week = gaussian(40.4374, 152.4589)
    sensitiveAttribute(sex < 1, flag)
    qualified(age > 18)

    N_age = (age - 17.0) / 62.0
    N_education_num = (education_num - 3.0) / 13.0
    N_capital_gain = (capital_gain - 0.0) / 22040.0
    N_capital_loss = (capital_loss - 0.0) / 1258.0
    N_hours_per_week = (hours_per_week - 4.0) / 73.0
    t = -0.0106  *  N_age + -0.0194 * N_education_num + -5.7412 * N_capital_gain + 0.0086 * N_capital_loss + -0.0123 * N_hours_per_week + 1.0257
    if sex > 1:
        t = t + -0.0043
    return int(t < 0)
    fairnessTarget(t < 0)

