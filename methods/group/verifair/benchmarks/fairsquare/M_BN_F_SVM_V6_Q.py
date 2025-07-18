from methods.individual.verifair.benchmarks.fairsquare.helper import *

def sample(flag):
    sex = step([(0,1,0.3307), (1,2,0.6693)])
    if sex < 1:
        capital_gain = gaussian(568.4105, 24248365.5428)
        if capital_gain < 7298.0000:
            age = gaussian(38.4208, 184.9151)
            education_num = gaussian(10.0827, 6.5096)
            capital_loss = gaussian(86.5949, 157731.9553)
            hours_per_week = gaussian(40.4959, 151.4148)
        else:
            age = gaussian(38.8125, 193.4918)
            education_num = gaussian(10.1041, 6.1522)
            capital_loss = gaussian(117.8083, 252612.0300)
            hours_per_week = gaussian(41.6916, 165.3773)
    else:
        capital_gain = gaussian(1329.3700, 69327473.1006)
        if capital_gain < 5178.0000:
            age = gaussian(38.6361, 187.2435)
            education_num = gaussian(10.0817, 6.4841)
            capital_loss = gaussian(87.0152, 161032.4157)
            hours_per_week = gaussian(40.3897, 150.6723)
        else:
            age = gaussian(38.2668, 187.2747)
            education_num = gaussian(10.0974, 7.1793)
            capital_loss = gaussian(101.7672, 189798.1926)
            hours_per_week = gaussian(40.6473, 153.4823)
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

