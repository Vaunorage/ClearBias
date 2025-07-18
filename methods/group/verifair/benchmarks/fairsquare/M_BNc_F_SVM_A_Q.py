from methods.individual.verifair.benchmarks.fairsquare.helper import *

def sample(flag):
    sex = step([(0,1,0.3307), (1,2,0.6693)])
    if sex < 1:
        capital_gain = gaussian(568.4105, 24248365.5428)
        if capital_gain < 7298.0000:
            age = gaussian(38.4208, 184.9151)
            education_num = gaussian(10.0827, 6.5096)
            capital_loss = gaussian(86.5949, 157731.9553)
        else:
            age = gaussian(38.8125, 193.4918)
            education_num = gaussian(10.1041, 6.1522)
            capital_loss = gaussian(117.8083, 252612.0300)
    else:
        capital_gain = gaussian(1329.3700, 69327473.1006)
        if capital_gain < 5178.0000:
            age = gaussian(38.6361, 187.2435)
            education_num = gaussian(10.0817, 6.4841)
            capital_loss = gaussian(87.0152, 161032.4157)
        else:
            age = gaussian(38.2668, 187.2747)
            education_num = gaussian(10.0974, 7.1793)
            capital_loss = gaussian(101.7672, 189798.1926)
    if (education_num > age):
        age = education_num
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

