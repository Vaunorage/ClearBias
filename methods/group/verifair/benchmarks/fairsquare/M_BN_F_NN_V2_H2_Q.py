from methods.individual.verifair.benchmarks.fairsquare.helper import *

def sample(flag):
    sex = step([(0,1,0.3307), (1,2,0.6693)])
    if sex < 1:
        capital_gain = gaussian(568.4105, 24248365.5428)
        if capital_gain < 7298.0000:
            age = gaussian(38.4208, 184.9151)
            education_num = gaussian(10.0827, 6.5096)
        else:
            age = gaussian(38.8125, 193.4918)
            education_num = gaussian(10.1041, 6.1522)
    else:
        capital_gain = gaussian(1329.3700, 69327473.1006)
        if capital_gain < 5178.0000:
            age = gaussian(38.6361, 187.2435)
            education_num = gaussian(10.0817, 6.4841)
        else:
            age = gaussian(38.2668, 187.2747)
            education_num = gaussian(10.0974, 7.1793)
    sensitiveAttribute(sex < 1, flag)
    qualified(age > 18)

    N_age = ((age - 17.0) / 73.0  - 0.5) * 10 + 0.5
    N_education_num = ((education_num - 3.0) / 13.0  - 0.5) * 10 + 0.5
    h1 =  0.0326 * N_age + 0.9998 * N_education_num + -0.8278
    if h1 < 0:
        h1 = 0
    h2 = -0.4210 * N_age + -0.6857 * N_education_num +  0.8345
    if h2 < 0:
        h2 = 0
    o1 =  0.6004 * h1 +  1.1573 * h2 +  0.8849
    if o1 < 0:
        o1 = 0
    o2 =  1.5905 * h1 +  1.0555 * h2 +  0.9152
    if o2 < 0:
        o2 = 0
    return int(o1 < o2)
    fairnessTarget(o1 < o2)

