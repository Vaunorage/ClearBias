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
    N_capital_gain = ((capital_gain - 0.0) / 22040.0 - 0.5) * 10 + 0.5
    h1 = -0.2277 * N_age +  0.6434 * N_education_num +  2.3643 * N_capital_gain +  3.7146
    if h1 < 0:
        h1 = 0
    h2 = -0.0236 * N_age + -3.3556 * N_education_num + -1.8183 * N_capital_gain + -1.7810
    if h2 < 0:
        h2 = 0
    o1 =  0.4865 * h1 +  1.0685 * h2 + -1.8079
    if o1 < 0:
        o1 = 0
    o2 =  1.7044 * h1 + -1.3880 * h2 +  0.6830
    if o2 < 0:
        o2 = 0
    return int(o1 < o2)
    fairnessTarget(o1 < o2)

