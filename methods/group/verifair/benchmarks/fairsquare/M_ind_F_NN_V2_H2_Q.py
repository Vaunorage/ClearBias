from methods.individual.verifair.benchmarks.fairsquare.helper import *

def sample(flag):
    age = gaussian(38.5816, 186.0614)
    education_num = gaussian(10.0806, 6.6188)
    sex = step([(0,1,0.3307), (1,2,0.6693)])
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

