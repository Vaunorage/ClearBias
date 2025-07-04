from methods.individual.verifair.benchmarks.fairsquare.helper import *

def sample(flag):
    sex = step([(0,1,0.3307), (1,2,0.6693)])
    if sex < 1:
        capital_gain = gaussian(568.4105, 24248365.5428)
        if capital_gain < 7298.0000:
            age = gaussian(38.4208, 184.9151)
            education = step([(0,1,0.1638), (1,2,0.2308), (2,3,0.0354), (3,4,0.3230), (4,5,0.0173), (5,6,0.0321), (6,7,0.0412), (7,8,0.0156), (8,9,0.0200), (9,10,0.0112), (10,11,0.0528), (11,12,0.0050), (12,13,0.0290), (13,14,0.0119), (14,15,0.0092), (15,16,0.0017)])
            relationship = step([(0,1,0.0491), (1,2,0.1556), (2,3,0.4012), (3,4,0.2589), (4,5,0.0294), (5,6,0.1058)])
        else:
            age = gaussian(38.8125, 193.4918)
            education = step([(0,1,0.1916), (1,2,0.2000), (2,3,0.0500), (3,4,0.3542), (4,5,0.0208), (5,6,0.0125), (6,7,0.0375), (7,8,0.0125), (8,9,0.0292), (9,10,0.0042), (10,11,0.0541), (11,12,0.0000), (12,13,0.0250), (13,14,0.0042), (14,15,0.0042), (15,16,0.0000)])
            relationship = step([(0,1,0.0416), (1,2,0.1667), (2,3,0.4583), (3,4,0.2292), (4,5,0.0166), (5,6,0.0876)])
    else:
        capital_gain = gaussian(1329.3700, 69327473.1006)
        if capital_gain < 5178.0000:
            age = gaussian(38.6361, 187.2435)
            education = step([(0,1,0.1670), (1,2,0.2239), (2,3,0.0358), (3,4,0.3267), (4,5,0.0159), (5,6,0.0320), (6,7,0.0426), (7,8,0.0155), (8,9,0.0198), (9,10,0.0121), (10,11,0.0518), (11,12,0.0047), (12,13,0.0287), (13,14,0.0125), (14,15,0.0096), (15,16,0.0014)])
            relationship = step([(0,1,0.0497), (1,2,0.1545), (2,3,0.4021), (3,4,0.2590), (4,5,0.0294), (5,6,0.1053)])
        else:
            age = gaussian(38.2668, 187.2747)
            education = step([(0,1,0.1569), (1,2,0.2205), (2,3,0.0417), (3,4,0.3071), (4,5,0.0255), (5,6,0.0302), (6,7,0.0409), (7,8,0.0155), (8,9,0.0178), (9,10,0.0147), (10,11,0.0619), (11,12,0.0062), (12,13,0.0317), (13,14,0.0139), (14,15,0.0139), (15,16,0.0016)])
            relationship = step([(0,1,0.0417), (1,2,0.1624), (2,3,0.3976), (3,4,0.2606), (4,5,0.0356), (5,6,0.1021)])
    sensitiveAttribute(sex < 1, flag)
    qualified(age > 18)

    if relationship < 1:
        if education < 1:
            t = 0
        elif education < 2:
            t = 1
        elif education < 3:
            t = 1
        elif education < 4:
            t = 1
        elif education < 5:
            t = 0
        elif education < 6:
            t = 0
        elif education < 7:
            t = 0
        elif education < 8:
            t = 1
        elif education < 9:
            t = 1
        elif education < 10:
            t = 1
        elif education < 11:
            t = 0
        elif education < 12:
            t = 1
        elif education < 13:
            t = 1
        elif education < 14:
            t = 0
        elif education < 15:
            t = 1
        else:
            t = 1
    elif relationship < 2:
        if capital_gain < 4718.5:
            t = 1
        else:
            t = 0
    elif relationship < 3:
        if education < 1:
            t = 0
        elif education < 2:
            t = 1
        elif education < 3:
            t = 1
        elif education < 4:
            t = 1
        elif education < 5:
            t = 0
        elif education < 6:
            t = 1
        elif education < 7:
            t = 1
        elif education < 8:
            t = 1
        elif education < 9:
            t = 1
        elif education < 10:
            t = 1
        elif education < 11:
            t = 0
        elif education < 12:
            t = 1
        elif education < 13:
            t = 1
        elif education < 14:
            t = 0
        elif education < 15:
            t = 1
        else:
            t = 1
    elif relationship < 4:
        if capital_gain < 8296:
            t = 1
        else:
            t = 0
    elif relationship < 5:
        t = 1
    else:
        if capital_gain < 4668.5:
            t = 1
        else:
            t = 0
    return int(t < 0.5)
    fairnessTarget(t < 0.5)

