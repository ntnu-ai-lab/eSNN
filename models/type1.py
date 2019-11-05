import numpy as np

def sim_def_lin(model, test_instances, test_targets, training_instances,
            training_targets, batch_size, anynominal, colmap):
    tot = np.concatenate((test_instances,training_instances),axis=0)
    tot_maximums = np.max(tot,axis=0)
    tot_minmums = np.min(tot,axis=0)
    res = list()
    nominallist = list()
    if anynominal:
        for key,value in colmap.items():
            #if value["class"]:
            #    continue
            if value["type"] is "nominal":
                nominallist.append(True)
            else:
                nominallist.append(False)
    for test_instance,test_target in zip(test_instances,test_targets):
        max_sim=0
        most_similar_case_target = None
        for training_instance,training_target in zip(training_instances,training_targets):
            sim = global_sim_def(test_instance,training_instance,tot_maximums,tot_minmums,anynominal,nominallist)
            if sim > max_sim:
                max_sim = sim
                most_similar_case_target = training_target
        if (most_similar_case_target is not None) and (np.rint(test_target) == np.rint(most_similar_case_target)).all():
            res.append(1)
        else:
            res.append(0)
    return res

def global_sim_def(v1,v2,maximums,minimums,anynominal, nominallist):
    res = 0
    if not anynominal:
        for i in range(0,v1.shape[0]):
            res = res + local_sim_n_def(v1[i],v2[i],maximums[i],minimums[i])
    else:
        for i in range(0, v1.shape[0]):
            if nominallist[i] is True:
                res = res + local_sim_s_def(v1[i], v2[i])
            else:
                res = res + local_sim_n_def(v1[i], v2[i], maximums[i], minimums[i])
    ret = res/len(v1)
    return ret

def local_sim_n_def(v1,v2,d_max,d_min):
    return 1-(abs(v1-v2)/(d_max-d_min))

def local_sim_s_def(v1,v2):
    if v1 == v2:
        return 1
    else:
        return 0

def getValue(arg0,param,max_val,min_val):
    val1 = min_val
    val2 = abs(arg0)+min_val
    if arg0 > 0:
        return sim_nonlin(val1,val2,param,max_val,min_val)
    else:
        return sim_nonlin(val2,val1,param,max_val,min_val)

def sim_nonlin(q,c,paramL,max_val,min_val):
    d = q-c
    maxrange = max_val-min_val # max-min
    if d < 0:
        return f(d,paramL,maxrange)
    else:
        return f(d,paramL,-maxrange)

def f(value,exponent,diff):
    return myfilter(np.power(value/diff + 1.0, exponent))

def mySimFct(c1, c2, param, diff):
    return

def myrounder(num):
    return round(num*100,0)/100.0

def myfilter(num):
    if num < 0.0 or num > 1:
        return -1
    else:
        return num

def global_sim_nonlin(v1,v2,maximums,minimums,anynominal, nominallist, params):
    res = 0
    if not anynominal:
        for i in range(0,v1.shape[0]):
            res = res + sim_nonlin(v1[i],v2[i],params[i],maximums[i],minimums[i])
    else:
        for i in range(0, v1.shape[0]):
            if nominallist[i] is True:
                res = res + local_sim_s_def(v1[i], v2[i])
            else:
                res = res + sim_nonlin(v1[i],v2[i],params[i],maximums[i],minimums[i])
    ret = res/len(v1)
    return ret

def sim_def_nonlin(model, test_instances, test_targets, training_instances,
            training_targets, batch_size, anynominal, colmap):
    tot = np.concatenate((test_instances,training_instances),axis=0)
    tot_maximums = np.max(tot,axis=0)
    tot_minmums = np.min(tot,axis=0)
    params = np.zeros((tot.shape[1],1))
    for i in range(0,tot.shape[1]):
        q1 = np.percentile(tot[:,i],25)
        q3 = np.percentile(tot[:,i],75)
        params[i] = abs(q1-q3)
    res = list()
    nominallist = list()
    if anynominal:
        for key,value in colmap.items():
            #if value["class"]:
            #    continue
            if value["type"] is "nominal":
                nominallist.append(True)
            else:
                nominallist.append(False)
    for test_instance,test_target in zip(test_instances,test_targets):
        max_sim=0
        most_similar_case_target = None
        for training_instance,training_target in zip(training_instances,training_targets):
            sim = global_sim_nonlin(test_instance,training_instance,tot_maximums,tot_minmums,anynominal,nominallist,params)
            if sim > max_sim:
                max_sim = sim
                most_similar_case_target = training_target
        if (most_similar_case_target is not None) and (np.rint(test_target) == np.rint(most_similar_case_target)).all():
            res.append(1)
        else:
            res.append(0)
    return res
