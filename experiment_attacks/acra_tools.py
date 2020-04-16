import pandas as pd
import numpy as np
from scipy.stats import beta
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from joblib import Parallel, delayed
import multiprocessing


###############################################################################################################################
###############################################################################################################################
############## MAIN ACRA FUNCTIONS ################################################################### ###########################################################################################
###############################################################################################################################



################################################################################################################################
################################################################################################################################
############## NB RELATED FUNCTIONS  ###########################################################################################
################################################################################################################################
################################################################################################################################

"""
This function trains a raw Naive Bayes in a given training set. In particular, it calculates all relevant parameters such as likelihoods a apriori distributions. Its inputs are:

   * `X_train`: An array where each row is a given email, in the bag-of-words representation (1 if word present, 0 else).

   * `y_train`: An array containing the labels of each email of `X_train` (1 if the email is spam, 0 if ham).

This function returns an `sklearn.naive_bayes.BernoulliNB` object with all relevant information.
"""

def trainRawNB(X_train, y_train):
    from sklearn.naive_bayes import BernoulliNB
    clf = BernoulliNB(alpha=1.0e-10)
    clf.fit(X_train, y_train)
    return(clf)

"""
This function returns the priors $p_C(y = 1)$ and $p_C(y = 0)$. The input is `obj`, an `sklearn.naive_bayes.BernoulliNB` object. The resutl is an array whose first element is $p_C(y = 0)$ and the second $p_C(y = 1)$.
"""

def priors(obj):
    return(np.exp(obj.class_log_prior_ ))

"""
For a given instance (email) $X$ and given classifier `obj` (`sklearn.naive_bayes.BernoulliNB` object) , this function returns $p_C(X|y)p_C(y)$ for $y \in \lbrace 0, 1 \rbrace$.
In particular, it returns an array whose first element is $p_C(X|0)p_C(0)$ and the second $p_C(X|1)p_C(1)$.
"""

def xposterior(X, obj):
    from scipy.special import comb, logsumexp
    return(np.exp(obj._joint_log_likelihood(X)))


"""
For given emails, given utilities, and given `sklearn.naive_bayes.BernoulliNB` classifier, this function returns the label of the emails calculated using utility sensitive NB algorithm.

The inputs are
* `Xp`: array containing the instance to predict on.
* `obj`: the classifier.
* `ut`: an array containing the utilities, `ut[i,j]` $= u_C(y_C = i, y = j)$

The output is an array containing the predicted label of each email, 1 for spam and 0 for ham.

"""


def nbusXlabel(Xp, obj, ut):
    aux = np.dot(ut, xposterior(Xp,obj).transpose())
    return(np.argmax(aux, axis=0))


################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################



################################################################################################################################
################################################################################################################################
############## ACRA BASIC FUNCTIONS  ###########################################################################################
################################################################################################################################
################################################################################################################################


"""
This function computes the set of possible originating instances of a given one, under n-GWI attack strategy. It returns an array containing all these instances.
"""

def getXp(X, n):

    def subs1(X):
        X = np.reshape(X, (1,-1))
        aux = np.ones( ( X.shape[1] , X.shape[1] ) )
        np.fill_diagonal(aux, 0)
        return( np.logical_and(X, aux).astype(int) )


    X = np.reshape(X, (1,-1))
    z = np.apply_along_axis(subs1, 1, X)
    z = z.reshape(z.shape[0]*z.shape[1], z.shape[2])
    for i in range(1,n):
        if i >= np.sum(X == 1):
            break
        z = np.apply_along_axis(subs1, 1, z)
        z = z.reshape(z.shape[0]*z.shape[1], z.shape[2])

    return(np.unique(np.insert(z, 0, X, 0), axis = 0))




"""
This function generates random attacker utilities for a set of attacks, sampling from the gamma distribution. That is, it generate samples from $U_A(y_C,y,a)$.
The inputs are

* `yc`: $y_C$ the classification result.
* `y`: $y$ the true label.
* `a`: array containing number of words added in each attack.
"""

def randut(yc,y,a):

    d = len(a)
    # if y label is malicious and yc label is malicious
    if( (y == 1) and (yc == 1) ):
        Y = - np.random.gamma(shape = np.repeat(2500.0, d), scale = np.repeat(1.0/500.0, d))

    else:
        # if y label is malicious and yc label innocent
        if( (y == 1) and (yc == 0) ):
            Y = np.random.gamma(shape = np.repeat(2500.0, d), scale = np.repeat(1.0/500.0, d))
        # if y label is innocent and yc label is malicious OR y label is innocent and yc label is innocent
        else:
            Y = np.repeat(yc*y, d)


    # Generate random cost of implementing attack
    B = a*np.random.uniform(high = 0.6, low = 0.4, size = 1)

    # Risk proneness
    rho = np.random.uniform(high = 0.6, low = 0.4, size = 1)

    return (np.exp( rho * (Y - B) ))


"""
### Get random probabilities $P_{a(x)}^A$ for given set of instances
First we have to define some auxiliar functions, useful later.
"""

"""
For a given set of instances this function computes the mean of the beta disttribution to be used later.
The inputs are:
* `X`: a 2D-array containing the set of instances.
* `obj`: the classifier (`sklearn.naive_bayes.BernoulliNB` object).

It returns an array containing $r_a$ for each email.
"""


def getRa2(X, obj, n):

    def aux(Z, obj, n):
        q = np.sum( np.apply_along_axis(lambda x: xposterior(x.reshape(1, -1), obj)[0,1],\
                                        1, getXp(Z,1) ) )
        return ( q / (q + xposterior(Z, obj)[0, 0]) )

    ra = np.apply_along_axis( lambda x : aux(x.reshape(1 , -1), obj, n), 1, X)
    ra[ra==1.0] -= 0.0001
    return(ra)


"""
For a given set of instances this function computes the mean of the beta disttribution to be used later
(Alternative heuristic).
The inputs are:
* `X`: a 2D-array containing the set of instances.
* `obj`: the classifier (`sklearn.naive_bayes.BernoulliNB` object).
* `n`: an integer indicating the number of word changes of the attacks.

It returns an array containing $r_a$ for each email.
"""


def getRa(X, obj):
    ra = obj.predict_proba(X)[:,1]
    ra[ra==1.0] -= 0.0001
    return(ra)



"""
This function return the shape parameters of the beta distribution, for given set of means `ra` and given variance $var\cdot ra \min \big(\frac{ra(1-ra)}{1+ra},\frac{(1-ra)^2}{2-ra}\big)$.
"""

def deltas(ra, var):

    deltas = np.zeros((len(ra),2))

    for i in range(len(ra)):
        s2 = var *ra[i]* min(ra[i] * (1.0 - ra[i]) / (1.0 + ra[i]) , \
                             (1.0 - ra[i])**2 / (2.0 - ra[i]))     ## proportion of maximum
                                                                #variance of convex beta
        deltas[i][0] = ( ( 1.0 - ra[i] ) / s2 - 1.0 / ra[i]) * ra[i]**2
        deltas[i][1] = deltas[i][0] * ( 1.0/ra[i] - 1.0 )

    return(deltas)

"""
For a given email $X$, this function computes $\mathcal{A}(X)$ under some attack strategy (n word insertion, in this particular case). For each $a \in \mathcal{A}(X)$, it computes $a(X)$, and returns an array containing all $a(X)$.
"""

def getxax(X, n):

    def add1(X):
        X = np.reshape(X, (1,-1))
        return( np.logical_or(X, np.identity(X.shape[1])).astype(int) )


    X = np.reshape(X, (1,-1))
    z = np.apply_along_axis(add1, 1, X)
    z = z.reshape(z.shape[0]*z.shape[1], z.shape[2])
    for i in range(1,n):
        if i >= np.sum(X == 0):
            break
        z = np.apply_along_axis(add1, 1, z)
        z = z.reshape(z.shape[0]*z.shape[1], z.shape[2])

    return(np.unique(np.insert(z, 0, X, 0), axis = 0))
"""
For a given array `delta`, where `delta[0]` correspond to $\delta_1$ and `delta[1]` to $\delta_2$, this function generates one sample from the beta distribution for each pair of deltas.
"""

def randprob(deltas):
    return( np.random.beta(deltas[:,0], deltas[:,1]) )

"""
This is the main function in ACRA algorithm. For a pair of instances $x$ and $x'$, this function returns the probability that the attacker, given that he has instance $x$, will execute the attack that transfors $x$ into $x'$.
"""

def pxaxp(x, xp, obj, var, n, K = 1000):

    # First we compute the set of all a(x) for the given x, and store them in the array aX.
    aX = getxax(x, n)

    # We store in ix, the index of the element of aX coinciding with xp, this is the index of the attack
    # conecting x with xp.
    ix = np.where(np.all(aX == xp, axis=1))[0]

    # We compute de deltas of the instances in aX and store them in d
    #d = deltas(getRa2(aX, obj, n), var)
    d = deltas(getRa(aX, obj), var)

    # We compute the distances between tha attacked instances (those in aX) and the original instance x
    distances = np.sum(aX - x, axis=1)

    # We start the simulation
    distribution = np.zeros( len(distances) ) #here we will store the number of times each attack is maximum

    for i in range(K):
        PA = randprob(d)
        psi = PA * randut(1,1,distances) + (1.0 - PA)* randut(0,1,distances)
        distribution[np.argmax(psi)] += 1



    return( sum(distribution[ix])/K )


################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################



################################################################################################################################
################################################################################################################################
############## ACRA IMPLEMENTATION FUNCTIONS  ##################################################################################
################################################################################################################################
################################################################################################################################

"""
For a given email, given `sklearn.naive_bayes.BernoulliNB` classifier and given `var` this function returns the ACRA posteriors $p_C(X'|y)p_C(y)$ for $y \in \lbrace 0, 1 \rbrace$.
In particular, it returns an array whose first element is $p_C(X'|0)p_C(0)$ and the second $p_C(X'|1)p_C(1)$.

The inputs are
* `Xp`: array containing the instance to predict on.
* `obj`: the classifier.
* `var`: variance = $var\cdot ra \min \big(\frac{ra(1-ra)}{1+ra},\frac{(1-ra)^2}{2-ra}\big)$

"""

def ACRAposterior(Xp, obj, var, n):
    aux = getXp(Xp, n)
    sum = 0
    for i in range(aux.shape[0]):
        sum += pxaxp(aux[[i],:],Xp,obj,var,n)*xposterior(aux[[i],:], obj)[0,1]
    return(np.array( [ xposterior(Xp, obj)[0,0] , sum ] ))



"""
The same for multiple instances, parallelizing the code
"""

def posteriorInput(i,Xp,obj,var,n):
        return ACRAposterior(Xp[[i],:],obj,var,n)

def ACRAparPosterior(Xp, obj, var, n):
    inputs = range(Xp.shape[0])
    num_cores = multiprocessing.cpu_count()
    result = Parallel(n_jobs=num_cores)(delayed(posteriorInput)(i,Xp,obj,var,n) for i in inputs)
    return(np.array(result))

def ACRAlabel(posterior, ut):

    aux = np.dot(ut, posterior.transpose())

    return(np.argmax(aux, axis = 0))

"""
For a given email, this function returns the ACRA label.
The inputs:
* `Xp`: array containing the instance to predict on.
* `obj`: the classifier.
* `var`: variance = $var\cdot ra \min \big(\frac{ra(1-ra)}{1+ra},\frac{(1-ra)^2}{2-ra}\big)$
* `ut`: an array containing the utilities, `ut[i,j]` $= u_C(y_C = i, y = j)$

The output is 0 for ham, 1 for spam
"""

def ACRA(Xp, obj, ut, var, n):
    aux = np.dot(ut, ACRAposterior(Xp,obj, var, n).transpose())
    return(np.argmax(aux, axis=0))


def seqACRA(Xp,obj, ut, var, n):
    t = (ut[0,0] - ut[1,0]) * xposterior(Xp, obj)[0,0] / (ut[1,1] - ut[0,1])
    aux = getXp(Xp, n)

    sum = 0
    for i in range(aux.shape[0]):
        sum += pxaxp(aux[[i],:],Xp,obj,var,n)*xposterior(aux[[i],:], obj)[0,1]
        if sum > t:
            return(1)

    return(0)

################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################



################################################################################################################################
################################################################################################################################
############## COMPUTATIONAL ENHANCENMENTS  ####################################################################################
################################################################################################################################
################################################################################################################################


"""
This function computes a set of size m of possible originating instances of a given one, under n-GWI attack strategy. It chooses those m according to the probability p(x|+). It returns an array containing all these instances.
"""


def getRedXp(XX, obj, n, m):
    p = xposterior(XX, obj)[:,1]
    K = np.sum(p)
    p = p/K
    m = int(np.ceil(XX.shape[0]*m))
    return( XX[np.random.choice(XX.shape[0], m, replace=False, p=p.tolist())], K )


def seqMCACRA(Xp, obj, ut, var, n, m):

    aux = getXp(Xp, n)
    t = (ut[0,0] - ut[1,0]) * xposterior(Xp, obj)[0,0] /  (ut[1,1] - ut[0,1]) #* np.exp(obj.class_log_prior_)[1]
    aux, K = getRedXp(aux, obj, n, m)

    I = K * pxaxp(aux[[0],:],Xp,obj,var,n)
    #V = 0

    I_old = I
    #V_old = 0

    if I > t: # - 0.0 * np.sqrt(V) > t:
            return(1)

    for i in range(1, aux.shape[0]):

        p = pxaxp(aux[[i],:],Xp,obj,var,n)

        I = 1.0/(i+1) * ( I_old*i +  K*p)
        #V = 1.0/(i) * (  (i-1)*V_old + (K*p)**2 + i*I_old**2 - (i+1)*I**2   )
        I_old = I
        #V_old = V

        if I > t: # - 0.0 * np.sqrt(V) > t:
            return(1)

    return(0)


def sumInput(i,X,Xp,obj,var,n):
        return pxaxp(X[[i],:],Xp,obj,var,n)

def parSum(X, Xp, obj, var, n):
    inputs = range(X.shape[0])
    num_cores = multiprocessing.cpu_count()
    result = Parallel(n_jobs=num_cores)(delayed(sumInput)(i,X,Xp,obj,var,n) for i in inputs)
    return(np.sum( np.array(result) ) )

def MCParACRA(Xp, obj, ut, var, n, m):

    t = (ut[0,0] - ut[1,0]) * xposterior(Xp, obj)[0,0] /  (ut[1,1] - ut[0,1]) #* np.exp(obj.class_log_prior_)[1]
    aux = getXp(Xp, n)
    aux, K = getRedXp(aux, obj, n, m)

    I_par = K * parSum(aux, Xp, obj, var, n) / aux.shape[0]

    if I_par > t:
        return(1)
    else:
        return(0)

################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################



################################################################################################################################
################################################################################################################################
############## PRUEBAS DE FUNCIONAMIENTO  ######################################################################################
################################################################################################################################
################################################################################################################################


def idiotACRA(Xp, obj, ut, var, n, m = 1):

    aux = getXp(Xp, n)

    t = (ut[0,0] - ut[1,0]) * xposterior(Xp, obj)[0,0] /  (ut[1,1] - ut[0,1]) #* np.exp(obj.class_log_prior_)[1]

    aux, K = getRedXp(aux, obj, n, m)

    if K * pxaxp(aux[[0],:],Xp,obj,var,n) > t:
        return(1)
    else:
        return(0)


def simpleACRA(Xp, obj, ut, var, n, m = 1):

    aux = getXp(Xp, n)

    t = (ut[0,0] - ut[1,0]) * xposterior(Xp, obj)[0,0] /  (ut[1,1] - ut[0,1]) #* np.exp(obj.class_log_prior_)[1]

    aux, K = getRedXp(aux, obj, n, m)

    if xposterior(aux, obj)[0,1] > t:
        return(1)
    else:
        return(0)


################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################

################################################################################################################################
################################################################################################################################
############## ATTACKER DESIGN  ################################################################################################
################################################################################################################################
################################################################################################################################

"""
## Attacker simulation

In order to test ACRA algorithm, we need to get an attacked test set. As long as there are no benchmarks for that purpose, we will generate it artificially by simulating the attacker's behaviour.

At a first stage, we can simulate the attacker using the same assumptions we use to solve the classifier problem, but removing the uncertainty that is not present from the attackers point of view. Therefore, the attacker will not change ham emails. For spam email he will solve

$$
argmax_a \big[u_A(+,+,a)-u_A(-,+,a)\big]p_{a(x)}^A + u_A(-,+,a)
$$

Now the utilities are not random anymore, specifically we shall use the same as in the classifier problem, but collapsing every probability distribution to its mean value.

$p_{a(x)}^A$ will be the probability given by the naive Bayes classifier.



"""
def adversarialUt(yc,y,a):

    d = len(a)
    # if y label is malicious and yc label is malicious
    if( (y == 1) and (yc == 1) ):
        Y = np.repeat(-5.0, d)

    else:
        # if y label is malicious and yc label innocent
        if( (y == 1) and (yc == 0) ):
            Y = np.repeat(5.0, d)
        # if y label is innocent and yc label is malicious OR y label is innocent and yc label is innocent
        else:
            Y = np.repeat(0.0, d)


    # Generate random cost of implementing attack
    B = a*np.repeat(0.5, d)

    # Risk proneness
    rho = np.repeat(0.5, d)

    return (np.exp( rho * (Y - B) ))

"""
Super clever attacker
"""

def sc_attackit(X, y, obj, n):
    if y == 0:
        return(X)
    else:
        possibleAttacks = getxax(X, n)
        pr = getRa(possibleAttacks, obj)
        distances = np.sum(possibleAttacks - X, axis=1)
        psi = pr * adversarialUt(1,1,distances) + (1.0 - pr)* adversarialUt(0,1,distances)
        return(possibleAttacks[np.argmax(psi),:])


def sc_attack(X, y, obj, n):
    att = np.zeros(X.shape, dtype=int)
    for i in range(len(y)):
        att[i,:] = sc_attackit(X[[i],:], y[i], obj, n)
    return(att)



"""
Clever attacker
"""

def attackit(X, y, obj, n):
    if y == 0:
        return(X)
    else:
        possibleAttacks = getxax(X, n)
        pr = getRa2(possibleAttacks, obj,n)
        distances = np.sum(possibleAttacks - X, axis=1)
        psi = pr * adversarialUt(1,1,distances) + (1.0 - pr)* adversarialUt(0,1,distances)
        return(possibleAttacks[np.argmax(psi),:])


def attack(X, y, obj, n):
    att = np.zeros(X.shape, dtype=int)
    for i in range(len(y)):
        att[i,:] = attackit(X[[i],:], y[i], obj, n)
    return(att)

"""

Ignorant attacker. This attacker adds n words at random

"""

def ign_attackit(X, y, n):
    if y == 0:
        return(X)
    else:
        Z = X.copy()
        for i in range(n):
            if np.where(Z[0] == 0)[0].size != 0:
               Z[0][np.random.choice(np.where(Z[0] == 0)[0], size=1, replace = False)] = 1
            else:
                break
        return(Z)


def ign_attack(X, y, n):
    att = np.zeros(X.shape, dtype=int)
    for i in range(len(y)):
        att[i,:] = ign_attackit(X[[i],:], y[i], n)
    return(att)


################################################################################################################################
################################################################################################################################
################################################################################################################################
################################################################################################################################


################################################################################################################################
################################################################################################################################
############## WRITING FUNCTIONS  ################################################################################################
################################################################################################################################
################################################################################################################################
"""
Write dataframe to csv
"""

def write_to_csv(name, NBC_post, ACRA_post, NB_post, y_test):

    cols = ["None"]*7
    cols[ 0 ] = "NBCpost0"
    cols[ 1 ] = "NBCpost1"
    cols[ 2 ] = "ACRApost0"
    cols[ 3 ] = "ACRApost1"
    cols[ 4 ] = "NBpost0"
    cols[ 5 ] = "NBpost1"
    cols[ 6 ] = "spam"


    bigResult = pd.DataFrame(columns=cols)

    for i in range(y_test.shape[0]):
        bigResult.loc[i] = list(np.concatenate( (NBC_post[i,:], ACRA_post[i,:], \
                                                 NB_post[i,:],y_test[[i]]) , axis = 0))


    bigResult.to_csv(name)
