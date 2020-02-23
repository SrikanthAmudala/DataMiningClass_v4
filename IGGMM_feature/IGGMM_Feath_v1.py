"""
Infinite Generalized Gaussian Mixture model working
@author: Srikanth Amudala
"""

import copy
import math
import time
from collections import defaultdict

import mpmath
import numpy as np
from scipy import special
from IGGMM_feature.utils import *

from scipy.special import digamma
from scipy.special import gamma
from sklearn.cluster import KMeans

from matplotlib import pyplot


# from IGGMM_feature.clustering import *


# E[ln pik]
def e_ln_pi_k(gama0, Nk):
    gammak = gama0 + Nk
    return digamma(gammak) - digamma(gammak.sum())


def e_mean_n(sk, mk, shape, k):
    """
    E(|mean^shape|)
    """
    # print(sk, mk, shape, k)
    temp = []
    for cluster in range(k):
        p = (1 / math.sqrt(sk[cluster])) ** shape * 2 ** (shape / 2) * gamma((1 + shape) / 2) / math.sqrt(
            math.pi) * mpmath.hyp1f1(-shape / 2, 1 / 2, -1 / 2 * mk[cluster] ** 2 * sk[0])
        temp.append(p)
    return np.asarray(temp).reshape(-1, 1)


def lowerbound_first_dir(rnk, x, shape, mk, s0, e_precision_):
    x_mu = np.abs(x.reshape(-1, 1) - mk)
    p11 = np.power(x_mu, shape)
    p12 = np.log(x_mu)
    p1 = p11 * p12 * (s0 - e_precision_)
    p2 = -(1 / np.power(shape, 2)) * np.log(np.abs(e_precision_)) + 1 / shape - 1 / (2 * gamma(1 / shape)) * digamma(
        1 / shape)
    p3 = e_precision_ * np.power(x_mu, shape) * np.log(x_mu)
    return rnk * (p1 + p2 + p3)


def lowerbound_second_dir(rnk, x, shape, mk, s0, e_precision_):
    x_mu = np.abs(x.reshape(-1, 1) - mk)
    p1 = 2 * np.power(x_mu, shape) * np.log(x_mu) * (s0 - e_precision_)

    p2 = (2 / np.power(shape, 3)) * np.log(np.abs(e_precision_)) - 1 / np.power(shape, 2) + 1 / (
            2 * np.power(gamma(1 / shape), 2)) * np.power(digamma(1 / shape), 2) - 1 / (
                 2 * gamma(1 / shape)) * special.polygamma(1, 1 / shape)

    p3 = e_precision_ * 2 * np.power(x_mu, shape) * np.log(x_mu)
    return rnk * (p1 + p2 + p3)


def initilization(k, x):
    # init
    kmeans = KMeans(n_clusters=k, random_state=0).fit(x.reshape(-1, 1))

    means = kmeans.cluster_centers_
    clustered_img = defaultdict(list)

    labels = kmeans.labels_

    for v, k in zip(x, labels): clustered_img[k].append(v)

    variance = []
    for i in clustered_img.keys():
        variance.append(np.var(np.asarray(clustered_img.get(i))))
    variance = np.asarray(variance)
    # precision = 1 / variance
    weights = []

    for i in clustered_img.keys():
        weights.append(len(clustered_img.get(i)) / len(x))

    weights = np.asarray(weights)
    # resp = np.hstack((labels.reshape(-1,1), labels_1.reshape(-1, 1)))
    return (means.reshape(-1), variance, weights, labels)


class Sample:
    """Class for defining a single sample"""

    def __init__(self, mk, sk, pi, alpha_k, beta_k, alpha, M):
        self.mk = mk
        self.sk = sk
        self.pi = pi
        self.alpha_k = alpha_k
        self.beta_k = beta_k
        self.M = M
        self.alpha = alpha


class Samples:
    """Class for generating a collection of samples"""

    def __init__(self, N, D):
        self.sample = []
        self.N = N
        self.D = D

    def __getitem__(self, key):
        return self.sample[key]

    def addsample(self, S):
        return self.sample.append(S)

    # def infinte_mixutre_model(X, Nsamples=500, Nint=50, anneal=False):
    """
    infinite asymmetric gaussian distribution(AGD) mixture model
    using Gibbs sampling
    input:
        Y : the input datasets
        Nsamples : the number of Gibbs samples
        Nint : the samples used for evaluating the tricky integral
        anneal : perform simple siumulated annealing
    output:
        Samp : the output samples
        Y : the input datasets
    """


# compute some data derived quantities, N is observations number, D is dimensionality number

import cv2

input_img_path = r"C:\Users\Sunny\PycharmProjects\DataMiningClass_v4\datasets\testSample_copy.jpg"
X = x = cv2.imread(input_img_path, 0).reshape(-1)

# Nsamples=500
Nint = 50
anneal = False
N = X.shape[-1]
Nsamples = 50
muy = np.mean(X, axis=0)
# vary = np.zeros(D)
# for k in range(D):
#     vary[k] = np.var(X[:, k])

# initialise a single sample
# Samp = Samples(Nsamples, D)

o_shape = x.shape
x = x.reshape(-1)
# z = y_train

k = 1

c = []
alpha0 = np.asarray([x.mean() ** 2 / x.var() for _ in range(k)])
beta0 = np.asarray([x.mean() / i for i in alpha0])
shape = np.asarray([2 for _ in range(k)])
m0, s0, gama0, resp = initilization(k, x)
# pyplot.subplot(3, 1, 2)
# pyplot.imshow(resp.reshape(o_shape))
# pyplot.show()
temp = np.zeros((len(x), k))

c = []
import pandas

for i, j in zip(resp, temp):
    j[i] = 1
    c.append(j)

c = np.asarray(c)
# z = np.array([np.random.dirichlet(np.ones(k)) for _ in range(len(x))])
rnk = np.exp(c) / np.reshape(np.exp(c).sum(axis=1), (-1, 1))
Nk = rnk.sum(axis=0)
sk = s0 + 2 * Nk * (alpha0 / beta0)

test_mk_num = c.copy()
test_mk_den = c.copy()

mk = m0.copy()

e_precision_ = alpha0 / beta0

e_x_mean_lambda_ = c.copy()
shape = np.asarray([2 for _ in range(k)])
gammak = gama0 + Nk
alphak = Nk / 2 + alpha0 - 1
betak = beta0 + (rnk * e_x_mean_lambda_).sum(axis=0)

e_ln_pi = e_ln_pi_k(gammak, Nk)
e_ln_precision_ = digamma(alphak) - np.log(betak)
e_precision_ = alphak / betak

# Feature

term1 = (rnk * (digamma(alphak) - np.log(betak))).sum(axis=1) / 2
term2 = 1 / 2 * (rnk * (alphak / betak) * ((x.reshape(-1, 1) - mk.reshape(-1, 1).T) ** 2 + 1 / sk)).sum(axis=1)

row_in_e = np.exp(term1 - term2)  # Eq. 24

# w = np.asarray([1 for _ in range(k)])
w = 1

no_of_iterations = 0
epsolon = 1
var_test = 1

"""
Change c to z

"""

# c = np.zeros(N)  # initialise the stochastic indicators


alpha = 1.0 / draw_gamma(0.5, 2.0)
# set only 1 component, m is the component number
M = 1
n = np.zeros(1)
# define the sample
# S = Sample(mk, sk, pi, alpha_k, beta_k, alpha, M)
# Samp.addsample(S)  # add the sample
print('{}: initialised parameters'.format(time.asctime()))

# loop over samples
# z = 1
oldpcnt = 0

no_of_iterations = 0
z = np.zeros(N)
# Nsamples = 1


while no_of_iterations < Nsamples:
    c = c.reshape(-1, k)

    test_mk_num = c.copy()
    test_mk_den = c.copy()

    epsolon_in = np.exp(
        -1 / 2 * 1 / var_test * ((x.reshape(-1, 1) - epsolon) ** 2) + 1 / 2 * np.log(1 / var_test))

    num1 = w * row_in_e.reshape(-1, 1)
    den1 = w * row_in_e.reshape(-1, 1) + (1 - w) * epsolon_in
    fik = np.divide(num1, den1, out=np.zeros_like(num1), where=den1 != 0).reshape(-1)
    # fik = (w * row_in_e.reshape(-1, 1)) / (w * row_in_e.reshape(-1, 1) + (1 - w) * epsolon_in)
    for cluster in range(k):
        for i in range(len(x)):
            if x[i] > mk[cluster]:
                if x[i] != 0:
                    e_x_mean_lambda_[i, cluster] = x[i] ** shape[cluster] - shape[cluster] * x[i] ** shape[
                        cluster] / x[
                                                       i] * \
                                                   mk[
                                                       cluster] + shape[cluster] / 2 * (
                                                           shape[cluster] - 1) * x[i] ** shape[cluster] / x[
                                                       i] ** 2 * (
                                                           1 / sk[cluster] + mk[cluster] ** 2)



            else:

                t1 = -shape[cluster] * x[i] * e_mean_n(sk, mk, shape[cluster] - 1, k)
                e_mean2 = e_mean_n(sk, mk, shape[cluster] - 2, k)
                t2 = [0, 0]

                if shape[cluster] > 1:
                    t2 = shape[cluster] / 2 * (shape[cluster] - 1) * x[i] ** 2 * e_mean2

                e_x_mean_lambda_[i, cluster] = abs(
                    e_mean_n(sk, mk, shape[cluster], k)[cluster] + t1[cluster] + t2[cluster])

                # e_x_mean_lambda_[i, cluster] = abs(e_x_mean_lambda_[i, cluster])

    for cluster in range(k):
        for i in range(len(x)):
            if x[i] > mk[cluster]:
                test_mk_num[i, cluster] = fik[i] * rnk[i][cluster] * e_precision_[cluster] * shape[
                    cluster] * abs(
                    x[i] ** shape[cluster]) / (
                                              x[i])

                test_mk_den[i, cluster] = fik[i] * rnk[i][cluster] * e_precision_[cluster] * abs(x[i]) ** \
                                          shape[cluster] * shape[
                                              cluster] * (
                                                  shape[cluster] - 1) / (2 * x[i] ** 2)
            else:
                test_mk_den[i, cluster] = fik[i] * rnk[i][cluster] * e_precision_[cluster] * mk[
                    cluster] ** (
                                                  shape[cluster] - 2)

                if test_mk_den[i, cluster] <= 0:
                    test_mk_den[i, cluster] = 0

                term0 = fik[i] * rnk[i][cluster] * e_precision_[cluster] * shape[cluster] / 2 * mk[
                    cluster] ** (
                                shape[cluster] - 2) * x[i]

                term1 = fik[i] * rnk[i][cluster] * e_precision_[cluster] * shape[cluster] / 4 * (
                        shape[cluster] - 1) * mk[
                            cluster] ** (shape[cluster] - 3) * x[i] ** 2

                if term1 <= 0:
                    term1 = 0

                if term0 <= 0:
                    term0 = 0

                test_mk_num[i, cluster] = term1 + term0

        # numerator = test_mk_num.sum(axis=0)[cluster] + sk[cluster] * m0[cluster] / 2
        numerator = test_mk_num.sum(axis=0)[cluster] + sk[cluster] * mk[cluster] / 2
        sk[cluster] = test_mk_den.sum(axis=0)[cluster] + sk[cluster] / 2

        mk[cluster] = numerator / sk[cluster]

    # alphak = Nk / 2 + alpha0 - 1

    # betak = beta0 + (rnk * e_x_mean_lambda_).sum(axis=0)

    alphak = (rnk * fik.reshape(-1, 1)).sum(axis=0) / 2 + alpha0 - 1
    betak = beta0 + (rnk * fik.reshape(-1, 1) * e_x_mean_lambda_).sum(axis=0)

    gammak = gama0 + Nk

    w = fik.sum(axis=0) / len(fik)

    for i in range(k):
        p1 = e_ln_pi[i] + (1 / shape[i]) * e_ln_precision_[i] + np.log(shape[i]) - np.log(2 * gamma(1 / shape[i]))
        c[:, i] = (p1 - (e_precision_[i] * e_x_mean_lambda_[:, i]).reshape(-1, 1)).reshape(-1)

    z1_num = np.exp(c)
    z1_den = np.reshape(np.exp(c).sum(axis=0), (-1, 1)).T

    # rnk = np.exp(z1) / np.reshape(np.exp(z1).sum(axis=0), (-1, 1)).T
    rnk = np.divide(z1_num, z1_den, out=np.zeros_like(z1_num), where=z1_den != 0)

    term1 = (rnk * (digamma(alphak) - np.log(betak))).sum(axis=1) / 2

    term2 = 1 / 2 * (rnk * (alphak / betak) * ((x.reshape(-1, 1) - mk.reshape(-1, 1).T) ** 2 + 1 / sk)).sum(
        axis=1)
    row_in_e = np.exp(term1 - term2)

    epsolon_in = np.exp(
        -1 / 2 * 1 / var_test * ((x.reshape(-1, 1) - epsolon) ** 2) + 1 / 2 * np.log(
            1 / var_test))
    num1 = w * row_in_e.reshape(-1, 1)
    den1 = w * row_in_e.reshape(-1, 1) + (1 - w) * epsolon_in
    fik = np.divide(num1, den1, out=np.zeros_like(num1), where=den1 != 0).reshape(-1)
    # fik = (w * row_in_e.reshape(-1, 1)) / (w * row_in_e.reshape(-1, 1) + (1 - w) * epsolon_in)
    epsolon_num = (fik * x).sum(axis=0)

    epsolon_den = fik.sum(axis=0)
    epsolon = np.divide(epsolon_num, epsolon_den, out=np.zeros_like(epsolon_num), where=epsolon_den != 0)

    var_test_num = (fik * ((x - epsolon) ** 2)).sum(axis=0)

    var_test_den = fik.sum(axis=0)

    var_test = np.divide(var_test_num, var_test_den, out=np.zeros_like(var_test_num), where=var_test_den != 0)
    # epsolon = (fik * x_test.reshape(-1, 1)).sum(axis=0) / fik.sum(axis=0)
    # var_test = (fik * ((x_test - epsolon.reshape(-1, 1)) ** 2).T).sum(axis=0) / fik.sum(axis=0)

    for i in range(k):
        p1 = (1 / shape[i]) * e_ln_precision_[i] + np.log(shape[i]) - np.log(2 * gamma(1 / shape[i]))
        p2 = fik * (p1 - (e_precision_[i] * e_x_mean_lambda_[:, i]).reshape(-1, 1)).reshape(-1)
        # p3_3 = np.divide(1, var_test[i], out=np.zeros_like(1), where=var_test[i] != 0)
        if var_test.any() == 0:
            p3_3 = 0
        else:
            p3_3 = 1 / var_test

        p3 = (1 / 2) * np.log(1 / var_test) + np.log(2) - np.log(2 * gamma(1 / 2)) - p3_3 * (
                x - epsolon) ** 2
        p4 = e_ln_pi[i] + p2 + (1 - fik) * p3
        c[:, i] = p4

    # np.seterr(divide='ignore', invalid='ignore')
    # rnk = np.exp(z1) / np.reshape(np.exp(z1).sum(axis=0), (-1, 1)).T
    z1_num = np.exp(c)
    z1_den = np.reshape(np.exp(c).sum(axis=0), (-1, 1)).T

    # rnk = np.exp(z1) / np.reshape(np.exp(z1).sum(axis=0), (-1, 1)).T
    rnk = np.divide(z1_num, z1_den, out=np.zeros_like(z1_num), where=z1_den != 0)
    # return rnk

    """
    make changes
    """

    # recompute muy and covy
    # muy = np.mean(X, axis=0)
    # for k in range(D):
    #     vary[k] = np.var(X[:, k])
    # precisiony = 1 / vary
    #
    # # the observations belonged to class j

    X_reshape = X.reshape(-1, 1)
    Xj = [X_reshape[np.where(z == j), :] for j, nj in enumerate(n)]

    # mu_cache = mu
    # mu = np.zeros((M, D))
    j = 0
    # draw muj from posterior (depends on sj, c, lambda, r), eq 4 (Rasmussen 2000)

    """
    Calculate your mean and sk normal way
    """

    # draw lambda from posterior (depends on mu, M, and r), eq 5 (Rasmussen 2000)

    """
    This will remain the same for eq 5 in Rasmussen
    """

    # draw alpha from posterior (depends on number of components M, number of observations N), eq 15 (Rasmussen 2000)
    # Because its not standard form, using ARS to sampling
    alpha = draw_alpha(M, N)

    """
    Finished until alpha part, check the idx and include the idx> part
    """

    # compute the unrepresented probability - apply simulated annealing, eq 17 (Rasmussen 2000)
    p_unrep = (alpha / (N - 1.0 + alpha)) * integral_approx(X, mk, sk, shape)

    p_indicators_prior = np.outer(np.ones(M + 1), p_unrep)

    # for the represented components, eq 17 (Rasmussen 2000)

    """
    Change c to single dimension array
    """

    c_list = list(c)
    c_list = [list(i) for i in c_list]
    c_list_resp = []
    for i in c_list:
        c_list_resp.append(i.index(max(i)))
    c_list_resp = np.asarray(c_list_resp)
    for j in range(M):
        # n-i,j : the number of oberservations, excluding Xi, that are associated with component j
        nij = n[j] - (c_list_resp == j).astype(int)
        idx = np.argwhere(nij > 0)
        idx = idx.reshape(idx.shape[0])
        likelihood_for_associated_data = np.ones(len(idx))
        for i in range(len(idx)):
            # for k in range(D):
            #     Generalized_Gaussin_PDF()

            likelihood_for_associated_data[i] = (shape[j] * np.power(sk[j], 1 / shape[j])) / (
                    2 * gamma_pdf(1 / shape[j])) * np.exp(
                -sk[j] * np.power(np.abs(X[i] - mk[j]), shape[j]))

        p_indicators_prior[j, idx] = nij[idx] / (N - 1.0 + alpha) * likelihood_for_associated_data

    # stochastic indicator (we could have a new component)
    c = np.hstack(draw_indicator(p_indicators_prior))

    # sort out based on new stochastic indicators
    nij = np.sum(c == M)  # see if the *new* component has occupancy
    print("C: ",np.unique(c))
    print("NIJ: ", nij)
    temp_c_holder = c.copy()

    if nij > 0:
        newmu = np.array([np.squeeze(draw_normal(m0, 1 / s0))])
        new_sk = np.array([np.squeeze(draw_gamma(alpha0, beta0))])
        mk = np.concatenate((mk, np.reshape(newmu, -1)))
        sk = np.concatenate((sk, np.reshape(new_sk, -1)))
        shape = np.concatenate((shape, np.reshape(np.asarray([2]), -1)))
        M = M + 1
        k = M
        temp = np.zeros((len(x), k))
        z_for_c = []
        for i, j in zip(c, temp):
            j[int(i)] = 1
            z_for_c.append(j)
        z_for_c = np.asarray(z_for_c)
        c = z_for_c
        rnk = np.exp(c) / np.reshape(np.exp(c).sum(axis=1), (-1, 1))
        Nk = rnk.sum(axis=0)
        # alphak = (rnk * fik).sum(axis=0) / 2 + alpha0 - 1
        # betak = beta0 + (rnk * fik * e_x_mean_lambda_).sum(axis=0)
        alphak = Nk / 2 + alpha0 - 1
        e_x_mean_lambda_ = c.copy()
        for cluster in range(k):
            for i in range(len(x)):
                if x[i] > mk[cluster]:
                    if x[i] != 0:
                        e_x_mean_lambda_[i, cluster] = x[i] ** shape[cluster] - shape[cluster] * x[i] ** shape[
                            cluster] / x[
                                                           i] * \
                                                       mk[
                                                           cluster] + shape[cluster] / 2 * (
                                                               shape[cluster] - 1) * x[i] ** shape[cluster] / x[
                                                           i] ** 2 * (
                                                               1 / sk[cluster] + mk[cluster] ** 2)



                else:

                    t1 = -shape[cluster] * x[i] * e_mean_n(sk, mk, shape[cluster] - 1, k)
                    e_mean2 = e_mean_n(sk, mk, shape[cluster] - 2, k)
                    t2 = [0, 0]

                    if shape[cluster] > 1:
                        t2 = shape[cluster] / 2 * (shape[cluster] - 1) * x[i] ** 2 * e_mean2

                    e_x_mean_lambda_[i, cluster] = abs(
                        e_mean_n(sk, mk, shape[cluster], k)[cluster] + t1[cluster] + t2[cluster])

        betak = beta0 + (rnk * e_x_mean_lambda_).sum(axis=0)

        e_precision_ = alphak / betak
        gammak = gama0 + Nk
        e_ln_pi = e_ln_pi_k(gammak, Nk)
        e_ln_precision_ = digamma(alphak) - np.log(betak)

        # term1 = (rnk * (digamma(alphak) - np.log(betak))).sum(axis=1) / 2
        #
        # term2 = 1 / 2 * (rnk * (alphak / betak) * ((x.reshape(-1, 1) - mk.reshape(-1, 1).T) ** 2 + 1 / sk)).sum(
        #     axis=1)
        # row_in_e = np.exp(term1 - term2)
        # w = fik.sum(axis=0) / len(fik)

    # find the associated number for every components
    n = np.array([np.sum(temp_c_holder == j) for j in range(M)])
    # find unrepresented components
    badidx = np.argwhere(n == 0)
    Nbad = len(badidx)

    # remove unrepresented components
    if Nbad > 0:
        print("#################")
        print("Nbad > 0")
        mk = np.delete(mk, badidx, axis=0)
        sk = np.delete(sk, badidx, axis=0)
        betak = np.delete(betak, badidx, axis=0)
        alphak = np.delete(alphak, badidx, axis=0)
        shape = np.delete(shape, badidx, axis=0)
        e_precision_ = np.delete(e_precision_, badidx, axis=0)
        gammak = np.delete(gammak, badidx, axis=0)
        e_ln_pi = np.delete(e_ln_pi, badidx, axis=0)
        e_ln_precision_ = np.delete(e_ln_precision_, badidx, axis=0)
        e_x_mean_lambda_ = np.delete(e_x_mean_lambda_, badidx, axis=1)
        rnk = np.delete(rnk, badidx, axis=1)
        Nk = np.delete(Nk, badidx, axis=0)
        c = np.delete(c, badidx, axis=1)

        # s_r = np.delete(s_r, badidx, axis=0)
        # if the unrepresented compont removed is in the middle, make the sequential component indicators change
        for cnt, i in enumerate(badidx):
            idx = np.argwhere(temp_c_holder >= (i - cnt))
            temp_c_holder[idx] = temp_c_holder[idx] - 1
        M -= Nbad  # update component number
        k -=Nbad
    # recompute n
    n = np.array([np.sum(temp_c_holder == j) for j in range(M)])

    # recompute pi
    pi = n.astype(float) / np.sum(n)
    print('mk: ', mk.shape)

    pcnt = int(100.0 * no_of_iterations / float(Nsamples))
    if pcnt > oldpcnt:
        print('{}: %--- {}% complete ----------------------%'.format(time.asctime(), pcnt))
        oldpcnt = pcnt



    no_of_iterations += 1
    # print(n)
    result = []
    rnk12 = list(rnk)
    rnk12 = [list(i) for i in rnk]
    for response in rnk12:
        result.append(response.index(max(response)))
    result = np.asarray(result)
    pyplot.imshow(result.reshape(67, 100))
    pyplot.savefig(r"C:\Users\Sunny\PycharmProjects\DataMiningClass_v4\IGGMM_feature\output_eagle/" + str(no_of_iterations) + ".png")
    pyplot.show()


# return Samp, X, c, n
