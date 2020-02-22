"""
GGMM WITH FEATURE SELECTION
"""
import matplotlib.pyplot as plt
import math
from collections import defaultdict
from sklearn.cluster import KMeans
import mpmath
import numpy as np
from scipy import special
from scipy.special import digamma
from scipy.special import gamma
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import datetime
from sklearn.metrics import confusion_matrix

from sklearn.metrics import classification_report

start = datetime.datetime.now()


# E[ln pik]
def e_ln_pi_k(gama0, Nk):
    gammak = gama0 + Nk
    return digamma(gammak) - digamma(gammak.sum())


def e_mean_n(sk, mk, shape, k):
    """
    E(|mean^shape|)
    """

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


import pandas

# data = pandas.read_csv("/home/k_mathin/PycharmProjects/DataMiningClass/datasets/classification/covtype.csv",
#                        header=None,
#                        skiprows=1)
#
# index_value = len(data.keys()) - 1
#
# X = data.iloc[:, 0:index_value]  # slicing: all rows and 1 to 4 cols
# # store response vector in "y"
# y = data.iloc[:, index_value]
# k = 7
# X = np.asarray(X)
# y = np.asarray(y)

# #
# data = pandas.read_csv(
#     "/Users/Srikanth/PycharmProjects/COMP551_Projects/DataMiningClass_v4/datasets/classification/heart.csv",
#     header=None,
#     skiprows=1)

# input_img_path = r"C:\Users\Sunny\PycharmProjects\DataMiningClass_v4\datasets\testSample_copy.jpg"
input_img_path = r"C:\Users\Sunny\PycharmProjects\DataMiningClass_v4\datasets\testSample.jpg"
input_img_path = r"C:\Users\Sunny\PycharmProjects\DataMiningClass_v4\datasets\testSamplelessres.jpg"
import cv2
from matplotlib import pyplot

X = cv2.imread(input_img_path, 0)

# X = data.iloc[:, 0:13]  # slicing: all rows and 1 to 4 cols
# # store response vector in "y"
# y = data.iloc[:, 13]
k = 2
x = np.asarray(X)
# y = np.asarray(y)
#

# return 0
# print("Dim: ", dim)
# x = X_train[:, dim]
o_shape = x.shape
x = x.reshape(-1)
# z = y_train
c = []
alpha0 = np.asarray([x.mean() ** 2 / x.var() for _ in range(k)])
beta0 = np.asarray([x.mean() / i for i in alpha0])
shape = np.asarray([2 for _ in range(k)])
m0, s0, gama0, resp = initilization(k, x)
# pyplot.subplot(3, 1, 2)
pyplot.imshow(resp.reshape(o_shape))
pyplot.show()
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
while no_of_iterations < 50:
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

        numerator = test_mk_num.sum(axis=0)[cluster] + sk[cluster] * m0[cluster] / 2
        sk[cluster] = test_mk_den.sum(axis=0)[cluster] + sk[cluster] / 2

        mk[cluster] = numerator / sk[cluster]

    # alphak = Nk / 2 + alpha0 - 1

    # betak = beta0 + (rnk * e_x_mean_lambda_).sum(axis=0)

    alphak = (rnk * fik.reshape(-1, 1)).sum(axis=0) / 2 + alpha0 - 1
    betak = beta0 + (rnk * fik.reshape(-1, 1) * e_x_mean_lambda_).sum(axis=0)

    gammak = gama0 + Nk

    #
    # temp1 = np.zeros((len(x_test), k))
    # z1 = []
    #
    # for i, j in zip(y_test, temp1):
    #     # j[i - 1] = 1
    #     j[i] = 1
    #     z1.append(j)
    #
    # z1 = np.asarray(z1)
    #
    # e_x_mean_lambda_ = z1.copy()
    #
    # for cluster in range(k):
    #     for i in range(len(x_test)):
    #         if x_test[i] > mk[cluster]:
    #             if x_test[i] != 0:
    #                 e_x_mean_lambda_[i, cluster] = x_test[i] ** shape[cluster] - \
    #                                                shape[cluster] * x_test[i] ** shape[cluster] / x_test[i] * mk[
    #                                                    cluster] + \
    #                                                shape[cluster] / 2 * (shape[cluster] - 1) * x_test[i] ** shape[
    #                                                    cluster] / \
    #                                                x_test[i] ** 2 * (
    #                                                        1 / sk[cluster] +
    #                                                        mk[cluster] ** 2)
    #
    #         else:
    #
    #             t1 = -shape[cluster] * x_test[i] * e_mean_n(sk, mk, shape[cluster] - 1, k)
    #             e_mean2 = e_mean_n(sk, mk, shape[cluster] - 2, k)
    #
    #             t2 = [0, 0]
    #             if shape[cluster] > 1:
    #                 t2 = shape[cluster] / 2 * (shape[cluster] - 1) * x_test[i] ** 2 * e_mean2
    #
    #             e_x_mean_lambda_[i, cluster] = e_mean_n(sk, mk, shape[cluster], k)[cluster] + t1[cluster] + t2[
    #                 cluster]
    #
    #         e_x_mean_lambda_[i, cluster] = abs(e_x_mean_lambda_[i, cluster])

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

    result = []
    counter = 0
    segmentedImage = np.zeros((len(x), 1), np.uint8)

    # assigning values to pixels of different segments
    rnk1 = list(rnk)
    rnk1 = [list(i) for i in rnk1]
    m12 = [0, 100]
    for response in rnk1:
        maxResp = max(response)
        respmax = response.index(maxResp)
        result.append(respmax)
        segmentedImage[counter] = m0[respmax]
        counter = counter + 1

    segmentedImage = segmentedImage.reshape(o_shape[0], o_shape[1])

    # pyplot.subplot(3, 1, 3)

    pyplot.imshow(segmentedImage)
    # pyplot.savefig("/Users/Srikanth/PycharmProjects/DataMiningClass/outputs/starfish/" + str(count) + ".png")
    pyplot.show()

    no_of_iterations += 1
