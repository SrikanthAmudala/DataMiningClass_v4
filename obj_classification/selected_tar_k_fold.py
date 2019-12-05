# labels = "BACKGROUND_Google,Faces,Faces_easy,Leopards,Motorbikes,accordion,airplanes,anchor,ant,barrel,bass,beaver,binocular,bonsai,brain,brontosaurus,buddha,butterfly,camera,cannon,car_side,ceiling_fan,cellphone,chair,chandelier,cougar_body,cougar_face,crab,crayfish,crocodile,crocodile_head,cup,dalmatian,dollar_bill,dolphin,dragonfly,electric_guitar,elephant,emu,euphonium,ewer,ferry,flamingo,flamingo_head,garfield,gerenuk,gramophone,grand_piano,hawksbill,headphone,hedgehog,helicopter,ibis,inline_skate,joshua_tree,kangaroo,ketch,lamp,laptop,llama,lobster,lotus,mandolin,mayfly,menorah,metronome,minaret,nautilus,octopus,okapi,pagoda,panda,pigeon,pizza,platypus,pyramid,revolver,rhino,rooster,saxophone,schooner,scissors,scorpion,sea_horse,snoopy,soccer_ball,stapler,starfish,stegosaurus,stop_sign,strawberry,sunflower,tick,trilobite,umbrella,watch,water_lilly,wheelchair,wild_cat,windsor_chair,wrench,yin_yang"
# labels = labels.split(",")
#
# data_len = [468,
#             435,
#             435,
#             200,
#             798,
#             55,
#             800,
#             42,
#             42,
#             47,
#             54,
#             46,
#             33,
#             128,
#             98,
#             43,
#             85,
#             91,
#             50,
#             43,
#             123,
#             47,
#             59,
#             62,
#             107,
#             47,
#             69,
#             73,
#             70,
#             50,
#             51,
#             57,
#             67,
#             52,
#             65,
#             68,
#             75,
#             64,
#             53,
#             64,
#             85,
#             67,
#             67,
#             45,
#             34,
#             34,
#             51,
#             99,
#             100,
#             42,
#             54,
#             88,
#             80,
#             31,
#             64,
#             86,
#             114,
#             61,
#             81,
#             78,
#             41,
#             66,
#             43,
#             40,
#             87,
#             32,
#             76,
#             55,
#             35,
#             39,
#             47,
#             38,
#             45,
#             53,
#             34,
#             57,
#             82,
#             59,
#             49,
#             40,
#             63,
#             39,
#             84,
#             57,
#             35,
#             64,
#             45,
#             86,
#             59,
#             64,
#             35,
#             85,
#             49,
#             86,
#             75,
#             239,
#             37,
#             59,
#             34,
#             56,
#             39,
#             60]
#
# base_path = '/home/k_mathin/PycharmProjects/DataMiningClass/datasets/classification/aero_bike/'
# base_path = "/home/k_mathin/PycharmProjects/DataMiningClass/datasets/classification/object_classification/with_targets/"

# for i in csv_files:
#     df = pandas.read_csv(
#         base_path + i,
#         header=None)
#     df1 = df.assign(target=bike)
#     df1.to_csv(base_path + 'with_targets/' + i, index=False)

# for i in csv_files:
#    ...:    df = pandas.read_csv(
#    ...:        base_path+i,
#    ...:        header=None)
#    ...:    df1 = df.assign(target= bike)
#    ...:    df1.to_csv(base_path+'with_targets/'+i, index=False)

"""
WIth out K fold and handled all the warnings from numpy
Feature selection 55 features
https://archive.ics.uci.edu/ml/datasets/Covertype
"""
import matplotlib.pyplot as plt
import math
from sklearn.model_selection import train_test_split
from collections import defaultdict
import pandas
import mpmath
import numpy as np
from scipy import special
from scipy.special import digamma
from scipy.special import gamma
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import datetime
import multiprocessing
import logging

start = datetime.datetime.now()
results = []

FORMAT = "%(asctime)-15s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE_PATH = "/home/k_mathin/PycharmProjects/DataMiningClass/logs/selected_targets_sunflower_heli_stop_revolver.txt"
logging.basicConfig(level=logging.DEBUG, filename=LOG_FILE_PATH,
                    format=FORMAT, datefmt="%a, %d %b %Y %H:%M:%S")
logger = logging.getLogger(__name__)


def collect_result(result):
    global results
    results.append(result)


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


def dim_calc(dim, X_train, X_test, y_train, y_test, alpthak_list, betak_list, mk_list, gammak_list, sk_list):
    # return 0
    # print("Dim: ", dim)
    x = X_train[:, dim]
    o_shape = x.shape
    x = x.reshape(-1)
    # z = y_train
    temp = np.zeros((len(y_train), k))
    z = []
    for i, j in zip(y_train, temp):
        # j[i - 1] = 1
        j[i] = 1
        z.append(j)
    z = np.asarray(z)
    rnk = np.exp(z) / np.reshape(np.exp(z).sum(axis=1), (-1, 1))
    Nk = rnk.sum(axis=0)

    alpha0 = np.asarray([x.mean() ** 2 / x.var() for _ in range(k)])
    beta0 = np.asarray([x.mean() / i for i in alpha0])

    clustered_img = defaultdict(list)
    for v, v1 in zip(x, y_train): clustered_img[v1].append(v)

    gamma0 = []

    for i in clustered_img.keys():
        gamma0.append(len(clustered_img.get(i)) / len(x))
    gamma0 = np.asarray(gamma0)

    variance = []
    for i in clustered_img.keys():
        variance.append(np.var(np.asarray(clustered_img.get(i))))

    sk = np.asarray(variance)
    sk_list.append(sk)
    m0 = []
    for i in clustered_img.keys():
        m0.append(np.asarray(clustered_img.get(i)).mean())
    mk = np.asarray(m0)
    mk_list.append(mk)

    test_mk_num = z.copy()
    test_mk_den = z.copy()

    e_x_mean_lambda_ = z.copy()
    shape = np.asarray([2 for _ in range(k)])
    gammak = gamma0 + Nk
    alphak = Nk / 2 + alpha0 - 1
    betak = beta0 + (rnk * e_x_mean_lambda_).sum(axis=0)

    e_ln_pi = e_ln_pi_k(gammak, Nk)
    e_ln_precision_ = digamma(alphak) - np.log(betak)
    # e_precision_ = alphak / betak
    e_precision_ = np.divide(alphak, betak, out=np.zeros_like(alphak), where=betak != 0)

    # Feature
    term1 = (rnk * (digamma(alphak) - np.log(betak))).sum(axis=1) / 2
    if sk.any() == 0 or sk.any() == np.nan:
        pass
        term2 = 1 / 2 * (rnk * (alphak / betak) * ((x.reshape(-1, 1) - mk.reshape(-1, 1).T) ** 2 + 0)).sum(axis=1)
    else:
        # temp1  = np.divide(alphak, betak, out=np.zeros_like(alphak), where=betak!=0)
        try:
            with np.errstate(divide='raise'):
                term2 = 1 / 2 * (rnk * (e_precision_) * ((x.reshape(-1, 1) - mk.reshape(-1, 1).T) ** 2 + 1 / sk)).sum(
                    axis=1)
        except Exception as e:
            # print("found: ", e)
            term2 = 0
    row_in_e = np.exp(term1 - term2)
    w = np.asarray([1 for _ in range(k)])

    epsolon = mk
    var_test = sk

    if var_test.any() == 0 or var_test.any() != np.nan:
        pass
        epsolon_in = np.exp(
            -1 / 2 * 0 * ((x.reshape(-1, 1) - epsolon.reshape(-1, 1).T) ** 2) + 1 / 2 * np.log(1))
    else:

        epsolon_in = np.exp(
            -1 / 2 * 1 / var_test * ((x.reshape(-1, 1) - epsolon.reshape(-1, 1).T) ** 2) + 1 / 2 * np.log(1 / var_test))

    num1 = w * row_in_e.reshape(-1, 1)
    den1 = w * row_in_e.reshape(-1, 1) + (1 - w) * epsolon_in
    fik = np.divide(num1, den1, out=np.zeros_like(num1), where=den1 != 0)
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
                test_mk_num[i, cluster] = fik[i][cluster] * rnk[i][cluster] * e_precision_[cluster] * shape[
                    cluster] * abs(
                    x[i] ** shape[cluster]) / (
                                              x[i])

                test_mk_den[i, cluster] = fik[i][cluster] * rnk[i][cluster] * e_precision_[cluster] * abs(x[i]) ** \
                                          shape[cluster] * shape[
                                              cluster] * (
                                                  shape[cluster] - 1) / (2 * x[i] ** 2)
            else:
                test_mk_den[i, cluster] = fik[i][cluster] * rnk[i][cluster] * e_precision_[cluster] * mk[
                    cluster] ** (
                                                  shape[cluster] - 2)

                if test_mk_den[i, cluster] <= 0:
                    test_mk_den[i, cluster] = 0

                term0 = fik[i][cluster] * rnk[i][cluster] * e_precision_[cluster] * shape[cluster] / 2 * mk[
                    cluster] ** (
                                shape[cluster] - 2) * x[i]

                term1 = fik[i][cluster] * rnk[i][cluster] * e_precision_[cluster] * shape[cluster] / 4 * (
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

    alphak = (rnk * fik).sum(axis=0) / 2 + alpha0 - 1
    betak = beta0 + (rnk * fik * e_x_mean_lambda_).sum(axis=0)

    alpthak_list.append(alphak)
    betak_list.append(betak)

    gammak = gamma0 + Nk
    gammak_list.append(gammak)

    # L1 = lowerbound_first_dir(rnk, x, shape, mk, sk, e_precision_).sum(axis=0)
    # L2 = lowerbound_second_dir(rnk, x, shape, mk, sk, e_precision_).sum(axis=0)
    # L1 += sys.float_info.epsilon
    # L2 += sys.float_info.epsilon
    # delta_lowerbound = L1 / L2
    # delta_lowerbound += sys.float_info.epsilon
    # shape = shape - 0.01 * (delta_lowerbound)
    x_test = X_test[:, dim]

    temp1 = np.zeros((len(x_test), k))
    z1 = []

    for i, j in zip(y_test, temp1):
        # j[i - 1] = 1
        j[i] = 1
        z1.append(j)

    z1 = np.asarray(z1)

    e_x_mean_lambda_ = z1.copy()

    for cluster in range(k):
        for i in range(len(x_test)):
            if x_test[i] > mk[cluster]:
                if x_test[i] != 0:
                    e_x_mean_lambda_[i, cluster] = x_test[i] ** shape[cluster] - \
                                                   shape[cluster] * x_test[i] ** shape[cluster] / x_test[i] * mk[
                                                       cluster] + \
                                                   shape[cluster] / 2 * (shape[cluster] - 1) * x_test[i] ** shape[
                                                       cluster] / \
                                                   x_test[i] ** 2 * (
                                                           1 / sk[cluster] +
                                                           mk[cluster] ** 2)

            else:

                t1 = -shape[cluster] * x_test[i] * e_mean_n(sk, mk, shape[cluster] - 1, k)
                e_mean2 = e_mean_n(sk, mk, shape[cluster] - 2, k)

                t2 = [0, 0]
                if shape[cluster] > 1:
                    t2 = shape[cluster] / 2 * (shape[cluster] - 1) * x_test[i] ** 2 * e_mean2

                e_x_mean_lambda_[i, cluster] = e_mean_n(sk, mk, shape[cluster], k)[cluster] + t1[cluster] + t2[
                    cluster]

            e_x_mean_lambda_[i, cluster] = abs(e_x_mean_lambda_[i, cluster])

    w = fik.sum(axis=0) / len(fik)

    for i in range(k):
        p1 = e_ln_pi[i] + (1 / shape[i]) * e_ln_precision_[i] + np.log(shape[i]) - np.log(2 * gamma(1 / shape[i]))
        z1[:, i] = (p1 - (e_precision_[i] * e_x_mean_lambda_[:, i]).reshape(-1, 1)).reshape(-1)

    z1_num = np.exp(z1)
    z1_den = np.reshape(np.exp(z1).sum(axis=0), (-1, 1)).T

    # rnk = np.exp(z1) / np.reshape(np.exp(z1).sum(axis=0), (-1, 1)).T
    rnk = np.divide(z1_num, z1_den, out=np.zeros_like(z1_num), where=z1_den != 0)

    term1 = (rnk * (digamma(alphak) - np.log(betak))).sum(axis=1) / 2

    term2 = 1 / 2 * (rnk * (alphak / betak) * ((x_test.reshape(-1, 1) - mk.reshape(-1, 1).T) ** 2 + 1 / sk)).sum(
        axis=1)
    row_in_e = np.exp(term1 - term2)

    epsolon_in = np.exp(
        -1 / 2 * 1 / var_test * ((x_test.reshape(-1, 1) - epsolon.reshape(-1, 1).T) ** 2) + 1 / 2 * np.log(
            1 / var_test))
    num1 = w * row_in_e.reshape(-1, 1)
    den1 = w * row_in_e.reshape(-1, 1) + (1 - w) * epsolon_in
    fik = np.divide(num1, den1, out=np.zeros_like(num1), where=den1 != 0)
    # fik = (w * row_in_e.reshape(-1, 1)) / (w * row_in_e.reshape(-1, 1) + (1 - w) * epsolon_in)
    epsolon_num = (fik * x_test.reshape(-1, 1)).sum(axis=0)

    epsolon_den = fik.sum(axis=0)
    epsolon = np.divide(epsolon_num, epsolon_den, out=np.zeros_like(epsolon_num), where=epsolon_den != 0)

    var_test_num = (fik * ((x_test - epsolon.reshape(-1, 1)) ** 2).T).sum(axis=0)
    var_test_den = fik.sum(axis=0)

    var_test = np.divide(var_test_num, var_test_den, out=np.zeros_like(var_test_num), where=var_test_den != 0)
    # epsolon = (fik * x_test.reshape(-1, 1)).sum(axis=0) / fik.sum(axis=0)
    # var_test = (fik * ((x_test - epsolon.reshape(-1, 1)) ** 2).T).sum(axis=0) / fik.sum(axis=0)

    for i in range(k):
        p1 = (1 / shape[i]) * e_ln_precision_[i] + np.log(shape[i]) - np.log(2 * gamma(1 / shape[i]))
        p2 = fik[:, i] * (p1 - (e_precision_[i] * e_x_mean_lambda_[:, i]).reshape(-1, 1)).reshape(-1)
        # p3_3 = np.divide(1, var_test[i], out=np.zeros_like(1), where=var_test[i] != 0)
        if var_test[i].any() == 0 or var_test[i].any() == np.nan:
            p3_3 = 0
        else:
            p3_3 = 1 / var_test[i]
        try:
            with np.errstate(divide='raise'):
                p3 = (1 / 2) * np.log(1 / var_test[i]) + np.log(2) - np.log(2 * gamma(1 / 2)) - p3_3 * (
                        x_test - epsolon[i]) ** 2
        except Exception as e:
            # print("found: ", e)
            p3 = (1 / 2) * np.log(1) + np.log(2) - np.log(2 * gamma(1 / 2)) - p3_3 * (
                    x_test - epsolon[i]) ** 2
        # p3 = (1 / 2) * np.log(1 / var_test[i]) + np.log(2) - np.log(2 * gamma(1 / 2)) - p3_3 * (
        #         x_test - epsolon[i]) ** 2
        p4 = e_ln_pi[i] + p2 + (1 - fik[:, i]) * p3
        z1[:, i] = p4

    # np.seterr(divide='ignore', invalid='ignore')
    # rnk = np.exp(z1) / np.reshape(np.exp(z1).sum(axis=0), (-1, 1)).T
    z1_num = np.exp(z1)
    z1_den = np.reshape(np.exp(z1).sum(axis=0), (-1, 1)).T

    # rnk = np.exp(z1) / np.reshape(np.exp(z1).sum(axis=0), (-1, 1)).T
    rnk = np.divide(z1_num, z1_den, out=np.zeros_like(z1_num), where=z1_den != 0)
    # print("Rnk: ", len(rnk))
    return rnk


base_path = "/home/k_mathin/PycharmProjects/DataMiningClass/datasets/classification/object_classification/yin_airpl_m_sun/"

import os

accuracy_table = []
for path in os.listdir(base_path):
    consider_files = [
                      "target_6_4_101_91_MICC_F220_bow_200.csv",
                      "target_6_4_101_91_MICC_F220_bow_800.csv"
                      "target_6_4_101_91_MICC_F220_bow_400.csv",
                      "target_6_4_101_91_MICC_F220_bow_90.csv",
                      "target_6_4_101_91_MICC_F220_bow_300.csv"
                      ]
    if path in consider_files:
        # if path in ["MICC_F220_bow_20.csv"]:

        data = pandas.read_csv(base_path + path,
                               header=None,
                               skiprows=1)
        # data = data.head(2336)  # taking first 5 cols
        X = data.iloc[:, 0:len(data.keys()) - 1]  # slicing: all rows and 1 to 4 cols
        # store response vector in "y"
        y = data.iloc[:, len(data.keys()) - 1]
        k = len(np.unique(y))
        X = np.asarray(X)
        y = np.asarray(y)

        skf = StratifiedKFold(n_splits=4)
        skf.get_n_splits(X, y)
        accuracy_list = []

        for train_index, test_index in skf.split(X, y):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

            X_train = np.asarray(X_train)
            X_test = np.asarray(X_test)
            y_train = np.asarray(y_train)
            y_test = np.asarray(y_test)

            d = X_train.shape[-1]

            z_list = []
            alpthak_list = []
            betak_list = []
            mk_list = []
            gammak_list = []
            sk_list = []
            print("Dataset: ", path)
            logger.info("Dataset" + path)

            maximum = d
            limit = 100
            while (maximum > 0):
                pool = multiprocessing.Pool(limit)
                print("Remaining Dimensions: ", maximum)
                for dim in range(d):
                    # print(dim)
                    if dim < limit:
                        pool.apply_async(dim_calc, args=(
                            dim, X_train, X_test, y_train, y_test, alpthak_list, betak_list, mk_list, gammak_list, sk_list),
                                         callback=collect_result)
                        # rnk = dim_calc(dim, X_train, X_test, y_train, y_test, alpthak_list, betak_list, mk_list, gammak_list, sk_list)

                        # z_list.append(rnk)
                #
                pool.close()
                pool.join()
                # limit +=100
                maximum = maximum - limit

            z_list = np.asarray(results).sum(axis=0)
            # z_list = np.asarray(results).prod(axis=0)
            results.clear()
            rnk = list(z_list)
            rnk = [list(i) for i in rnk]
            result = []

            for response in rnk:
                maxResp = max(response)
                respmax = response.index(maxResp)
                result.append(respmax)

            result = np.asarray(result)
            accuracy = metrics.accuracy_score(y_test, result)
            print("Accuracy: " + path, accuracy)
            logger.info("Accuracy: " + path + " : " + str(accuracy))
            accuracy_table.append([path, accuracy])

            accuracy_list.append(accuracy)

        accuracy_list = np.asarray(accuracy_list)
        print("Final Accuracy: " + path, accuracy_list.mean())
        logger.info("K-fold_Accuracy: " + path + " : " + str(accuracy_list.mean()))