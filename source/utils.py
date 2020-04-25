#
import numpy as np
import pandas as pd


# standardization function
def min_max(array): return (array - array.min(0)) / (array.max(0) - array.min(0))
def z_score(array,mean=None,std=None,calcute=False):
    if not calcute:
        return (array-mean)/std
    else:
        return (array - array.mean(0)) / array.std(0)

def angle(vij, vik):
    Lij=np.sqrt(vij.dot(vij))
    Lik=np.sqrt(vik.dot(vik))
    cos_angle=vij.dot(vik)/(Lij*Lik)
    angle=np.arccos(cos_angle)
    # angle2=angle*360/2/np.pi
    return angle#, angle2

# new feature calculation: category(one-hot)
def standard_category(category):
    particle_category = [22, -211, 211, -321, 321, 130, 2212, -2212, 2112, -2112, -11, 11, -13, 13]
    for i, a in enumerate(particle_category):
        category = np.where(category == a, i, category)
    return category[:, np.newaxis]


# new feature calculation: mass energy rate



def cal_energy(energy, mass):
    mass_e = mass/energy

    mc2 = mass * 9
    minus = energy - mc2
    pt = np.sqrt(np.power(energy, 2) - np.power(mass, 2))

    pow_minus = np.power(energy, 2) - np.power(mc2, 2)
    p = np.sqrt(np.clip(pow_minus, 0, None)) / 9

    ratio = mass / pt
    return np.concatenate(
        [mass_e.reshape(-1,1), minus.reshape(-1, 1), p.reshape(-1, 1), pt.reshape(-1, 1), ratio.reshape(-1, 1)],
        axis=1)  # (n,6)


def get_grouped_sorted_idx_list(messy_nums,data_dist):          # 代替 argwhere
    # in:  particle['jet_num']                        # messy jet_nums
    # out:
    #      [array([12094891, 11842640, ...]),         # idx for jet_num = 0
    #       array([9425742, 9706386, ...]),           # idx for jet_num = 1
    #       array([10402635, 10578281, ...]), ...]    # idx for jet_num = 2...
    # train_jet_nums_in_particle[out]:
    #       [[0,0,0,0,0,0,0,0,0,0,0,0],               # jet_num = 0
    #        [1,1,1,1,1,1,1,1,1,1,1,1,1,1],           # jet_num = 1
    #        [2,2,2,2,2,2,2,2,2], ... ]               # jet_num = 2...
    data = pd.DataFrame({
        'a': messy_nums,
        'index': np.arange(len(messy_nums))
    })
    # 根据 jet_nums排序，可以将index也排序好。
    sorted_data = data.sort_values('a')        # 7.7 s
    # 将 jet_nums的 分布找出来并按jet_nums大小排序
    # data_dist = data['a'].value_counts().sort_index().to_numpy()  # 数量分布   # 2.41 s
    # 将 index 分割，得到idx的列表
    list_idx = np.split(sorted_data['index'].to_numpy(), np.cumsum(data_dist))     # 1.64 s
    if len(list_idx[-1])==0:
        return list_idx[:-1]
    else:
        return list_idx







