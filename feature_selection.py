import pandas as pd
import statsmodels.api as sm
import numpy as np

class FeatureSelection(object):
    def __init__(self):
        pass

    @staticmethod
    def forward_selection(data, target, significance_level=0.05): # 9 pts
        forward_list = []
        N, D = data.shape
        x_final = np.ones((N, 1))
        used_feature = []

        for j in range(D):
            p_min = 1e3
            count_p = 0
            for i in range(D):
                if i in used_feature:
                    continue
                x = np.concatenate((x_final, data.values[:, i].reshape(N, 1)), axis=1)
                model = sm.OLS(target.values, x).fit()
                p = model.pvalues[-1]
                if p < significance_level:
                    count_p += 1
                    if p < p_min:
                        p_min = p
                        add_feature = i
            if count_p == 0:
                break
            x_final = np.concatenate((x_final, data.values[:, add_feature].reshape(N, 1)), axis=1)
            forward_list.append(data.axes[1][add_feature])
            used_feature.append(add_feature)

        return forward_list, x_final

    @staticmethod
    def backward_elimination(data, target, significance_level = 0.05): # 9 pts
        backward_list = data.axes[1].tolist()
        N, D = data.shape
        x_final = sm.add_constant(data.values)
        removed_feature = []
        for i in range(D):
            model = sm.OLS(target.values, x_final).fit()
            p_values = model.pvalues[1:]
            pmax_ind = np.argmax(p_values)
            if p_values[pmax_ind] > significance_level:
                removed_feature.append(backward_list[pmax_ind])
                x_final = np.delete(x_final, pmax_ind + 1, axis=1)
                del backward_list[pmax_ind]

            else:
                break

        return backward_list, x_final
