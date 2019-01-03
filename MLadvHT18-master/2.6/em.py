import numpy as np
import forward_backward as fb


def exp_max(xs, dist, max_iter=100, tol=1e-6):


    ll_old = -np.infty
    for i in range(max_iter):
        ll = np.array([np.sum(xs * np.log(theta), axis=1) for theta in dist])
        lik = np.exp(ll)
        ws = lik/lik.sum(0)
        vs = np.array([w[:, None] * xs for w in ws])
        dist = np.array([v.sum(0) / v.sum() for v in vs])
        ll_new = np.sum([w*l for w, l in zip(ws, ll)])
        if np.abs(ll_new - ll_old) < tol:
            break
        ll_old = ll_new
    return i, dist, ll_new


#
# # test examples
# print(fb.forward(fb.get_init, fb.get_transition, fb.get_emission, [0, 0, 1, 1, 1, 1]))
# print(fb.backward(fb.get_init, fb.get_transition, fb.get_emission, [0, 0, 1, 1, 1, 1]))



