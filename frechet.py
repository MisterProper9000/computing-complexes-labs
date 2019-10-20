import numpy as np
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

answer = -1
answer_ind = []

def checkAnswer(dst, i, j, pred):
    global answer, answer_ind
    if pred == 0: pair_prev = [i-1, j]
    elif pred == 1: pair_prev = [i-1, j-1]
    else: pair_prev = [i, j-1]
    if (answer == dst):
        answer = dst
        answer_ind.append([i, j])
    else:
        answer = dst
        answer_ind.clear()
        answer_ind.append([i, j])
        answer_ind.append(pair_prev)
    return

def _c(ca, i, j, p, q):
    global answer_ind, answer
    if ca[i, j] > -1:
        return ca[i, j]
    elif i == 0 and j == 0:
        ca[i, j] = np.linalg.norm(p[i]-q[j])
        answer = ca[0, 0]
        answer_ind.append([0,0])
    elif i > 0 and j == 0:
        t_dst = np.linalg.norm(p[i] - q[j])
        ca[i, j] = max(_c(ca, i - 1, 0, p, q), t_dst)
        if ca[i - 1, j] == t_dst:
            checkAnswer(t_dst, i, j, 0)

    elif i == 0 and j > 0:
        t_dst = np.linalg.norm(p[i] - q[j])
        ca[i, j] = max(_c(ca, 0, j - 1, p, q), t_dst)
        if ca[i, j - 1] == t_dst:
            checkAnswer(t_dst, i, j, 2)

    elif i > 0 and j > 0:
        t_dst = np.linalg.norm(p[i] - q[j])
        ca[i, j] = max(
            min(
                _c(ca, i - 1, j, p, q),
                _c(ca, i - 1, j - 1, p, q),
                _c(ca, i, j - 1, p, q)
            ),
            t_dst
        )
        if ca[i - 1, j - 1] == t_dst:
            checkAnswer(t_dst, i, j, 1)
        elif ca[i - 1, j] == t_dst:
            checkAnswer(t_dst, i, j, 0)
        elif ca[i, j - 1] == t_dst:
            checkAnswer(t_dst, i, j, 2)
    else:
        ca[i, j] = float('inf')

    return ca[i, j]


def frechetDist(p, q):
    p = np.array(p, np.float64)
    q = np.array(q, np.float64)

    len_p = len(p)
    len_q = len(q)

    if len_p == 0 or len_q == 0:
        raise ValueError('Input curves are empty.')

    ind1 = len_p - 1
    ind2 = len_q - 1
    ca = (np.ones((len_p, len_q), dtype=np.float64) * -1)
    dist = _c(ca, len_p - 1, len_q - 1, p, q)
    while (True):
        if (ind1 > 0 and ind2 > 0 and dist == ca[ind1 - 1, ind2 - 1]):
            ind1 = ind1 - 1
            ind2 = ind2 - 1
        if (ind1 > 0 and dist == ca[ind1 - 1, ind2]):
            ind1 = ind1 - 1
        elif (ind2 > 0 and dist == ca[ind1, ind2 - 1]):
            ind2 = ind2 - 1
        else:
            break
    return dist, ind1, ind2

def draw(axes, P, Q, ind1, ind2, cls, addit_ind):
    polygon_1 = mpatches.Polygon(P,
                                 fill=False,
                                 closed=cls, color='red', linewidth=2)
    axes.add_patch(polygon_1)
    polygon_2 = mpatches.Polygon(Q,
                                 fill=False,
                                 closed=cls, color='blue')
    axes.add_patch(polygon_2)
    line_1 = mpatches.Polygon([P[ind1], Q[ind2]], color='m', linestyle=':', linewidth=2)
    axes.add_patch(line_1)
    for pair in addit_ind:
        line = mpatches.Polygon([P[pair[0]], Q[pair[1]]], color='m', linestyle=':', linewidth=2)
        axes.add_patch(line)

ax = plt.gca()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Fresche distance')
ax.grid(True)
plt.legend(handles=[
    mpatches.Patch(color='red', label='P'),
    mpatches.Patch(color='blue', label='Q'),
    mpatches.Patch(color='magenta', label='dist')
], loc='best')


plt.cla()
P = [[2, 3], [3, 4], [2, 7], [5, 6], [9, 8], [6, 18], [10, 1], [6, 3]]
Q = [[12, 1], [10, 3], [6, 6], [9, 7], [10, 5], [12, 6], [15, 5], [13, 3]]
closed = False
d, ind1, ind2 = frechetDist(P, Q)
print(d)
print(ind1, ind2)
print(answer_ind, answer)
draw(ax, P, Q, ind1, ind2, closed, answer_ind)
plt.plot()
plt.show()


#P=[[1,1], [2,1], [2,2]]
#Q=[[2,2], [0,1], [2,4]]

#P=[[1,1], [2,1], [2,2]]
#Q=[[1,1], [2,1], [2,2]]

#P = [[0, 0],[4, 2],[6, 5],[12, 6],[15, 7],[15, 10],[18, 13]]
#Q = [[1, 1],[2, 5],[7, 7],[8, 12],[13, 14],[15, 16],[15,16]]