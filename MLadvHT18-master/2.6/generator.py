import pickle
import random
import itertools
import numpy as np
from collections import defaultdict
from em import exp_max
import forward_backward as fb
from sklearn import mixture



#Variables you need
M = 30
N = 20
R = 10
beta = np.pi
sigma = 5
# mean for each subfield with respect to the player
mu = np.array([np.random.normal(10, sigma, (M,N)), 
               np.random.normal(23, sigma, (M,N))])
print ("Mean matrix shape: ",mu.shape)


#Function to save dictionary in a pickle file
def save_obj(obj, name ):
    with open('./'+ name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        

def generate_data():            
    '''
    Function to generate tournament data
    '''
    # Generate Players List and list of pairs
    players_list = [ i+1 for i in range (N)]
    player_pairs = list(itertools.combinations(players_list, 2))
    output_sequences = defaultdict(dict)

    for player_pair in player_pairs:
        output_sequences[player_pair] = []
        for i in range(R):
            sequence = []
            table_idx = 0     #If we are in left side of M or right side
            for j in range(M):
                choice = np.random.choice(['switch', 'stay'], p=[3/float(4), 1/float(4)])
                if choice == 'switch':
                    table_idx = 1 - table_idx
                player1_performance = np.random.normal(mu[table_idx,j,player_pair[0]-1], beta, 1)
                player2_performance = np.random.normal(mu[table_idx,j,player_pair[1]-1], beta, 1)
                sequence.append(player1_performance+player2_performance)

            output_sequences[player_pair] = sequence
    return output_sequences

def load_obj(name ):
    with open('./' + name + '.pkl', 'rb') as f:
        return pickle.load(f, encoding='latin1')





#print output_sequences
'''Save the dictionary'''
#save_obj(output_sequences,"sequence_output")

# dummy_data = generate_data()
#
# dummy_xs = list(dummy_data.keys())

# observations = dummy_data[dummy_xs[0]]
#
# alpha = list(fb.forward(fb.get_init, fb.get_transition, fb.get_emission, observations)[0])
# alpha2 = list(fb.backward(fb.get_init, fb.get_transition, fb.get_emission, observations)[0])

data = load_obj("sequence_output_2")

xs = list(data.keys())

players = []

for p in range(1,N+1):
    players.append(np.where([i[0] == p or i[1] == p for i in xs])[0])

mu_sums = []

for elem in xs:
    cur_set = data[elem]
    for field in range(1,N):
        obs = [cur_set[i][field][0] for i in range(1, R+1)] # use range because there was an error in data
        obs = np.asarray(obs).reshape(-1, 1)
        g = mixture.GMM(n_components=2, covariance_type='tied')
        g.fit(obs)
        mu_sums.append([elem, field, g.means_])
        print(str(elem)+ '   '+ str(field))


sum_mat = np.zeros([M, 2, M])



# for m in range(M):
#     sum_mat[m, 0, m:m+1] = alpha[m][0]
#     sum_mat[m, 1, m:m + 1] = alpha[m][1]
#
# sum_mat_lin = np.hstack((sum_mat[:,0,:],sum_mat[:,1,:])).transpose()
#
# x = np.linalg.lstsq(sum_mat_lin, observations)


# xs = np.array([(5,5), (9,1), (8,2), (4,6), (7,3)])
# thetas = np.array([[0.6, 0.4], [0.5, 0.5]])
# i, thetas, ll = exp_max(xs, thetas)
# print(i)
# for theta in thetas:
#     print(theta)
# print(ll)


a = 1