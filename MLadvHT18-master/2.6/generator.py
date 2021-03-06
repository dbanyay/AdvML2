import pickle
import random
import itertools
import numpy as np
from collections import defaultdict
from sklearn import mixture
import warnings
import matplotlib.pyplot as plt
from scipy.stats import norm

warnings.filterwarnings("ignore", category=DeprecationWarning)



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


mat = []
mu_sums = []
betas = []

# for elem in range(1,N+1):
#     if elem == N:
#         key = tuple([1, N])
#     else:
#         key = tuple([elem,elem+1])
#     cur_set = data[key]
#     for field in range(M):
#         obs = [cur_set[i][field][0] for i in range(1, R+1)] # use range because there was an error in data
#         obs = np.asarray(obs).reshape(-1, 1)
#         g = mixture.GMM(n_components=2, covariance_type='tied')
#         g.fit(obs)
#         mu_sums.append([g.means_[0][0], g.means_[1][0]])
#         betas.append(g.covars_[0][0])
#         vect = np.zeros((N+1)*M)
#         vect[(elem-1)*M+field] = 1
#         vect[(elem)*M+field] = 1
#         mat.append(vect)
#
#         print(str(key)+ '   '+ str(field))
#
#
# np.save("beta_s", betas)
# np.save("mu_sums_s", mu_sums)
# np.save("mat_s", mat)


mat = np.load("mat_s.npy")

mat = mat[0:570,0:570]
mu_sums = np.load("mu_sums_s.npy")

betas = np.load("beta_s.npy")

# plt.imshow(mat)
# plt.show()

mu_sums0 = mu_sums[0:570,0]
mu_sums1 = mu_sums[0:570,1]




sol0 = np.linalg.solve(mat,mu_sums0)
mu, std = norm.fit(sol0)
x = np.linspace(np.min(sol0), np.max(sol0), 100)
p = norm.pdf(x,mu,std)
plt.plot(x, p, 'k', linewidth = 2)
plt.title("Normal Distribution of field 0:  mu = %.2f, beta = %.2f" % (mu, std))
plt.show()


sol1 = np.linalg.solve(mat,mu_sums1)
mu, std = norm.fit(sol1)
x = np.linspace(np.min(sol1), np.max(sol1), 100)
p = norm.pdf(x,mu,std)
plt.plot(x, p, 'k', linewidth = 2)
plt.title("Normal Distribution of field 0:  mu = %.2f, beta = %.2f" % (mu, std))
plt.show()


a = 1
