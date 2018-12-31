import pickle
from ex_2_3 import Node, load_params, load_sample, print_tree
from ex_2_3_tree_helper import Tree, find_leaf, dynamic_sampler
import numpy as np


"""
    The data is stored in Newick form:
        [A,B,[C,D]E]F;
        
           ________A 
          |
        F |________B
          |          ________C
          |         |
          |________E
                    |________D
                    
    But we are working in python. So we will work with lists that look
    the following way:
        
    Tree  = ['F', pF, ['A', pA, [], 'B', pB, [], 'E', pE, ['C', pC, [], 'D', pD, []]]]
        
    Each variable has a name (in string format), a list of categorical parameters,
    e.g. pF = [0.3, 0.2, 0.5], and a list of child nodes.
"""


"""
  Load the parameters of each tree and three samples from it.
  The data is stored in a dictionary. 
  
  The key indicates
      k := the number of categorical variables
      md:= maximum depth of the generated tree
      mb:= maximum branching factor of each node
      alpha:= the alpha values used to draw the parameters from a Dirichlet
  
  None of these numbers is important for your implementation but can be 
  used for interpretation.
"""



"""
Use pickle
"""

my_data_path = 'D:/KTH/Advanced Machine Learning/Assignment 2/AdvML2/MLadvHT18-master/2_3/'

with open(my_data_path + 'tree_params.pickle', 'rb') as handle:
    params = pickle.load(handle,  encoding='latin1')

with open(my_data_path + 'tree_samples.pickle', 'rb') as handle:
    samples = pickle.load(handle,  encoding='latin1')


"""
    Construct a tree with parameters from the loaded parameter dict.
"""


params_name = list(params.keys())[0]

params = params[list(params.keys())[0]]
root = load_params(params)


"""
    Load a matching sample into the tree.
"""
samples_name = params_name + '_sample_1'
sample = samples[samples_name]
load_sample(root, sample)


print('Root: \n')
print_tree(root, print_sample = True)
beta = find_leaf(root)


my_data_path = 'D:/KTH/Advanced Machine Learning/Assignment 2/AdvML2/MLadvHT18-master/2_5/'
with open(my_data_path + 'tree_with_CPD.pkl', 'rb') as handle:
    params2 = pickle.load(handle,  encoding='latin1')

with open(my_data_path + 'tree_with_leaf_samples.pkl', 'rb') as handle:
    sample2 = pickle.load(handle, encoding='latin1')

root2 = load_params(params2)

load_sample(root2, sample2)
print_tree(root2, print_sample = True)

beta = find_leaf(root2)

memo = []
memo = dynamic_sampler(root2, memo)

print('\nTree after sampling:\n')

print_tree(root2, print_sample = True)

prob = 1
for i in memo:
    prob *= i[2]

print('Tree Joint Probability: :' + str(prob))
