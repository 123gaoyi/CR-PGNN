from utils import sparse_to_adjlist
from scipy.io import loadmat
import numpy as np

"""
    Read IEEE power system data and save the adjacency matrices to adjacency lists
"""


if __name__ == "__main__":

    prefix = 'data/'

    # IEEE 14-bus system
    ieee14 = loadmat('data/IEEE14.mat')
    net_physical = ieee14['net_physical']      # physical transmission lines
    net_geographic = ieee14['net_geographic']  # geographic proximity (kNN)
    net_logical = ieee14['net_logical']        # logical control dependencies
    ieee14_homo = ieee14['homo']               # homogeneous graph

    sparse_to_adjlist(net_physical, prefix + 'ieee14_physical_adjlists.pickle')
    sparse_to_adjlist(net_geographic, prefix + 'ieee14_geographic_adjlists.pickle')
    sparse_to_adjlist(net_logical, prefix + 'ieee14_logical_adjlists.pickle')
    sparse_to_adjlist(ieee14_homo, prefix + 'ieee14_homo_adjlists.pickle')

    # IEEE 57-bus system
    ieee57 = loadmat('data/IEEE57.mat')
    net_physical = ieee57['net_physical']
    net_geographic = ieee57['net_geographic']
    net_logical = ieee57['net_logical']
    ieee57_homo = ieee57['homo']

    sparse_to_adjlist(net_physical, prefix + 'ieee57_physical_adjlists.pickle')
    sparse_to_adjlist(net_geographic, prefix + 'ieee57_geographic_adjlists.pickle')
    sparse_to_adjlist(net_logical, prefix + 'ieee57_logical_adjlists.pickle')
    sparse_to_adjlist(ieee57_homo, prefix + 'ieee57_homo_adjlists.pickle')

    # IEEE 118-bus system
    ieee118 = loadmat('data/IEEE118.mat')
    net_physical = ieee118['net_physical']
    net_geographic = ieee118['net_geographic']
    net_logical = ieee118['net_logical']
    ieee118_homo = ieee118['homo']

    sparse_to_adjlist(net_physical, prefix + 'ieee118_physical_adjlists.pickle')
    sparse_to_adjlist(net_geographic, prefix + 'ieee118_geographic_adjlists.pickle')
    sparse_to_adjlist(net_logical, prefix + 'ieee118_logical_adjlists.pickle')
    sparse_to_adjlist(ieee118_homo, prefix + 'ieee118_homo_adjlists.pickle')