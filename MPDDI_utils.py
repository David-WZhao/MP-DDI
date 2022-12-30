import argparse

import numpy as np
import dgl
import torch
from scipy import sparse
def load_data(network_path):
    drug_drug = np.loadtxt(network_path + 'adj_ddi.txt')
    protein_protein = np.loadtxt(network_path + 'adj_ppi.txt')
    drug_carrier_protein = np.loadtxt(network_path + 'adj_d_c.txt')
    drug_enzyme_protein = np.loadtxt(network_path + 'adj_d_e.txt')
    drug_target_protein = np.loadtxt(network_path + 'adj_d_a.txt')
    drug_transporter_protein = np.loadtxt(network_path + 'adj_d_t.txt')


    print('data loaded')

    return drug_drug,protein_protein,drug_carrier_protein,drug_enzyme_protein,drug_target_protein,drug_transporter_protein

def ConstructGraph(drug_drug,protein_protein,drug_carrier_protein,drug_enzyme_protein,drug_target_protein,drug_transporter_protein):

    protein_carrier_drug = drug_carrier_protein.T
    protein_enzyme_drug = drug_enzyme_protein.T
    protein_target_drug = drug_target_protein.T
    protein_transporter_drug = drug_transporter_protein.T

    d_d = dgl.graph(sparse.csr_matrix(drug_drug),ntype='drug', etype='dd')
    p_p = dgl.graph(sparse.csr_matrix(protein_protein),ntype='protein', etype='pp')

    d_c_p = dgl.bipartite(sparse.csr_matrix(drug_carrier_protein),'drug', 'dcp', 'protein')
    p_c_d = dgl.bipartite(sparse.csr_matrix(protein_carrier_drug),'protein', 'pcd', 'drug')
    d_e_p = dgl.bipartite(sparse.csr_matrix(drug_enzyme_protein),'drug', 'dep', 'protein')
    p_e_d = dgl.bipartite(sparse.csr_matrix(protein_enzyme_drug),'protein', 'ped', 'drug')
    d_a_p = dgl.bipartite(sparse.csr_matrix(drug_target_protein),'drug', 'dap', 'protein')
    p_a_d = dgl.bipartite(sparse.csr_matrix(protein_target_drug),'protein', 'pad', 'drug')
    d_t_p = dgl.bipartite(sparse.csr_matrix(drug_transporter_protein),'drug', 'dtp', 'protein')
    p_t_d = dgl.bipartite(sparse.csr_matrix(protein_transporter_drug),'protein', 'ptd', 'drug')

    graph = dgl.hetero_from_relations([d_d,p_p,d_c_p,p_c_d,d_e_p,p_e_d,d_a_p,p_a_d,d_t_p,p_t_d])

    return graph

def parse_args():
    parser = argparse.ArgumentParser(description='MGNN')

    parser.add_argument("--epochs", type=int, default=1000,
                        help="number of training epochs")
    parser.add_argument("--dim-embedding", type=int, default=128,
                        help="dimension of embeddings")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="learning rate")
    parser.add_argument('--weight-decay', type=float, default=0,
                        help="weight decay")
    parser.add_argument('--patience', type=float, default=50,
                        help="Early stopping patience")
    parser.add_argument('--layer', type=float, default=3,
                        help="layers L")
    return parser.parse_args()

def meta_path_matrixs(g,meta_paths):
    meta_path_matrixs=[]
    for meta_path in meta_paths:
        new_g=dgl.metapath_reachable_graph(g,meta_path)
        meta_path_matrixs.append(new_g.adjacency_matrix().to_dense().numpy().T)
    return meta_path_matrixs
def to_ones(array):

    j = i=0
    while i < len(array):
        j=0
        while j<len(array[i]):
            if array[i][j]!=0:
                array[i][j]=1
            j+=1
        i+=1

    return array

def row_normalize(t):
    t = t.float()
    row_sums = t.sum(1) + 1e-12
    output = t / row_sums[:, None]
    output[torch.isnan(output) | torch.isinf(output)] = 0.0
    return output


def sigmod2(x):
    return 1. / (1 + np.exp(-x))



