import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
from MPDDI_utils import *
from sklearn.metrics import roc_auc_score,roc_curve
from sklearn.metrics import average_precision_score,precision_recall_curve

import time
import xlwt
import torch.optim as optim
from sklearn.model_selection import train_test_split,StratifiedKFold

device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu' )

t1=time.time()
class MGNN(nn.Module):
    def __init__(self,d,p,dim,layer,dropout):
        super(MGNN, self).__init__()

        self.layer=layer
        self.dim_embedding=dim

        self.fc_drug_drug = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_protein_protein = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_drug_protein = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_protein_drug = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_drug_drug_1 = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_protein_protein_1 = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_drug_protein_1 = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_protein_drug_1 = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_drug_drug_0 = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_protein_protein_0 = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_drug_protein_0 = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_protein_drug_0 = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_drug = nn.Linear(self.dim_embedding, self.dim_embedding).float()
        self.fc_protein = nn.Linear(self.dim_embedding, self.dim_embedding).float()

        self.project_drug_feature=nn.Linear(d,self.dim_embedding)
        self.project_protein_feature=nn.Linear(p,self.dim_embedding)

        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        self.dropout = dropout

        self.attention = Attention(in_size=dim)
    def forward(self,drug_feat,protein_feat,drug_agg_matrixs,protein_agg_matrixs,drug_drug,mask):

        drug_feat=self.project_drug_feature(drug_feat)
        protein_feat=self.project_protein_feature(protein_feat)

        for i in range (self.layer):

            drug_features = []
            drug_features0 = []
            drug_features1 = []
            drug_features2 = []

            drug_features1.append(torch.mm((row_normalize(torch.from_numpy(drug_agg_matrixs[0])).float()).to(device),self.fc_drug_drug(drug_feat)))
            drug_features1.append(torch.mm((row_normalize(torch.from_numpy(drug_agg_matrixs[1])).float()).to(device),self.fc_drug_drug(drug_feat)))
            drug_features1.append(torch.mm((row_normalize(torch.from_numpy(drug_agg_matrixs[2])).float()).to(device),self.fc_drug_drug(drug_feat)))
            drug_features1.append(torch.mm((row_normalize(torch.from_numpy(drug_agg_matrixs[3])).float()).to(device),self.fc_drug_drug(drug_feat)))
            drug_features1.append(torch.mm((row_normalize(torch.from_numpy(drug_agg_matrixs[4])).float()).to(device),self.fc_drug_drug(drug_feat)))
            drug_features1.append(torch.mm((row_normalize(torch.from_numpy(drug_agg_matrixs[5])).float()).to(device),self.fc_protein_drug(protein_feat)))
            drug_features1.append(torch.mm((row_normalize(torch.from_numpy(drug_agg_matrixs[6])).float()).to(device),self.fc_protein_drug(protein_feat)))
            drug_features1.append(torch.mm((row_normalize(torch.from_numpy(drug_agg_matrixs[7])).float()).to(device),self.fc_protein_drug(protein_feat)))
            drug_features1.append(torch.mm((row_normalize(torch.from_numpy(drug_agg_matrixs[8])).float()).to(device),self.fc_protein_drug(protein_feat)))
            drug_features1=F.relu(torch.sum(torch.stack([drug_features1[0],drug_features1[1],drug_features1[2],drug_features1[3],drug_features1[4],drug_features1[5], drug_features1[6], drug_features1[7],drug_features1[8]],dim=1),dim=1))

            drug_features2.append(torch.mm((row_normalize(torch.from_numpy(drug_agg_matrixs[9])).float()).to(device),self.fc_drug_drug_1(drug_feat)))
            drug_features2.append(torch.mm((row_normalize(torch.from_numpy(drug_agg_matrixs[10])).float()).to(device),self.fc_drug_drug_1(drug_feat)))
            drug_features2.append(torch.mm((row_normalize(torch.from_numpy(drug_agg_matrixs[11])).float()).to(device),self.fc_drug_drug_1(drug_feat)))
            drug_features2.append(torch.mm((row_normalize(torch.from_numpy(drug_agg_matrixs[12])).float()).to(device),self.fc_protein_drug_1(protein_feat)))
            drug_features2.append(torch.mm((row_normalize(torch.from_numpy(drug_agg_matrixs[13])).float()).to(device),self.fc_protein_drug_1(protein_feat)))
            drug_features2=F.relu(torch.sum(torch.stack([drug_features2[0], drug_features2[1], drug_features2[2],drug_features2[3], drug_features2[4]], dim=1), dim=1))

            drug_features0.append(torch.mm((row_normalize(drug_agg_matrixs[14]).float()).to(device),self.fc_drug_drug_0(drug_feat)))
            drug_features0.append(torch.mm((row_normalize(torch.from_numpy(drug_agg_matrixs[15])).float()).to(device),self.fc_protein_drug_0(protein_feat)))
            drug_features0.append(torch.mm((row_normalize(torch.from_numpy(drug_agg_matrixs[16])).float()).to(device),self.fc_protein_drug_0(protein_feat)))
            drug_features0.append(torch.mm((row_normalize(torch.from_numpy(drug_agg_matrixs[17])).float()).to(device),self.fc_protein_drug_0(protein_feat)))
            drug_features0.append(torch.mm((row_normalize(torch.from_numpy(drug_agg_matrixs[18])).float()).to(device),self.fc_protein_drug_0(protein_feat)))
            drug_features0=F.relu(torch.sum(torch.stack([drug_features0[0],drug_features0[1], drug_features0[2], drug_features0[3], drug_features0[4]], dim=1), dim=1))

            drug_features = torch.stack([drug_features0,drug_features1, drug_features2], dim=1)

            protein_features = []
            protein_features1 = []
            protein_features2 = []
            protein_features0 = []

            protein_features1.append(torch.mm((row_normalize(torch.from_numpy(protein_agg_matrixs[0])).float()).to(device),self.fc_drug_protein(drug_feat)))
            protein_features1.append(torch.mm((row_normalize(torch.from_numpy(protein_agg_matrixs[1])).float()).to(device),self.fc_drug_protein(drug_feat)))
            protein_features1.append(torch.mm((row_normalize(torch.from_numpy(protein_agg_matrixs[2])).float()).to(device),self.fc_drug_protein(drug_feat)))
            protein_features1.append(torch.mm((row_normalize(torch.from_numpy(protein_agg_matrixs[3])).float()).to(device),self.fc_drug_protein(drug_feat)))
            protein_features1.append(torch.mm((row_normalize(torch.from_numpy(protein_agg_matrixs[4])).float()).to(device),self.fc_protein_protein(protein_feat)))
            protein_features1.append(torch.mm((row_normalize(torch.from_numpy(protein_agg_matrixs[5])).float()).to(device),self.fc_protein_protein(protein_feat)))
            protein_features1.append(torch.mm((row_normalize(torch.from_numpy(protein_agg_matrixs[6])).float()).to(device),self.fc_protein_protein(protein_feat)))
            protein_features1.append(torch.mm((row_normalize(torch.from_numpy(protein_agg_matrixs[7])).float()).to(device),self.fc_protein_protein(protein_feat)))
            protein_features1.append(torch.mm((row_normalize(torch.from_numpy(protein_agg_matrixs[8])).float()).to(device),self.fc_protein_protein(protein_feat)))
            protein_features1=F.relu(torch.sum(torch.stack([protein_features1[0], protein_features1[1], protein_features1[2],protein_features1[3],protein_features1[4],protein_features1[5],protein_features1[6],protein_features1[7],protein_features1[8]], dim=1),dim=1))

            protein_features2.append(torch.mm((row_normalize(torch.from_numpy(protein_agg_matrixs[9])).float()).to(device),self.fc_drug_protein_1(drug_feat)))
            protein_features2.append(torch.mm((row_normalize(torch.from_numpy(protein_agg_matrixs[10])).float()).to(device),self.fc_drug_protein_1(drug_feat)))
            protein_features2.append(torch.mm((row_normalize(torch.from_numpy(protein_agg_matrixs[11])).float()).to(device),self.fc_protein_protein_1(protein_feat)))
            protein_features2.append(torch.mm((row_normalize(torch.from_numpy(protein_agg_matrixs[12])).float()).to(device),self.fc_protein_protein_1(protein_feat)))
            protein_features2.append(torch.mm((row_normalize(torch.from_numpy(protein_agg_matrixs[13])).float()).to(device),self.fc_protein_protein_1(protein_feat)))
            protein_features2=F.relu(torch.sum(torch.stack([protein_features2[0], protein_features2[1],protein_features2[2], protein_features2[3], protein_features2[4]], dim=1), dim=1))

            protein_features0.append(torch.mm((row_normalize(torch.from_numpy(protein_agg_matrixs[14])).float()).to(device),self.fc_protein_protein_0(protein_feat)))
            protein_features0.append(torch.mm((row_normalize(torch.from_numpy(protein_agg_matrixs[15])).float()).to(device),self.fc_drug_protein_0(drug_feat)))
            protein_features0.append(torch.mm((row_normalize(torch.from_numpy(protein_agg_matrixs[16])).float()).to(device),self.fc_drug_protein_0(drug_feat)))
            protein_features0.append(torch.mm((row_normalize(torch.from_numpy(protein_agg_matrixs[17])).float()).to(device),self.fc_drug_protein_0(drug_feat)))
            protein_features0.append(torch.mm((row_normalize(torch.from_numpy(protein_agg_matrixs[18])).float()).to(device),self.fc_drug_protein_0(drug_feat)))
            protein_features0=F.relu(torch.sum(torch.stack([protein_features0[0],protein_features0[1], protein_features0[2], protein_features0[3], protein_features0[4]], dim=1), dim=1))

            protein_features = torch.stack([protein_features0, protein_features1, protein_features2], dim=1)

            drug_feat_1,drug_alpha=self.attention(drug_features)
            drug_feat=F.relu(drug_feat + drug_feat_1)

            protein_feat_1,protein_alpha = self.attention(protein_features)
            protein_feat = F.relu(protein_feat + protein_feat_1)

        predict=self.sigmoid(torch.mm(drug_feat,torch.transpose(drug_feat,0,1)))

        tmp = torch.mul(mask.float(), (predict - drug_drug.float()))

        loss = torch.sum(tmp ** 2)
        return predict,loss,drug_alpha,protein_alpha

class Attention(nn.Module):
    def __init__(self, in_size,hidden_size=128):
        super(Attention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        alpha = torch.softmax(w, dim=1)
        return (alpha * z).sum(1),alpha

def train_and_evaluate(DSAtrain, DSAvalid, DSAtest,drug_feat,protein_feat,args,k):
    drug_drug = torch.zeros((2410, 2410))
    mask = torch.zeros((2410, 2410)).to(device)

    for ele in DSAtrain:
        drug_drug[ele[0], ele[1]] = ele[2]
        mask[ele[0], ele[1]] = 1

    best_valid_aupr = 0.
    best_valid_auc = 0
    test_aupr = 0.
    test_auc = 0.
    patience = 0.

    g = ConstructGraph(drug_drug, protein_protein,drug_carrier_protein, drug_enzyme_protein, drug_target_protein,drug_transporter_protein)

    all_meta_paths = [[('dd', 'dd'), ('dap', 'pad'),('dcp', 'pcd'),('dep', 'ped'),('dtp', 'ptd'),('dd','dap'),('dd','dcp'),('dd','dep'),('dd','dtp'),('dd', 'dd', 'dd'),('dd', 'dap', 'pad'),('dd', 'dep', 'ped'),('dd', 'dd', 'dap'),('dd', 'dd', 'dep')],
                      [('pad', 'dd'),('pcd', 'dd'),('ped', 'dd'),('ptd', 'dd'), ('pp', 'pp'), ('pad', 'dap'),('pcd', 'dcp'),('ped', 'dep'),('ptd', 'dtp'), ('pp', 'pp', 'pad'),('pp', 'pp', 'ped'), ('pp', 'pp', 'pp'), ('pp', 'pad', 'dap'),('pp', 'ped', 'dep')]]

    drug_agg_matrixs = meta_path_matrixs(g, all_meta_paths[0])
    protein_agg_matrixs = meta_path_matrixs(g, all_meta_paths[1])

    drug_direct_link=[drug_drug,drug_carrier_protein,drug_enzyme_protein,drug_target_protein,drug_transporter_protein]
    protein_direct_link=[protein_protein,drug_carrier_protein.T,drug_enzyme_protein.T,drug_target_protein.T,drug_transporter_protein.T]

    drug_agg_matrixs.extend(drug_direct_link)
    protein_agg_matrixs.extend(protein_direct_link)

    drug_drug=drug_drug.to(device)

    model=MGNN(2410,1127,dim=args.dim_embedding,layer=args.layer,dropout=0)
    model=model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)

    for epoch in range(args.epochs):

        t1=time.time()
        model.train()
        predict,loss,drug_alpha,protein_alpha=model(drug_feat,protein_feat,drug_agg_matrixs,protein_agg_matrixs,drug_drug,mask)
        results = predict.detach().cpu()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred_list = []
            ground_truth = []

            for ele in DSAvalid:
                pred_list.append(results[ele[0], ele[1]])
                ground_truth.append(ele[2])

            valid_auc = roc_auc_score(ground_truth, pred_list)
            valid_aupr = average_precision_score(ground_truth, pred_list)

            if valid_aupr >= best_valid_aupr:
                best_valid_aupr = valid_aupr
            if valid_auc >= best_valid_auc:
                best_valid_auc = valid_auc

                patience = 0
                pred_list = []
                ground_truth = []

                for ele in DSAtest:
                    pred_list.append(results[ele[0], ele[1]])
                    ground_truth.append(ele[2])

                test_auc = roc_auc_score(ground_truth, pred_list)
                test_aupr = average_precision_score(ground_truth, pred_list)
            else:
                patience += 1
                if patience > args.patience:
                    print("Early Stopping")
                    break
            print('Valid auc & aupr:', valid_auc, valid_aupr, ";  ", 'Test auc & aupr:', test_auc, test_aupr)

        t2=time.time()
        print(f'epoc: {epoch+1} loss:{loss.item()} time consum:{t2-t1} s')
    return best_valid_auc,best_valid_aupr



def main(args):
    best_auc=[]
    best_aupr=[]
    drug_feat = torch.eye(2410,2410,dtype=torch.float)
    protein_feat = torch.eye(1127,1127,dtype=torch.float)
    drug_feat=drug_feat.to(device)
    protein_feat=protein_feat.to(device)

    whole_positive_index=[]
    whole_negative_index=[]
    for i in range(np.shape(drug_drug_original)[0]):
        for j in range(np.shape(drug_drug_original)[1]):
            if int(drug_drug_original[i][j])==1:
                whole_positive_index.append([i,j])
            elif int(drug_drug_original[i][j])==0:
                whole_negative_index.append([i,j])
    negative_sample_index=np.random.choice(np.arange(len(whole_negative_index)),size=len(whole_positive_index),replace=False)
    data_set = np.zeros((len(negative_sample_index)+len(whole_positive_index),3),dtype=int)
    count = 0
    for i in whole_positive_index:
        data_set[count][0] = i[0]
        data_set[count][1] = i[1]
        data_set[count][2] = 1
        count += 1
    for i in negative_sample_index:
        data_set[count][0] = whole_negative_index[i][0]
        data_set[count][1] = whole_negative_index[i][1]
        data_set[count][2] = 0
        count += 1
    auc=0
    fold=1
    kf = StratifiedKFold(n_splits=10, random_state=None, shuffle=True)
    for train_index, test_index in kf.split(data_set[:,:2],data_set[:,2]):
        train, DSAtest = data_set[train_index], data_set[test_index]
        DSAtrain, DSAvalid = train_test_split(train, test_size=0.05, random_state=None)

        print ("#############%d fold"%fold+"#############")

        best_valid_auc,best_valid_aupr=train_and_evaluate(DSAtrain, DSAvalid, DSAtest,drug_feat,protein_feat,args,fold)
        fold = fold + 1
        best_auc.append(best_valid_auc)
        best_aupr.append(best_valid_aupr)
        if auc < best_valid_auc:
            auc = best_valid_auc

    print('ave_aupr', format((best_aupr[0] + best_aupr[1] + best_aupr[2] + best_aupr[3] + best_aupr[4] + best_aupr[5] + best_aupr[6] + best_aupr[7] + best_aupr[8] + best_aupr[9])/10,'.4f'))
    print('ave_auc', format((best_auc[0] + best_auc[1] + best_auc[2] + best_auc[3] + best_auc[4] + best_auc[5] +best_auc[6] + best_auc[7] + best_auc[8] + best_auc[9]) / 10, '.4f'))

if __name__ == "__main__":

    Patience=20
    drug_drug_original,protein_protein,drug_carrier_protein,drug_enzyme_protein,drug_target_protein,drug_transporter_protein=load_data('dataset/')
    args = parse_args()
    print(args)
    main(args)
