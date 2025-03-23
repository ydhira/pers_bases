import os, sys
import pickle
import numpy as np

from run_umap import read_words, load_embeddings_sentences, load_embeds, norman_75_mapping_idx2word
from analyse_kmeans import analyse_cluster_entropy, analyse_cluster_representative_mlp
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.stats import entropy
from collections import Counter

import torch 
from torch.utils.data import Dataset
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# Datset
class bert_dataset(Dataset):
    def __init__(self, in_embeds_dir):
        all_sentence_embeddings, true_word_ids = load_embeddings_sentences(in_embeds_dir)
        non_nan_idx = np.where(np.isnan(all_sentence_embeddings).any(axis=1)==False)[0]
        true_word_ids_expand = []
        for i in range(len(true_word_ids)):
            true_word_ids_expand.extend([i]*true_word_ids[i])
         
        self.all_sentence_embeddings = all_sentence_embeddings[non_nan_idx]
        self.true_word_ids_expand = np.array(true_word_ids_expand)[non_nan_idx]
  
    def __len__(self):
        return len(self.all_sentence_embeddings)

    def __getitem__(self, idx):
        embedding = self.all_sentence_embeddings[idx]
        word_label = self.true_word_ids_expand[idx]
        return embedding, word_label
# Model
class simple_mlp(nn.Module):
   def __init__(self, n_classes, llm_name):
        super(simple_mlp, self).__init__()
        if llm_name == 'bert':
            self.mlp = nn.Linear(768, n_classes) # bert 
        elif llm_name == 'llama':
            self.mlp = nn.Linear(4096, n_classes) # llama
        elif llm_name == "opt":
            self.mlp = nn.Linear(512, n_classes) # llama
        elif llm_name == "t5":
            self.mlp = nn.Linear(512, n_classes) # llama
   def forward(self, x):
        logits = self.mlp(x)
        return logits 

# training loop 
def training(dataloader, model, optimizer, weight):
    loss_vectors = []
    
    for i, data in enumerate(dataloader):
        # Every data instance is an input + label pair
        inputs, labels = data
        inputs = inputs.cuda()

        # Zero your gradients for every batch!
        optimizer.zero_grad()

        # Make predictions for this batch
        logits = model(inputs)

        # Compute the loss and its gradients
        
        # import pdb 
        # pdb.set_trace()

        loss1 = torch.mean(Categorical(probs=logits.softmax(-1)).entropy())
        loss2 = get_loss2(labels, logits)
        loss3 = Categorical(probs=torch.mean(logits.softmax(-1), dim=0)).entropy()
        # print(loss1, loss2, loss3) 

        #loss = loss_fn(outputs, labels)
        # descrease the entropy of the fist two losses. increase the entorpy of the third loss. 
        loss = weight[0] * loss1 + weight[1] * loss2 - weight[2] * loss3 
        # loss = 0.25 * loss1 + 0.25 * loss2 - loss3 

        loss.backward()

        # Adjust learning weights
        optimizer.step()

        # Gather data and report
        # running_loss += loss.item()
        loss_vectors.append([loss1.item(), loss2.item(), loss3.item()])
    
    return np.array(loss_vectors)

def get_loss2(labels, logits):
    unique_labels=torch.unique(labels)
    word_avg_softmax = []
    for i in range(len(unique_labels)): 
        word_avg_softmax.append(torch.mean(logits[torch.where(labels==unique_labels[i])].softmax(dim=-1), dim=0).unsqueeze(0))

    word_avg_softmax = torch.cat(word_avg_softmax, dim=0)
    loss2 = torch.mean(Categorical(probs=word_avg_softmax).entropy())
    return loss2

def predict(embeddings, model):
    #import pdb 
    #pdb.set_trace()
    embeddings = embeddings.cuda()
    logits = model(embeddings)
    clusters = torch.argmax(logits.softmax(dim=-1), dim=1)
    return clusters.cpu().numpy()

def main():
    torch.manual_seed(0)
    n_cluster = int(sys.argv[1] )
    weight_0 =  float(sys.argv[2] )
    weight_1 =  float(sys.argv[3] )
    weight_2 =  float(sys.argv[4] )
    llm_name = sys.argv[5]
    assert llm_name in ['bert', 'opt', 't5', 'llama']

    weight = [weight_0, weight_1, weight_2]

    BATCHSIZE = 10000
    EPOCHS = 10
    N_CLUSTERS = n_cluster #4
    if llm_name == 'bert':
        indir = 'bert_embeddings/norman_75_sentences_10000'
    elif llm_name == 'llama':
        indir = 'llama_embeddings/norman_75_sentences_10000'
    elif llm_name == "opt":
        indir = 'opt_embeddings/norman_75_sentences_10000'
    elif llm_name == "t5":
        indir = 't5_embeddings/norman_75_sentences_10000'

    dataset = bert_dataset(indir)
    dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
    model = simple_mlp(N_CLUSTERS, llm_name)
    if torch.cuda.is_available():
        model = model.cuda()
    # Optimizers specified in the torch.optim package
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    avg_entropies = []
    for epoch in range(EPOCHS):
        loss_vectors = training(dataset_loader, model, optimizer, weight)
        print(np.mean(loss_vectors[:,0]), np.mean(loss_vectors[:,1]), np.mean(loss_vectors[:,2]))
        np.savetxt(f"results/loss_epoch_{epoch}.csv", loss_vectors, delimiter=",")
        #import pdb 
        #pdb.set_trace()
        clusters = predict(torch.Tensor(dataset.all_sentence_embeddings), model)
        avg_entropy = analyse_cluster_entropy(dataset.all_sentence_embeddings, dataset.true_word_ids_expand, clusters, N_CLUSTERS)
        analyse_cluster_representative_mlp(dataset.all_sentence_embeddings, dataset.true_word_ids_expand, clusters, N_CLUSTERS)

        avg_entropies.append(avg_entropy)
    print(f'Ending .... Result for cluster: {N_CLUSTERS} for weight: ', weight)
    print(avg_entropies)
    print("**************")
main() 
        

