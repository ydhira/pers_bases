import os, sys
import pickle
import numpy as np 

from run_umap import read_words, load_embeddings_sentences, load_embeds, norman_75_mapping_idx2word
from sklearn.metrics import pairwise_distances_argmin_min
from scipy.stats import entropy
from collections import Counter
'''
Load a kmeans model. 
Load the word embeddings, and 10,000 sentence embeddings 
Cluster the 2 sets of embeddings 
Find the cluster representation for each k cluser
'''

def analyse_representative_cluster():
    in_embeds_dir = sys.argv[1] #'bert_embeddings/norman_75'
    words = read_words('../vocabs/norman_75.txt')
    kmeans_model_saved = sys.argv[2]#'kmeans_model_berr_norman75'
    n_clusters = int(sys.argv[3])
    embedding_model_name = sys.argv[4]
    embed_type = 'words'
    kmeans_model = pickle.load(open(kmeans_model_saved, 'rb'))

    if embed_type == 'words':
        all_embeddings, true_word_ids = load_embeds(in_embeds_dir, words)
        print('Loaded all embeddings: ', all_embeddings.shape)
        all_embeddings = all_embeddings[~np.isnan(all_embeddings).any(axis=1)]
        print('Removed nans. Embeddings: ', all_embeddings.shape)
    
    norman75_embeddings_cluster = kmeans_model.predict(all_embeddings) 
    closest, distances = pairwise_distances_argmin_min(all_embeddings, kmeans_model.cluster_centers_)
    #closest, _ = pairwise_distances_argmin_min(kmeans_model.cluster_centers_, all_embeddings)
    
    # norman75_embeddings_cluster and closest are the same vector (atleast for the n_clusters=2 case)
    
    # cluster representative word 
    for i in range(n_clusters):
        #cluster_i_points = np.where(closest == i)[0]
        #distances_i_points = distances[cluster_i_points]
        #np.argmin(distances_i_points)
        min_dist_i = 10000
        idx_min_dist = -1
        for j in range(len(closest)):
            if closest[j] == i: #belongs to the i th cluster 
                if distances[j] < min_dist_i: 
                    min_dist_i = distances[j]
                    idx_min_dist = j
        print(f'Closest representative word for cluster {i} is {norman_75_mapping_idx2word[idx_min_dist]}, with distance of {min_dist_i} ')
                    
    
    return 

'''
Load a kmeans model 
Load the sentences embeddings. multiple embeddings belong to the same word. 
Cluster the sentence embeddings. 
1. figure out for each of the 75 words, what percentage of the words are they mostly in. Then calculate a metric of agreement between the words. 
2. For each word, the cluster they are mostly in, that is the representative cluster. What is the representative word for that cluster. 
'''

def analyse_representative_cluster_sentences():
    in_embeds_dir = sys.argv[1] #'bert_embeddings/norman_75_sentences_subset_10000'
    kmeans_model_saved = sys.argv[2]#kmeans_model_bert_norman75_sentences
    n_clusters = int(sys.argv[3])#depending the the model given aboce.
    embedding_model_name = sys.argv[4]#bert
    
    embed_type = 'sentences'
    all_sentence_embeddings, true_word_ids = load_embeddings_sentences(in_embeds_dir)
    #import pdb
    #pdb.set_trace()
    #nan_idx = np.where(np.isnan(all_sentence_embeddings).any(axis=1)==True)[0]
    non_nan_idx = np.where(np.isnan(all_sentence_embeddings).any(axis=1)==False)[0]
    true_word_ids_expand = []
    for i in range(len(true_word_ids)):
        true_word_ids_expand.extend([i]*true_word_ids[i])

    print('Loaded all embeddings: ', all_sentence_embeddings.shape)
    all_sentence_embeddings = all_sentence_embeddings[non_nan_idx]
    true_word_ids_expand = np.array(true_word_ids_expand)[non_nan_idx]

    print('Removed nans. Embeddings: ', all_sentence_embeddings.shape)

    kmeans_model = pickle.load(open(kmeans_model_saved, 'rb'))
    
    norman75_embeddings_cluster = kmeans_model.predict(all_sentence_embeddings)
    closest, distances = pairwise_distances_argmin_min(all_sentence_embeddings, kmeans_model.cluster_centers_)

    word_ids = np.unique(true_word_ids_expand)

    print(word_ids)
    all_entropy = []

    for word_id in word_ids:
        assigned_cluster = norman75_embeddings_cluster[np.where(true_word_ids_expand == word_id)[0]]
        counter = Counter(assigned_cluster)
        counter_sort = counter.most_common()
        print(counter_sort)
        #counter_prob = [(i[0], i[1] / len(assigned_cluster)) for i in counter_sort]
        counter_perc = [(i[0], i[1] / len(assigned_cluster)*100) for i in counter_sort]
        print(counter.values())
        prob = [counter[i] if i in counter else 0 for i in range(n_clusters)]
        print(prob)
        ent = entropy(list(prob), base=2)
        print(f'For word {norman_75_mapping_idx2word[word_id]}, the cluster it is in are: \n {counter_perc} \n with entropy --> {ent}\n\n')
        all_entropy.append(ent)
         
    print(all_entropy)
    print(f'Avg entropy: {np.mean(all_entropy)}')
    
    return 

    # get the representativue word for each cluster, based on the cluster where most of its sentences belong 
    for i in range(n_clusters):
        #cluster_i_points = np.where(closest == i)[0]
        #distances_i_points = distances[cluster_i_points]
        #np.argmin(distances_i_points)
        min_dist_i = 10000
        idx_min_dist = -1
        for j in range(len(closest)):
            if closest[j] == i: #belongs to the i th cluster
                if distances[j] < min_dist_i:
                    min_dist_i = distances[j]
                    idx_min_dist = j #closest[j]
                    #print(min_dist_i, idx_min_dist)

        print(f'Closest representative word for cluster {i} is {norman_75_mapping_idx2word[true_word_ids_expand[idx_min_dist]]}, with distance of {min_dist_i} ') 

def analyse_cluster_representative_sentence_standalone(n_clusters, kmeans_model_saved, in_embeds_dir):
    # embeddings, labels, assigned_cluster, n_clusters

    all_sentence_embeddings, true_word_ids = load_embeddings_sentences(in_embeds_dir)
    non_nan_idx = np.where(np.isnan(all_sentence_embeddings).any(axis=1)==False)[0]
    true_word_ids_expand = []
    for i in range(len(true_word_ids)):
        true_word_ids_expand.extend([i]*true_word_ids[i])

    all_sentence_embeddings = all_sentence_embeddings[non_nan_idx]
    true_word_ids_expand = np.array(true_word_ids_expand)[non_nan_idx]

    kmeans_model = pickle.load(open(kmeans_model_saved, 'rb'))
    norman75_embeddings_cluster = kmeans_model.predict(all_sentence_embeddings) 
    closest, distances = pairwise_distances_argmin_min(all_sentence_embeddings, kmeans_model.cluster_centers_)
    
    # get the representativue word for each cluster, based on the cluster where most of its sentences belong 
    ### Closest Word to the cluster center 
    for i in range(n_clusters):
        #cluster_i_points = np.where(closest == i)[0]
        #distances_i_points = distances[cluster_i_points]
        #np.argmin(distances_i_points)
        min_dist_i = 10000
        idx_min_dist = -1
        for j in range(len(closest)):
            if closest[j] == i: #belongs to the i th cluster
                if distances[j] < min_dist_i:
                    min_dist_i = distances[j]
                    idx_min_dist = j #closest[j]
                    #print(min_dist_i, idx_min_dist)

        print(f'Closest representative word for cluster {i} is {norman_75_mapping_idx2word[true_word_ids_expand[idx_min_dist]]}, with distance of {min_dist_i} ') 

    print("*********")
    ### Entropy based cluster representative
    
    ### Most word present in a cluster normalized by its total frequency  
    total_frequency = Counter(true_word_ids_expand)
    cluster_frequency = {}
    for i in range(n_clusters):
        ## which words belong to this cluster 
        words_in_clusters = true_word_ids_expand[np.where(norman75_embeddings_cluster == i)[0]]
        words_freq = Counter(words_in_clusters) #0:12, 1: 13
        # words_freq_norm = {j: v/total_frequency[j] for j, v in words_freq.items()}
        words_freq_norm = {j: v/len(words_in_clusters) for j, v in words_freq.items()}

        words_freq_norm_high = Counter(words_freq_norm).most_common()[0]
        most_freq_word = norman_75_mapping_idx2word[words_freq_norm_high[0]]
        print(f'Closest representative word for cluster {i} is {most_freq_word}, with normalized frequency of {words_freq_norm_high[1]} ') 


    print(cluster_frequency)

    return 

def analyse_cluster_representative_mlp(embeddings, labels, assigned_cluster, n_clusters):
    cluster_centers = []
    for n in range(n_clusters):
        incluster = embeddings[np.where(assigned_cluster ==n)]
        if incluster.shape[0] > 0: # if this is 0 then np.mean gives nan and that causes issues in pairwise function 
            cluster_centers.append(np.mean(embeddings[np.where(assigned_cluster ==n)], axis=0))
        else: 
            random_center = [1000] * embeddings[0].shape[0] # randomly selecting a point 
            cluster_centers.append(random_center)

    closest, distances = pairwise_distances_argmin_min(embeddings, cluster_centers)
    
    ### Closest Word to the cluster center 
    for i in range(n_clusters):
        min_dist_i = 10000
        idx_min_dist = -1
        for j in range(len(closest)):
            if closest[j] == i: #belongs to the i th cluster
                if distances[j] < min_dist_i:
                    min_dist_i = distances[j]
                    idx_min_dist = j #closest[j]
                    #print(min_dist_i, idx_min_dist)

        print(f'Closest representative word for cluster {i} is {norman_75_mapping_idx2word[labels[idx_min_dist]]}, with distance of {min_dist_i} ') 

    print("*********")

    ### Most word present in a cluster normalized by its total frequency  
    total_frequency = Counter(labels)
    cluster_frequency = {}
    printout = []
    for i in range(n_clusters):
        ## which words belong to this cluster 
        words_in_clusters = labels[np.where(assigned_cluster == i)[0]]
        words_freq = Counter(words_in_clusters) #0:12, 1: 13
        words_freq_norm = {j: v/total_frequency[j] for j, v in words_freq.items()}
        words_freq_norm_high = Counter(words_freq_norm).most_common()[0] if len(Counter(words_freq_norm).most_common()) > 0 else (None , None)
        most_freq_word = norman_75_mapping_idx2word[words_freq_norm_high[0]] if words_freq_norm_high[0] else None
        if most_freq_word == 'imperceptivity':
            words_freq_norm_high = Counter(words_freq_norm).most_common()[1]
            most_freq_word = norman_75_mapping_idx2word[words_freq_norm_high[0]]
        print(f'Closest representative word for cluster {i} is {most_freq_word}, with normalized frequency of {words_freq_norm_high[1]} ') 
        printout.append(most_freq_word)
    print(printout)
    print(cluster_frequency)

    return 

def analyse_cluster_entropy(embeddings, labels, assigned_cluster, n_clusters):
    word_ids = np.unique(labels)

    print(word_ids)
    all_entropy = []

    for word_id in word_ids:
        assigned_cluster_i = assigned_cluster[np.where(labels == word_id)[0]]
        counter = Counter(assigned_cluster_i)
        counter_sort = counter.most_common()
        print(counter_sort)
        #counter_prob = [(i[0], i[1] / len(assigned_cluster)) for i in counter_sort]
        counter_perc = [(i[0], i[1] / len(assigned_cluster_i)*100) for i in counter_sort]
        prob = [counter[i] if i in counter else 0 for i in range(n_clusters)]
        ent = entropy(list(prob), base=2)
        print(ent)
        print(f'For word {norman_75_mapping_idx2word[word_id]}, the cluster it is in are: \n {counter_perc} \n with entropy --> {ent}\n\n')
        all_entropy.append(ent)

    print(all_entropy)
    print(f'Avg entropy: {np.mean(all_entropy)}')

    return np.mean(all_entropy)

if __name__ == "__main__":
    #analyse_representative_cluster()
    
    analyse_representative_cluster_sentences()
    in_embeds_dir = sys.argv[1] #'bert_embeddings/norman_75_sentences_subset_10000'
    kmeans_model_saved = sys.argv[2]#kmeans_model_bert_norman75_sentences
    n_cluster  = int(sys.argv[3])
    # n_cluster = int(sys.argv[1])
    analyse_cluster_representative_sentence_standalone(n_clusters=n_cluster, kmeans_model_saved=kmeans_model_saved, \
                                                       in_embeds_dir=in_embeds_dir)
