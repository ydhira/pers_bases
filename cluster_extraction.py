from transformers import AutoTokenizer, BertModel
from transformers import LlamaModel, LlamaForCausalLM, LlamaTokenizer
from transformers import OPTModel
from transformers import AutoModelForSeq2SeqLM
import torch
import numpy as np 
from sklearn.cluster import BisectingKMeans, KMeans
from tqdm import tqdm  
import pickle
import os, sys 

norman_75_mapping_word2idx = {'spirit': 0, 'talkativeness': 1, 'sociability': 2, 'spontaneity': 3, 'boisterousness': 4, 'adventure': 5, 'energy': 6, 'conceit': 7, 'vanity': 8, 'indiscretion': 9, 'sensuality': 10, 'lethargy': 11, 'aloofness': 12, 'silence': 13, 'modesty': 14, 'pessimism': 15, 'unfriendliness': 16, 'trust': 17, 'amiability': 18, 'generosity': 19, 'agreeableness': 20, 'tolerance': 21, 'courtesy': 22, 'altruism': 23, 'warmth': 24, 'honesty': 25, 'vindictiveness': 26, 'ill_humor': 27, 'criticism': 28, 'disdain': 29, 'antagonism': 30, 'aggressiveness': 31, 'dogmatism': 32, 'temper': 33, 'distrust': 34, 'greed': 35, 'dishonesty': 36, 'industry': 37, 'order': 38, 'self-discipline': 39, 'evangelism': 40, 'consistency': 41, 'grace': 42, 'reliability': 43, 'sophistication': 44, 'formality': 45, 'foresight': 46, 'religiosity': 47, 'maturity': 48, 'passionlessness': 49, 'thrift': 50, 'negligence': 51, 'inconsistency': 52, 'rebelliousness': 53, 'irreverence': 54, 'provinciality': 55, 'intemperance': 56, 'durability': 57, 'poise': 58, 'self-reliance': 59, 'callousness': 60, 'candor': 61, 'self_pity': 62, 'anxiety': 63, 'insecurity': 64, 'timidity': 65, 'passivity': 66, 'immaturity': 67, 'wisdom': 68, 'originality': 69, 'objectivity': 70, 'knowledge': 71, 'reflection': 72, 'art': 73, 'imperceptivity': 74}
        
def read_words(filename):
    with open(filename, 'r') as fopen:
        words = fopen.readlines()
    for i, word in enumerate(words):
        words[i] = word.lower().replace('\n','').split(',')[0]
    return words

def read_sentences(filename):
    _dict = pickle.load(open(filename, 'rb'))
    sentences = _dict[list(_dict.keys())[0]]
    return sentences 

def get_bert_embeddings(words, personality_word, tokenizer, model):
    #tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    #model = BertModel.from_pretrained("bert-base-uncased")

    embeddings = []
    for word in tqdm(words):
        #print(word)
        #if len(word.split()) > 512: continue # the model has a problem with processing longer than 512 sequences. so 
        inputs = tokenizer(word, add_special_tokens=False, return_tensors="pt")
        a1 = inputs['input_ids']
        if a1.shape[1] > 512: 
            print(a1.shape[1])
            continue # the model has a problem with processing longer than 512 sequences. so
        
        a2 = tokenizer(personality_word, add_special_tokens=False)['input_ids']
        needed_idx = np.nonzero(np.in1d(a1, a2))
        outputs = model(**inputs)
        last_hidden_states = outputs.last_hidden_state.detach().cpu().numpy()
        last_hidden_states = last_hidden_states[:,needed_idx[0], :]
        #mean_states = np.mean(last_hidden_states[:, 1:-1], axis=1) # first and last are start and end tokens so we can remove those 
        mean_states = np.mean(last_hidden_states, axis=1)
        
        embeddings.append(mean_states)
    return np.array(embeddings)

def get_llama2_embeddings(words, personality_word, tokenizer, model):

    embeddings = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for word in tqdm(words):
        inputs = tokenizer(word, return_tensors="pt")
        a1 = inputs['input_ids']
        inputs.to(device)
        if inputs['input_ids'].shape[1] > 400: 
            continue 

        a2 = tokenizer(personality_word, return_tensors="pt")['input_ids']
        needed_idx = np.nonzero(np.in1d(a1, a2))

        #print('len of tokens', inputs['input_ids'].shape)
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden_states = outputs.last_hidden_state.detach().cpu().numpy()
        last_hidden_states = last_hidden_states[:,needed_idx[0], :]
        #print(last_hidden_states.shape)
        mean_states = np.mean(last_hidden_states, axis=1)
        embeddings.append(mean_states)
        del outputs
        del inputs 
    return np.array(embeddings)

def get_llama_embeddings(words):
    model = LlamaForCausalLM.from_pretrained("/ocean/projects/cis220031p/hyd")
    tokenizer = AutoTokenizer.from_pretrained("/ocean/projects/cis220031p/hyd")

    embeddings = []
    for word in words:
        inputs = tokenizer(word, return_tensors="pt")
        outputs = model(**inputs, output_hidden_states=True)

        last_hidden_states = outputs.last_hidden_state
        embeddings.append(last_hidden_states)
    return embeddings

def get_opt_embeddings(words, personality_word, tokenizer, model):
    
    embeddings = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for word in words:
        inputs = tokenizer(word, return_tensors="pt")
        a1 = inputs['input_ids']
        inputs.to(device)
        if inputs['input_ids'].shape[1] > 400: 
            continue 
        a2 = tokenizer(personality_word, return_tensors="pt")['input_ids']
        needed_idx = np.nonzero(np.in1d(a1, a2))

        outputs = model(**inputs, output_hidden_states=True)
        last_hidden_states = outputs.last_hidden_state.detach().cpu().numpy()
        last_hidden_states = last_hidden_states[:,needed_idx[0], :]
        # mean_states = np.mean(last_hidden_states[:, 1:-1], axis=1) # first and last are start and end tokens so we can remove those 
        mean_states = np.mean(last_hidden_states, axis=1)
        embeddings.append(mean_states)
        del outputs
        del inputs 
    return np.array(embeddings)

def get_flant5_embeddings(words, personality_word, tokenizer, model):

    embeddings = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for word in words:
        inputs = tokenizer(word, return_tensors="pt")
        inputs.to(device)
        if inputs['input_ids'].shape[1] > 400: 
            continue 
        outputs = model(input_ids=inputs['input_ids'],decoder_input_ids=inputs['input_ids'])
        last_hidden_states = outputs.encoder_last_hidden_state.detach().cpu().numpy()
        mean_states = np.mean(last_hidden_states[:, :-1], axis=1) # last are  end tokens so we can remove those 
        embeddings.append(mean_states)
        del outputs
        del inputs 
    return np.array(embeddings)

def load_embeddings(dirname):
    files = os.listdir(dirname)
    files = list(map(lambda x: os.path.join(dirname, x), files))
    embeddings = []
    for f in files:
        le_i = np.load(f)
        embeddings.append(le_i.squeeze())
    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings

def load_embeddings_sentences(dirname):
    # dirname = bert_embeddings/norman_75_sentencesn
    # warning, this might take a hell lot of space. 
    embeddings = []
    for word, idx in norman_75_mapping_word2idx.items():
        sent_embeddings_files = os.listdir(os.path.join(dirname, word))
        sent_embeddings = []
        for f in sent_embeddings_files:
            f = os.path.join(dirname, word, f)
            le_i = np.load(f)
            try:
                le_i = le_i.squeeze(axis=1)
            except Exception as e:
                le_i = le_i
            assert len(le_i.shape) == 2
            sent_embeddings.append(le_i)
        sent_embeddings = np.concatenate(sent_embeddings, axis=0)
        print(idx, word, sent_embeddings.shape)
        embeddings.append(sent_embeddings)
    all_embeddings = np.concatenate(embeddings, axis=0) 
    return all_embeddings


def run_kmeans(n_clusters, X):
    print('Training kmeans model...')
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(X)
    inertia = kmeans.inertia_
    n_iter = kmeans.n_iter_
    print(f'n_cluster, inertia, n_iter: {n_clusters, inertia, n_iter}')
    return kmeans 

def run_bisecting_kmeans(n_clusters, X):
    print('Training kmeans model...')
    kmeans = BisectingKMeans(n_clusters=n_clusters, random_state=0).fit(X)
    inertia = kmeans.inertia_
    n_iter = kmeans.n_iter_
    print(f'n_cluster, inertia, n_iter: {n_clusters, inertia, n_iter}')
    return kmeans 

def main_extracting_embeddings_words():
    input_file = sys.argv[1] # vocab 
    embedding_model_name = sys.argv[2]
    out_dir = sys.argv[3]
    start = int(sys.argv[4])
    words = read_words(input_file)
    personality_word = None 

    print('input file: ', input_file)
    print('embedding_model_name: ', embedding_model_name)
    print('out_dir: ', out_dir)
    print("Total len of words: ", len(words))

    if embedding_model_name == 'opt':
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
        model = OPTModel.from_pretrained("facebook/opt-350m")
        get_embed_fn = get_opt_embeddings
    elif embedding_model_name == 'bert':
        print('loading the model')
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")
        print('Model loaded')
        get_embed_fn = get_bert_embeddings
    elif embedding_model_name == 'llama':
        device_map = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = LlamaTokenizer.from_pretrained("/ocean/projects/cis220031p/hyd/llama/llama_output_dir",  add_bos_token =False, add_eos_token =False, clean_up_tokenization_spaces =True, device_map=device_map )
        model = LlamaModel.from_pretrained("/ocean/projects/cis220031p/hyd/llama/llama_output_dir", device_map=device_map)
        get_embed_fn = get_llama2_embeddings
 
    jump = 100 # 100
    #start = 0 #16500 #8400 #1300
    end = start+jump
    while(start < len(words)):
        subset = words[start:end]
        # print(start, end)
        # bert_embeddings = get_bert_embeddings(subset, tokenizer, model)
        # llama_embeddings = get_llama2_embeddings(subset, tokenizer, model)
        # opt_embeddings = get_opt_embeddings(subset, tokenizer, model)
        embeddings = get_embed_fn(subset, personality_word, tokenizer, model)
        print(f'{embedding_model_name}. Len of embeddings {len(embeddings)}. embeddings shape: {embeddings.shape}')
        output_file = embedding_model_name + '_embeddings_' + str(end)+'.npy'
        #os.makedirs(output_file, exists_ok=True)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        np.save(out_dir + output_file, embeddings)
        start=end
        end+=jump
    return 

def save_10000_gutenberg_sentences():
    input_dir = sys.argv[1] # dir containing all pickle files #gutenberg_sentences/
    input_files_words = os.listdir(input_dir) # pickle files 
    input_files = list(map(lambda x: os.path.join(input_dir, x), input_files_words))
    
    for i, filei in enumerate(input_files):
        sentences = read_sentences(filei)
        idx = np.arange(len(sentences))
        np.random.shuffle(idx)
        print(len(sentences))
        sentences_small = []
        for j in idx[:10000]:
            sentences_small.append(sentences[j])
        #sentences_small = np.array(sentences)[idx]
        print(len(sentences_small))
        word = input_files_words[i]
        res_dict={}
        res_dict[word.split('.')[0]]=sentences_small
        with open(f'gutenberg_sentences_10000/{word}', 'wb') as handle:
                pickle.dump(res_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
        #break 
    return 

def main_extracting_embeddings_sentences():
    input_file = sys.argv[1] 
    personality_word = sys.argv[2]
    embedding_model_name = sys.argv[3] # bert, llama, opt
    out_dir = sys.argv[4] #'bert_embeddings/norman_75_sentences/'
    #start = int(sys.argv[5])
    
    # reading sentences from pkl files 
    sentences = read_sentences(input_file)
    words = sentences #[:2] 
    print('input file: ', input_file)
    print('personality_word: ', personality_word)
    print('embedding_model_name: ', embedding_model_name)
    print('out_dir: ', out_dir)
    print("Total len of sentences: ", len(words))
    # try:
    #     last_done = sorted(list(map(lambda x: int(x.split('_')[-1].split('.')[0]), os.listdir(out_dir))))[-1]

    #     if last_done >= len(words): 
    #         print(last_done, last_done >= len(words))
    #         print('Already completed')
    #         print('Exiting ...')
    # except Exception as e:
    last_done = 0
    if embedding_model_name == 'opt':
        device_map = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m", device_map=device_map )
        model = OPTModel.from_pretrained("facebook/opt-350m", device_map=device_map )
        get_embed_fn = get_opt_embeddings
    if embedding_model_name == 't5':
        device_map = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small", device_map=device_map )
        tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small", device_map=device_map )
        get_embed_fn = get_flant5_embeddings
    elif embedding_model_name == 'bert':
        print('loading the model')
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased")
        print('Model loaded')
        get_embed_fn = get_bert_embeddings
    elif embedding_model_name == 'llama':
        device_map = 'cuda' if torch.cuda.is_available() else 'cpu'
        tokenizer = LlamaTokenizer.from_pretrained("/ocean/projects/cis220031p/hyd/llama/llama_output_dir",  add_bos_token =False, add_eos_token =False, clean_up_tokenization_spaces =True, device_map=device_map )
        model = LlamaModel.from_pretrained("/ocean/projects/cis220031p/hyd/llama/llama_output_dir", device_map=device_map)
        get_embed_fn = get_llama2_embeddings
    
    print('Embedding Model Loaded')
    
    jump = 100
    start = last_done
    #start = 0 #16500 #8400 #1300
    end = start+jump
    while(start < len(words)):
        subset = words[start:end]
        embeddings = get_embed_fn(subset, personality_word, tokenizer, model)
        
        print(f'{embedding_model_name}. Len of embeddings {len(embeddings)}. embeddings shape: {embeddings.shape}')
        output_file = embedding_model_name + '_embeddings_' + str(end)+'.npy'
        print(out_dir + output_file)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        np.save(out_dir + output_file, embeddings)
        start=end
        end+=jump
        del embeddings
    
def main_clustering_embeddings():
    in_dir = sys.argv[1]
    kmeans_save_dir = sys.argv[2]
    n_clusters = int(sys.argv[3])
    unit_type = 'sentences'

    if unit_type == 'sentences':
        embeddings = load_embeddings_sentences(in_dir)
    elif unit_type == 'words':
        embeddings = load_embeddings(in_dir)
    print('Loaded all embeddings: ', embeddings.shape)
    embeddings = embeddings[~np.isnan(embeddings).any(axis=1)]
    print('Removed nans. Embeddings: ', embeddings.shape)
    model = run_kmeans(n_clusters, embeddings)
    # model = run_bisecting_kmeans(n_clusters, embeddings)
    pickle.dump(model, open(kmeans_save_dir+str(n_clusters)+'.model', 'wb'))
    print('saved model')
    return 
    
#save_10000_gutenberg_sentences()
main_extracting_embeddings_sentences()
#main_extracting_embeddings_words()
#main_clustering_embeddings()
