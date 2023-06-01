import torch
from tqdm import tqdm

import faiss

def graph_search(args,sim_matrix,dataset):
    '''
        inputs:
            sim_matrix -> Tensor[n,n] similarity score of the i-th and j-th sample

        outputs:
            noise_idx -> list the index of the noisy sample
    '''
    num_k = args.num_k
    labels = dataset.labels
    n = len(labels)
    vis = [1 for i in range(n)]

    for i in tqdm(range(n)):
        category = labels[i]
        vec = []
        for j in range(n):
            if labels[j] == category:
                vec.append([sim_matrix[i][j].item(),j])
        vec.sort(key=lambda x:x[0],reverse=True)

        for j in range(min(num_k,len(vec))):
            vis[vec[j][1]] += 1

    return vis

def check_select(is_clean,pred_clean):
    num = is_clean.shape[0]
    clean_clean = 0
    clean_noisy = 0 # true clean pred noise
    noisy_clean = 0
    noisy_noisy = 0
    for i in range(num):
        if is_clean[i].item() == True:
            if pred_clean[i].item() == True:
                clean_clean += 1
            else:
                clean_noisy += 1
        else:
            if pred_clean[i].item() == True:
                noisy_clean += 1
            else:
                noisy_noisy += 1
    # print(clean_clean,clean_noisy)
    print("pred clean num: {}; noise ratio in select set: {}, pred noise num: {}; noise ratio in noise set: {}".format(clean_clean + noisy_clean,\
                                                                                                                       noisy_clean/(noisy_clean+clean_clean),\
                                                                                                                       clean_noisy + noisy_noisy,\
                                                                                                                       noisy_noisy/(noisy_noisy+clean_noisy)))

def knn_search(args,features,dataset):
    '''
        description: 选出k个近邻样本,由这一轮的分类器logits 拿到样本的knn logits
    '''
    cls_logits = dataset.logits.cuda() # 没有softmax以后的结果,每个样本这一轮的logits
    labels = torch.tensor(dataset.labels,dtype=int).cuda() #
    num_class = dataset.num_class
    features = features.cpu().numpy()
    index = faiss.IndexFlatIP(features.shape[1])  
    N = features.shape[0]

    k = args.num_k
    index.add(features)  
    _,I = index.search(features,k)  
    neighbors = torch.LongTensor(I) # find k nearest neighbors including itself -> Tensor [N,k]

    knn_logits = torch.zeros((N,num_class),dtype=float).cuda()

    for i in tqdm(range(N)):
        # import pdb
        # pdb.set_trace()
        # neighbors_labels = labels[neighbors[i]]
        tmp = cls_logits[neighbors[i]].sum(dim=0)
        knn_logits[i] = tmp
    return knn_logits

def knn(index, y, query, k):

    index.add(y)
    distances, indices = index.search(query, k)
    index.reset()

    return indices