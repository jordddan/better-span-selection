import torch
from tqdm import tqdm

import faiss
import time
import numpy as np
def knn_search(y, q, k, device="cuda:0"):
    '''
        y ()
    '''
    start_time = time.time()
    index = faiss.IndexFlatIP(256)  # 使用 L2 距离进行索引
    # index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)  # 将索引移动到 GPU 上
    # 将数据添加到索引中
    index.add(y)
    distances, indices = index.search(-q, k)
    end_time = time.time()
    run_time = end_time - start_time
    print("程序运行时间: {:.2f} 秒".format(run_time))
    import pdb
    pdb.set_trace()

    start_time = time.time()
    index.reset()
    end_time = time.time()
    run_time = end_time - start_time
    print("程序运行时间: {:.2f} 秒".format(run_time))

    index.add(y)
    # import pdb
    # pdb.set_trace()

    distances, indices = index.search(q, k)
    print(indices)
    # y = y.cpu().numpy()
    # index = faiss.GpuIndexFlat(y.shape[1])  
    # N = y.shape[0]

    # index.add(y)  
    # _,I = index.search(q,k)  
    # neighbors = torch.LongTensor(I) # find k nearest neighbors including itself -> Tensor [N,k]

    return 
if __name__ == "__main__":

    q = torch.rand((200,256))
    y = torch.rand((100000,256))

    knn_search(y,q,10)