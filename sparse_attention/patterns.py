import torch

class SlidingWindow:
    def __init__(self, window=256, global_tokens=32): self.w, self.g = window, global_tokens
    def mask(self, n):
        m = torch.zeros(n,n,dtype=torch.bool)
        for i in range(n):
            m[i,max(0,i-self.w//2):min(n,i+self.w//2)] = True
            m[i,:self.g] = True
        return m

class BlockSparse:
    def __init__(self, block=64, top_k=8): self.b, self.k = block, top_k
    def mask(self, n):
        nb = (n+self.b-1)//self.b
        m = torch.zeros(n,n,dtype=torch.bool)
        for i in range(nb):
            for j in range(max(0,i-self.k),min(nb,i+self.k)):
                m[i*self.b:min((i+1)*self.b,n), j*self.b:min((j+1)*self.b,n)] = True
        return m
