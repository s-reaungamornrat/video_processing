## Jupyter

Data rate exceed limit
```
jupyter notebook --NotebookApp.iopub_data_rate_limit=1.0e10 
```

## Selection using indices

Assume we want to index into batched tensor. See below on how to do so
```
# we create data to show how to index into it
proposals=torch.rand(2, 100) # BxN
indices=torch.stack([torch.randperm(100)[:10] for _ in range(2)]) # assume this Bx10 is the index
# we need to create batch indices of shape Bx1
batch_indices=torch.arange(2)[:,None]
# we can index proposals like below
proposals[batch_indices, indices] # Bx10
```
Assume that we want to index into smaller tensor 
```
# we create data to show how to index into it
gt_boxes_in_image=torch.rand(2, 4) # 2x4
# 1D tensor of length 100 with value 0 or 1 to index into the 1st dim of gt_boxes_in_image
indices=(torch.rand(100)>0.5).to(dtype=torch.long) 
gt_boxes_in_image[indices] # 100x4

```