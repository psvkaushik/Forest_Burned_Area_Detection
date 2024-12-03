# Forest_Burned_Area_Detection
Term Project for CSC 791- Geospatial  AI

All the notebooks are present in the [`notebooks`](./notebooks/) directory. The dataset is taken from the paper [CaBuAr: California Burned Areas dataset for delineation ](https://arxiv.org/abs/2401.11519), and the dataset is available on [huggingface](https://huggingface.co/datasets/DarthReca/california_burned_areas).

# Best Performing Model 

## The architecture
The architecture is as shown ![architecture of the best perfroming model](img/best.png)
It beats the baseline provided by the paper by a significant margin as shown in the table.
| Model Variant       | F1-Score | IOU |
|---------------------|----------|-----|
| UNet (Weighted Concat)      | 88%      | 67% |
| UNet     | 62%      | 49.5% |