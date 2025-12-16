community:

1. 16^2
   {'degree': 0.092859, 'cluster': 0.07484, 'orbit': 0.06235, 'avg': 0.076683}
2. 256^1
   {'degree': 0.0181, 'cluster': 0.076102, 'orbit': 0.00552, 'avg': 0.033241}, 0.4068s
   {'degree': 0.036404, 'cluster': 0.051566, 'orbit': 0.004089, 'avg': **0.030686**} , average: 0.0074 sec / 0.2427s

enzymes, 32^2:
enzymes, 1024^1:

community, 16^2, mask predict:   16_2-mlm

community, 256^1, mask predict:  256_1-mlm
{'degree': 0.027782, 'cluster': 0.07029, 'orbit': 0.01335, 'avg': **0.037141**} 没有比0.0306低很多, speed: 0.0039 sec

community, 256^1, mask predict, 从autoencoder开始训，并画图, recon_plot:  256_1-mlm-recon_plot

enzymes, 32^2, mask predict:
enzymes, 1024^1, mask predict:

| exp_name | dataset   | codebooksize | folder_name                     | mmd                                                                                                                                                                    | speed                                      | notes                 |  |
| :------: | --------- | ------------ | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------ | --------------------- | - |
| baseline | community | 16 2         |                                 | {'degree': 0.092859, 'cluster': 0.07484, 'orbit': 0.06235, 'avg': 0.076683}                                                                                            | 0.0097 sec                                 |                       |  |
|          |           | 256 1        |                                 | **{'degree': 0.036404, 'cluster': 0.051566, 'orbit': 0.004089, 'avg':0.030686}<br />{'degree': 0.0181, 'cluster': 0.076102, 'orbit': 0.00552, 'avg': 0.033241}** | **0.0074 s** / 0.2427s<br />0.4068s | 比2通道好             |  |
|          | enzymes   | 32 2         | baseline-16-2                   | sample报错                                                                                                                                                             | 0.0270 sec                                 |                       |  |
|          |           | 1024 1       | baseline-1024-1                 | 在跑                                                                                                                                                                   |                                            |                       |  |
|   mlm   | community | 16 2         | baseline-cb16_2-mlm             | {'degree': 0.010799, 'cluster': 0.069588, 'orbit': 0.002332, 'avg':**0.027573**} **sota!!**                                                                |                                            |                       |  |
|          |           | 256 1        | baseline-cb256_1-mlm            | {'degree': 0.027782, 'cluster': 0.07029, 'orbit': 0.01335, 'avg':**0.037141**}                                                                                   | 0.0039 sec                                 | avg没有低很多,快了2倍 |  |
|          |           | 256 1        | baseline-cb256_1-mlm-recon_plot | 在跑                                                                                                                                                                   |                                            |                       |  |
|          | enzymes   | 32 2         |                                 | 在跑                                                                                                                                                                   |                                            |                       |  |
|          |           | 1024 1       |                                 | 在跑                                                                                                                                                                   |                                            |                       |  |
