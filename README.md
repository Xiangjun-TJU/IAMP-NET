# IAMP-NET
Deep-unfolding method based on improved AMP is used to reconstuct underlying scenes from lensless measurements
# Introduction
—This paper proposes an unrolling learnable approximate message passing recurrent neural network (called ULAMP-Net) for lensless image reconstruction. By unrolling the optimization iterations, key modules and parameters are made learnable to achieve high reconstruction quality. Specifically, observation matrices are rectified on the fly through network learning to suppress systematic errors in the measurement of the point spread function. We devise a domain transformation structure to achieve a more powerful representation and propose a learnable multistage threshold function to accommodate a much richer family of priors with only a small amount of parameters. Finally, we introduce a multi-layer perceptron (MLP) module to enhance the input and an attention mechanism as an output module to refine the final results.
# Network

![Image text](https://github.com/Xiangjun-TJU/IAMP-NET/blob/main/Netork.png)
# Citation
If you use this code, please cite our work:

J. Yang, X. Yin, M. Zhang, H. Yue, X. Cui and H. Yue, "Learning Image Formation and Regularization in Unrolling AMP for Lensless Image Reconstruction," in IEEE Transactions on Computational Imaging, vol. 8, pp. 479-489, 2022, doi: 10.1109/TCI.2022.3181473.
