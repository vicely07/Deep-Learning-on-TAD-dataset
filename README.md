# Deep-Learning-on-TAD-dataset

# Introduction:
TADs are self-interacting regions in a DNA sequence. The functions of TADs are not fully understood but it is known that disrupting these boundaries can affect the expression of certain genes, leading to disease. However, the boundaries of TADs are highly conserved across different cell types even though the organization within a TAD may differ across cell types, which suggests an importance of the TAD boundaries. 

![TAD](https://github.com/vicely07/Deep-Learning-on-TAD-dataset/blob/master/Chart%20and%20Table/Picture9.png)
**Figure 1:** Representation of TAD compartments within the nucleus of a cell

# Dataset
The fruit fly DNA data is from the Gene Expression Omnibus (GEO) and consists of ~28,000 training and 2000 testing sequences. Each sequence is analyzed for every possible 9-mer combination, 256 total. 

![Raw Data](https://github.com/vicely07/Deep-Learning-on-TAD-dataset/blob/master/Chart%20and%20Table/Picture11.jpg)
**Figure 2:** Our raw data.

# Predictive model:
![Models](https://github.com/vicely07/Deep-Learning-on-TAD-dataset/blob/master/Chart%20and%20Table/Picture1.png)
**Figure 3:** Three representative deep learning models. Panel (A) shows the 1layerCNN model that consists of input layer, CNN layer, pooling layer, flatten layer, and output layer. Panel (B) shows the 1layerCNN_LSTM model which was comprised of input layer, CNN layer, pooling layer, bidirectional LSTM layer and output layer. Panel (C) shows 1layerCNN_Dense consisting of input layer, CNN layer, pooling layer, 2 dense layers, and output layer with sigmoid () activation function. 

# Evaluation of deep learning models:
![Boxplot](https://github.com/vicely07/Deep-Learning-on-TAD-dataset/blob/master/Chart%20and%20Table/Picture3.png)
**Figure 4:** Deep learning model’s performance metrics versus hyper-parameters. The figure shows boxplots for six different performance metrics (auc: area under the curve, acc: accuracy, mcc: Matthews correlation coefficient (MCC), precision, recall, and f1 score) of 12 different deep learning architectures for different permutations of hyper-parameter set tested. 

![Bar chart](https://github.com/vicely07/Deep-Learning-on-TAD-dataset/blob/master/Chart%20and%20Table/Picture2.png)
**Figure 5:** Best performance by 12 different deep learning model architectures. Figure shows the best performance achieved by each of the 12-different deep learning model architectures. Six different metrics were used (auc: area under the curve, acc: accuracy, mcc: Matthews correlation coefficient (MCC), precision, recall, and f1 score).


# Results:
**Table 1:** Deep Learning versus feature-based models. (A) Five different permutation sets of hyper-parameters for 3layersCNN_LSTM model that yielded the top five performances. The first six columns indicate six different performance metrics and the remaining columns indicate the hyper-parameters. (B) six evaluation metrics for 5 feature-based models and gkmSVM.

![Table](https://github.com/vicely07/Deep-Learning-on-TAD-dataset/blob/master/Chart%20and%20Table/Picture7.png)

# Motifs:

![Motif](https://github.com/vicely07/Deep-Learning-on-TAD-dataset/blob/master/Chart%20and%20Table/Picture10.png)

# Conclusion:
By extensive exploration and testing of several deep learning model architectures with hyper-parameter optimization, we show that a unique deep learning model consisting of three convolution layers followed by long short-term memory layer achieves an accuracy of 96%. This outperforms feature based model’s accuracy of 91% and an existing method’s accuracy of 73-78% based on motif TRAP scores. Out of the 64 motifs, only 12 matched known annotated motifs of fruit flies. Interestingly, Beaf-32 motif was detected as the second highest scoring motif by our model resonating with previous reports of strong enrichment of Beaf-32 motif in TAD boundaries of fruit flies. Previously reported insulator proteins associated with TAD boundaries in insects such as Trl and Z4 were also detected. Factors involved in development and chromatin modeling such as Byn, Ovo, and Pho were also detected but not previously reported in the context of TADs boundaries.

# Fututure Research:
Each of these motifs collectively or combinatorically contribute to the prediction power of the deep learning model. To fully decipher the nature of interaction between these motifs is beyond the scope of this paper and will be addressed in future studies.

# References:
1 .Drosophila Melanogaster Genome Dataset. (n.d.). Retrieved from https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=7227
2. .Dixon, J. R., Gorkin, D. U. & Ren, B. Chromatin Domains: The Unit Of Chromosome Organization. Mol. Cell 62, 668–680 (2016).
3. Tiana, G., et al. (2016). Structural Fluctuations of the 	Chromatin Fiber within Topologically Associating Domains.Journal,110(6), 1234-1245. 	
4. Matharu, N., & Ahituv, N. (2015). Minor Loops in Major Folds: Enhancer–Promoter Looping, Chromatin Restructuring, and Their Association with Transcriptional Regulation and Disease.PLOS Genetics,11(12).







