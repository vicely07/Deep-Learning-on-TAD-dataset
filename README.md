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






