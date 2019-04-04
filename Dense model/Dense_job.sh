#!/bin/bash
#SBATCH --job-name=Dense_job_name
#SBATCH --output=test.o.txt #output of your pogram prints here
#SBATCH --mail-user=ViceLy07@gmail.com #email
#SBATCH --error=test.e.txt #file where any error will be written
#SBATCH --mail-type=ALL

python DenseLayersNN_1_Layer_50_Nodes.py train_dixon_h1esc.prediction.csv test_dixon_h1esc.prediction.csv
python DenseLayersNN_2_Layer_50_Nodes.py train_dixon_h1esc.prediction.csv test_dixon_h1esc.prediction.csv
python DenseLayersNN_1_Layer_100_Nodes.py train_dixon_h1esc.prediction.csv test_dixon_h1esc.prediction.csv
python DenseLayersNN_2_Layer_100_Nodes.py train_dixon_h1esc.prediction.csv test_dixon_h1esc.prediction.csv
python DenseLayersNN_1_Layer_200_Nodes.py train_dixon_h1esc.prediction.csv test_dixon_h1esc.prediction.csv
python DenseLayersNN_2_Layer_200_Nodes.py train_dixon_h1esc.prediction.csv test_dixon_h1esc.prediction.csv

python DenseLayersNN_1_Layer_50_Nodes.py train_rao_hela_10kb.prediction.csv test_rao_hela_10kb.prediction.csv
python DenseLayersNN_2_Layer_50_Nodes.py train_rao_hela_10kb.prediction.csv test_rao_hela_10kb.prediction.csv
python DenseLayersNN_1_Layer_100_Nodes.py train_rao_hela_10kb.prediction.csv test_rao_hela_10kb.prediction.csv
python DenseLayersNN_2_Layer_100_Nodes.py train_rao_hela_10kb.prediction.csv test_rao_hela_10kb.prediction.csv
python DenseLayersNN_1_Layer_200_Nodes.py train_rao_hela_10kb.prediction.csv test_rao_hela_10kb.prediction.csv
python DenseLayersNN_2_Layer_50_Nodes.py train_rao_hela_10kb.prediction.csv test_rao_hela_10kb.prediction.csv

python DenseLayersNN_1_Layer_50_Nodes.py train_rao_hmec_10kb.prediction.csv test_rao_hmec_10kb.prediction.csv
python DenseLayersNN_2_Layer_50_Nodes.py train_rao_hmec_10kb.prediction.csv test_rao_hmec_10kb.prediction.csv
python DenseLayersNN_1_Layer_100_Nodes.py train_rao_hmec_10kb.prediction.csv test_rao_hmec_10kb.prediction.csv
python DenseLayersNN_2_Layer_100_Nodes.py train_rao_hmec_10kb.prediction.csv test_rao_hmec_10kb.prediction.csv
python DenseLayersNN_1_Layer_200_Nodes.py train_rao_hmec_10kb.prediction.csv test_rao_hmec_10kb.prediction.csv
python DenseLayersNN_2_Layer_200_Nodes.py train_rao_hmec_10kb.prediction.csv test_rao_hmec_10kb.prediction.csv
