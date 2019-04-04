#!/bin/bash
#SBATCH --job-name=test_job_name
#SBATCH --output=test.o.txt #output of your pogram prints here
#SBATCH --mail-user=ViceLy07@gmail.com #email
#SBATCH --error=test.e.txt #file where any error will be written
#SBATCH --mail-type=ALL

python DenseLayersNN_1_Layer_50_Nodes.py trainfile.csv testfile.csv

