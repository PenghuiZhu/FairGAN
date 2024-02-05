import numpy as np
import tensorflow as tf
from ctgan import CTGAN
from ctgan import load_demo
import pandas as pd

from ctgan.FairImprove.Classifier import Classifier
from ctgan.FairImprove.Dataset import Dataset
from ctgan.FairImprove.FairTransformerGAN import FairTransformerGAN
from ctgan.FairImprove.Metrics import Metrics

# real_data = load_demo()
# # Names of the columns that are discrete
# discrete_columns = [
#     'workclass',
#     'education',
#     'marital-status',
#     'occupation',
#     'relationship',
#     'race',
#     'sex',
#     'native-country',
#     'income'
# ]
#
# pd.set_option('display.max_row', 1000)
# pd.set_option('display.max_columns', 1000)
# ctgan = CTGAN(epochs=10)
# ctgan.fit(real_data, discrete_columns)
# synthetic_data = ctgan.sample(n=10)
# print(synthetic_data)

df = pd.read_csv('/Users/penghuizhu/Desktop/Workspace/FairGAN/examples/csv/adult.csv')
dataset = Dataset(df)
# saves processed data to interim/processed folder
np_input = dataset.pre_process(protected_var='sex',
                                outcome_var='income',
                                output_file_name='/Users/penghuizhu/Desktop/Workspace/FairGAN/ctgan/GeneratedData/test.txt', multiclass=True)

# get distribution of protected attribute race
p_z = dataset.get_protected_distribution(np_input)

# get distribution of outcome variable
p_y = dataset.get_target_distribution(np_input)

input_data_file='data/interim/adult_race_multi.pkl'
output_file='output/adult_race_fair_trans_gan/'

fairTransGAN = FairTransformerGAN(dataType='count',
                                    inputDim=np_input.shape[1] - 2,
                                    embeddingDim=128,
                                    randomDim=128,
                                    generatorDims=(128, 128),
                                    discriminatorDims=(256, 128, 1),
                                    compressDims=(),
                                    decompressDims=(),
                                    bnDecay=0.99,
                                    l2scale= 0.001,
                                    lambda_fair=1)

# clear any tf variables in current graph
tf.reset_default_graph()
fairTransGAN.train(dataPath=input_data_file,
                    modelPath="model.pth",
                    outPath=output_file,
                    pretrainEpochs=1,
                    nEpochs=1,
                    generatorTrainPeriod=1,
                    discriminatorTrainPeriod=1,
                    pretrainBatchSize=100,
                    batchSize=100,
                    saveMaxKeep=0,
                    p_z = p_z,
                    p_y = p_y)
# clear any tf variables in current graph
tf.reset_default_graph()
#  generate synthetic data using the trained model
fairTransGAN.generateData(nSamples=np_input.shape[0],
                modelFile='output/adult_race_fair_trans_gan/a1_1681280765-0',
                batchSize=100,
                outFile='data/generated/adult_race_fair_trans_gan_GEN/',
                p_z = p_z,
                p_y = p_y)
orig_data = np.load(input_data_file, allow_pickle = True)
orig_data.shape

output_gen_X = np.load('data/generated/adult_race_fair_trans_gan_GEN/.npy')
output_gen_Y = np.load('data/generated/adult_race_fair_trans_gan_GEN/_y.npy')
output_gen_z = np.load('data/generated/adult_race_fair_trans_gan_GEN/_z.npy')

output_gen = np.c_[output_gen_z, output_gen_X, output_gen_Y]

output_gen

# resize original data to be the same shape as generated data
orig_data = orig_data[:-42,]
print(output_gen.shape == orig_data.shape)
# convert numpy objects to df
gen_df = pd.DataFrame(output_gen)
orig_df = pd.DataFrame(orig_data)
# metrics evaluating the generated data
metrics = Metrics()
metrics.multi_fair_data_generation_metrics(gen_df)
# train a classifier using our logistic regression model (or use your own classifier) and return classification metrics
classifier = Classifier()
TestX, TestY, TestPred = classifier.logistic_regression(gen_df, orig_df)
# metrics evaluating the classifier trained on the generated data and predicted on the original data
metrics.multi_fair_classification_metrics(TestX, TestY, TestPred)
# train a classifier using our random forest model (or use your own classifier) and return classification metrics
TestX_r, TestY_r, TestPred_r = classifier.random_forest(gen_df, orig_df)
# metrics evaluating the classifier trained on the generated data and predicted on the original data
metrics.multi_fair_classification_metrics(TestX_r, TestY_r, TestPred_r)
# calculate euclidean distance metric
metrics.euclidean_distance(gen_df, orig_df)

