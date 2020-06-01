# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # Data science request #5 - an exploratory approach
#
# * Authors: **David <david@omind.me>** & **Raph <raphaelle@omind.me>**
# * Date: 2019-07-22
# * Link to trello card: https://trello.com/c/AS5yTMSC
#
# The original research question was:
#
# > Estimation de la distribution dans la population des individus qui s'activent plutôt en tonique ou plutôt en phasique (profils de régulation physiologique - gestion du stress) + sélection de features et clustering
#
# This notebook is a straight forward analysis to start working towards this question
# with a simple linear regression and a clustering approach.
# The main purpose is to keep this analysis simple and interpretable. 
#
# -------------------------------------------------
#
# # Summary
#
#
# This analysis aims to study the different galvanic activation profiles during the virtual reality experiment. The analysis is in two stages:
#  - it starts by training an activation model to estimate the probability of an observation to witness 'phasic activation' or 'tonic activation'. In order to train and cross-validated the model, we need to select sequences to be labeled as 'rest' versus 'stress'. **Those are to be discussed.** 
#  - then, once we have quantified the SCL and SCR activations, for each observation (ie. one participant during one module), we:
#      - visualize the relationships between these two activations (SCR and SCL) and their distributions against the context (ie. across modules or given the game difficulty). **The conclusion is that there is a correlation of 0.5 between phasic and tonic activations. In 75% of the observations, tonic activation (resp. decompression) goes together with phasic activation (resp. decompression); In 18.2 % of observations, we have tonic activation without phasic activation; In 7.4% of observations with have a phasic activation without tonic activation**
#      - select a subset of VR stimulations (eg. attack waves and suveys during space-stress) and cluster participants given their activations during those given stimulations. 
#      - estimate the variances and covariances between phasic/tonic activation and the context in space-stress (ie. shaped by wave difficulty and surveys)  
#      - **We could not clearly identify clusters**. 
#
# # Details
#
# Note that for this notebook, I am trying to discuss the results *before* showing the related figure or table.
#
# ## Preparations
#
# Code needed to import or prepare libraries needed for this notebook.
#
# Common important libraries:

# +
# %matplotlib inline

import collections
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing

# Set a random seed for reproducibility
np.random.seed(42)
# -

# Connection with **R** needed to use `ggplot` and import common important R libraries

# %load_ext rpy2.ipython

# + {"language": "R"}
#
# library(ggplot2)
# library(GGally)
# -

# Configuration of **Quetzal** client to obtain data

# +
from quetzal.client import helpers

if 'QUETZAL_USER' not in os.environ or 'QUETZAL_PASSWORD' not in os.environ:
    raise ValueError('You need to set the quetzal environment variables '
                     'to retrieve the data of this notebook')

client = helpers.get_client(
    url='https://quetzal.omind.me/api/v1',
    username=os.getenv('QUETZAL_USER'),
    password=os.getenv('QUETZAL_PASSWORD'),
    insecure=False
)
helpers.auth.login(client);  # Just to check if the username/password works


# + {"toc-hr-collapsed": false, "cell_type": "markdown"}
# ## Data
#
# This section consists on a description of how the GSR dataset is defined, then
# how this dataset was converted to different shapes specific for this analysis.

# + {"toc-hr-collapsed": false, "cell_type": "markdown"}
# ### GSR dataset
#
# Dataset for this notebook is the GSR feature matrix from Iguazu version 0.0.3.
#
# The following cell downloads the GSR features locally:

# +
GSR_SUMMARY_ID = '7e30a399-fc86-4796-bd85-de1e4045edcc'

gsr_csv = helpers.file.download(client, file_id=GSR_SUMMARY_ID, wid=50, output_dir='data')
print('GSR features CSV downloaded at', gsr_csv)
# -

# ... and now we can read it with pandas:

df = pd.read_csv(gsr_csv, index_col=0)  # index_col=0 needed because it was saved without xxx.to_csv(..., index=False)
print('Dataframe has shape', df.shape)
df.head()

duplicated_fid = np.unique(df[df.drop('file_id', axis=1).duplicated()].replace('bad', np.NaN).dropna().file_id)
print(f'Found {len(duplicated_fid)} duplicated files: {duplicated_fid}. Removing it from further analysis. ')
df = df[~df.file_id.isin(duplicated_fid)]

# replace 'bad' by NaN and convert to numeric if possible.
df = df.replace('bad', np.NaN).apply(lambda s: pd.to_numeric(s, errors='ignore'))

# droping an outlier for feature gsr_SCL_linregress_slope
df = df[df.gsr_SCL_linregress_slope.abs()<df.gsr_SCL_linregress_slope.abs().mean() + 1.5*df.gsr_SCL_linregress_slope.abs().std()]

# The number of files (i.e. subjects) considered in this analysis:

n_files_all = len(df.file_id.unique())
print(f'The initial dataframe had {n_files_all} files')

# The following sequences are available in the dataset:

df.groupby('sequence').size().to_frame(name='n')

# The following cell filters all observations to keep only the features of interest, and shortcuts their names for lisibility purpose.  
#
# It also introduces columns *module*/*module_label* that gives the sequence (eg. cardiac-coherence) and *submodule*/*submodule_label* that gives the subpart (eg. survey). 
#

# +
# Choice of the feature to include in the logistic regression 
#features_dict = {
#    # feature name in csv                 short name scl/scr
#    'gsr_SCL_median':                            'scl_median',
#    'gsr_SCR_peaks_detected_rate':               'scr_rate',
#}

features_dict = {
    # feature name in csv                 short name scl/scr
    'gsr_SCL_auc':                               'scl_auc',
    'gsr_SCL_linregress_rvalue':                 'scl_rvalue',
    'gsr_SCL_linregress_slope':                  'scl_slope', 
    'gsr_SCL_median':                            'scl_median',
    'gsr_SCL_ptp':                               'scl_ptp',
    'gsr_SCL_std':                               'scl_std',
    'gsr_SCR_peaks_detected_rate':               'scr_rate',
    'gsr_SCR_peaks_increase-amplitude_median':   'scr_amplitude',
    'gsr_SCR_peaks_increase-duration_median':    'scr_duration'
}

# +
df = (df[list(features_dict.keys()) + ['sequence', 'file_id']]
    .reset_index(drop=True)
    .rename(columns=features_dict)
)

df['module'] = df.sequence.apply(lambda column: column.split('_')[0])
df['submodule'] = df.sequence.apply(lambda column: column.split('_')[1])
le_module = preprocessing.LabelEncoder()
le_submodule = preprocessing.LabelEncoder()

df['module_label'] = le_module.fit_transform(df['module'])
df['submodule_label'] = le_submodule.fit_transform(df['submodule'])

df_galvanic = df
df.head()
# + {"toc-hr-collapsed": false, "cell_type": "markdown"}
# ### Context dataset
#
# Dataset for this notebook is the Behavior feature matrix from Iguazu version 0.0.3.
#
# The following cell downloads the behavior features locally:

# +
BEHAVIOR_SUMMARY_ID = '57ba28a7-00d1-4915-b89e-64b7ae43ef36'

behavior_csv = helpers.file.download(client, file_id=BEHAVIOR_SUMMARY_ID, wid=50, output_dir='data')
print('Behavior features CSV downloaded at', behavior_csv)
# -

# ... and now we can read it with pandas:

df = pd.read_csv(behavior_csv, index_col=0)  # index_col=0 needed because it was saved without xxx.to_csv(..., index=False)
print('Dataframe has shape', df.shape)
df.head()

space_stress_sequences = {
    # index name:         official sequence name 
    'wave0':             'space-stress_game_enemy-wave_0',
    'wave1':             'space-stress_game_enemy-wave_1', 
    'wave2':             'space-stress_game_enemy-wave_2', 
    'survey0':           'space-stress_survey_0',
    'wave3':             'space-stress_game_enemy-wave_3',
    'wave4':             'space-stress_game_enemy-wave_4',
    'wave5':             'space-stress_game_enemy-wave_5', 
    'survey1':           'space-stress_survey_1', 
    }
context_features = {'information_motion_tau': 'difficulty', 
                    'global_accuracy': 'accuracy'}

survey_difficulty = 0.0
df = df.rename(columns=context_features).reset_index(drop=True)
df = df.loc[df.sequence.isin(space_stress_sequences)][['sequence', 'file_id', 'difficulty', 'accuracy']]
    # select only the (columns) features and covariates of interest
df = pd.concat((df, 
            df.groupby('file_id')
            .apply(lambda d: pd.Series(dict(
                                            difficulty=survey_difficulty, 
                                            sequence='survey0')))
            .reset_index(), 
            df.groupby('file_id')
            .apply(lambda d: pd.Series(dict( 
                                            difficulty=survey_difficulty, 
                                            sequence='survey1')))
            .reset_index()), axis=0, ignore_index=True, sort=False)
df['sequence'] = df.sequence.apply(lambda s: space_stress_sequences[s])
df_ss_beh = df[df.difficulty<df.difficulty.median() + 3*df.difficulty.std()] # drop outlier
df_ss_beh.head()

# ## Phasic/Tonic activation 
#
# The first part of this DS request is to define/quantify the galvanic activation, either phasic or tonic. <br>
# ### Activation model 
# To answer this, we train an activation model on the SCR and SCL features to separate 'resting' from 'stressing' sequeences. <br>
# For simplicity and interpretability purpose, we choose a cross-validated regression logistic, that allows us to understand how each feature contributes to the activation. <br>
# The chosen sequences are: 
# - **rest**: surveys during cardiac-coherence and physio-sonification
# - **stress**: first and last wave from space-stress 
#
# The 'no go' threshold to validate the approach is to obtain a discirmination roc_auc greater than 0.75.


# +
# Choice of the sequences to train the logistic regression of activation. 
training_sequences = {
    # sequence name:                         context, sequence_short, category type 
    'physio-sonification_survey_0':             (0, 'survey_son', 'rest'),
    'cardiac-coherence_survey_1':               (0, 'survey_coh','rest'),
   # 'space-stress_breath-tutorial_0':           (0, 'breath_tuto','rest'),
    'space-stress_game_enemy-wave_0':           (1, 'wave_0','stress'),
    'space-stress_game_enemy-wave_5':           (1, 'wave_5','stress'), 
    #'space-stress_game-tutorial_0':             (1, 'game_tuto','rest'),

}
df = df_galvanic
df_train = (
    # filter only the (rows) sequences of interest
    df.loc[df.sequence.isin(training_sequences)]
)
df_train['sequence_short'] = df_train.sequence.apply(lambda s: training_sequences[s][1])
df_train['context'] = df_train.sequence.apply(lambda s: training_sequences[s][0])
df_train['sequence_label'] = df_train.sequence.apply(lambda s: training_sequences[s][2])

df_train = df_train.dropna()
# -



# + {"magic_args": "-i df_train  -w 5 -h 5 --units in -r 200", "language": "R"}
# # Change column order
# columns = c('scr_rate', 'scr_amplitude', 'scr_duration', 'sequence_label')
# tmp = df_train[columns]
# ggpairs(tmp, 
#     title='SCR features', 
#     columns=1:3,
#     mapping=aes(colour=sequence_label, alpha=.3), 
#     progress=FALSE) 
# -

# Visually, it seams that scr_rate itself can separate the two conditions.  It should be notted that the 0.0 values in scr_amplitude and scr_duration are due to the observation where no SCR peak could be detected. Indeed, the scr_rate has been 'baseline-corrected', which exlpains that we observe negattive rate (relative to the pseudo-baseline period) 

# + {"magic_args": "-i df_train  -w 5 -h 5 --units in -r 200", "language": "R"}
# # Change column order
# columns = c('scl_auc', 'scl_rvalue', 'scl_slope', 'scl_median', 'scl_ptp', 'scl_std', 'sequence_label')
# tmp = df_train[columns]
# ggpairs(tmp, 
#     title='SCL features', 
#     columns=1:6,
#     mapping=aes(colour=sequence_label, alpha=.3), 
#     progress=FALSE) 
# -
# Visually, it seams that scl_auc and scl_median are very discriminant against the two conditions. It should be notted that those are extremely correlated, which is not desirable when fitting a model. 


# +
# create a helper class to allow storing variables with same names for the different analysis 
class PencilCase():
    ''' Helper class to store stuffs'''
    def __init__(self, name=None):
        self._name = name

estimate_activation = PencilCase('train_activation')
estimate_activation.scr = PencilCase('scr')
estimate_activation.scl = PencilCase('scl')
estimate_activation.df_train = df_train
estimate_activation.df = df
# -

# Here, we use a logistic regression with a cross-validation on the choice of Cs and l1_ratios values. This is equivalent to selecting the best features before fitting the logistic regression model. <br>
# The 'no go' threshold to validate the approach was to obtain a discirmination roc_auc greater than 0.75. We got 0.84 and 0.89 with a cross-valdiation with 10 folds. <br>
#
#
#
# <div class="alert alert-info">
# Here, we scale the data using a StandardScaler, even though some distributions are clearly biased (eg. src_amplitude and scr_duration have lot's of 0.0 values). This results in variations in the features ranges. <br>
# --> Baseline correction of those?
# </div>
#
# <div class="alert alert-info">
# Surveys used as 'rest' state are also used in the estimation of the pseudo-baseline, so contributed to the correction, which could result in biasing those features. <br> 
# --> Use a different baseline correction   
# </div>
#

# +
from sklearn import preprocessing
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegressionCV

model =  LogisticRegressionCV(Cs=np.logspace(-6, 6, base=10, num=20), fit_intercept=True,
                                                cv=10, dual=False, penalty='l1', 
                                                scoring='roc_auc', solver='liblinear', 
                                                tol=0.0001, max_iter=100,
                                                class_weight=None, 
                                                n_jobs=None, 
                                                verbose=0, 
                                                refit=True, 
                                                intercept_scaling=1.0,
                                                random_state=None, l1_ratios=None)
scaler = preprocessing.StandardScaler()
#scaler = preprocessing.MinMaxScaler()
# -

case = estimate_activation.scl
case.features = ['scl_auc', 'scl_rvalue', 'scl_slope', 'scl_median', 'scl_ptp', 'scl_std']
#case.features = ['scl_median']
df = df_train[case.features + ['context']]
case.X = df.values[:,:-1]
case.X_scaled = scaler.fit_transform(case.X)
case.y = df.values[:,-1]
case.df = df
df.head()

# +
case = estimate_activation.scr
case.features = ['scr_rate', 'scr_amplitude', 'scr_duration']
# case.features = ['scr_rate']

df = df_train[case.features + ['context']]
case.X = df.values[:,:-1]
case.X_scaled = scaler.fit_transform(case.X)
case.y = df.values[:,-1]
case.df = df
df.head()

# +
from copy import copy
# KFold/StratifiedKFold cross validation with 10 folds 
def train_activation_model(case): 
    print(f'Training activation model for {case._name}')
    print('-'*35)
    cv_score = cross_validate(model, case.X_scaled, case.y, cv=10, scoring=['accuracy', 'roc_auc'])
    for s, vals in cv_score.items():
        print(f'{s} = {vals.mean():.2f} +/- {vals.std():.2f}')

    fitted_model = model.fit(case.X_scaled, case.y)
    case.features_coef = pd.DataFrame(fitted_model.coef_.ravel(), index=case.features, columns=['coef'])
    case.model = copy(fitted_model)
    print(case.features_coef)
    print('_'*100)
    
train_activation_model(estimate_activation.scr)
train_activation_model(estimate_activation.scl)
# -

# The cross-validated roc_auc scores being good, we validate the approach and estimate the probability to be 'phasic-activated' and the probability to be 'tonic-activated'. 

# <div class="alert alert-info">
#     We are going to introduce a variable rename in favor of readability, as explained below.
# </div>
#
# **Variable rename list:**
#
# * `SCL` will be used for the probability to be 'tonic-activated'
# * `SCR` will be used for the probability to be 'phasic-activated'

# +
case = estimate_activation.scr
df = estimate_activation.df[case.features]
valid_index = df.dropna().index
X = df.dropna().values

X_scaled = scaler.fit_transform(X)
proba = case.model.predict_proba(X_scaled)[:,1]
case.proba_df = pd.DataFrame(proba, index=valid_index, columns=[case._name])
# -

case = estimate_activation.scl
df = estimate_activation.df[case.features]
valid_index = df.dropna().index
X = df.dropna().values
X_scaled = scaler.fit_transform(X)
proba = case.model.predict_proba(X_scaled)[:,1]
case.proba_df = pd.DataFrame(proba, index=valid_index, columns=[case._name])

estimate_activation.proba_df = pd.concat([estimate_activation.df[['sequence', 'file_id', 'module', 'submodule', 'module_label', 'submodule_label']], 
                                          estimate_activation.scl.proba_df, 
                                          estimate_activation.scr.proba_df], axis=1)
estimate_activation.proba_df.head()

# ### Relationship between phasic and tonic activation

#estimate_activation.proba_df.plot.scatter(x='scl_activation',y='scr_activation', colors=)
df_all = estimate_activation.proba_df
df_all = df_all[~df_all.submodule.isin(['sequence'])]
df_all = df_all[~df_all.module.isin(['session', 'common', 'lobby', 'intro'])]
df_ss = df_all[df_all.module == 'space-stress']
df_coh = df_all[df_all.module == 'cardiac-coherence']
df_son = df_all[df_all.module == 'physio-sonification']

# Let us quantify if 'tonic-activation' goes together with 'phasic-activation'. We set 0.5 as threshold for the probability to be considered 'activated' and we *count* how many times we observe a phasic together with tonic activation, or on the contrary without. 

df = df_all.dropna()
R_noL = len(df_all[(df_all.scr>.5) & (df_all.scl<.5)])/len(df)
R_L = len(df_all[(df_all.scr>.5) & (df_all.scl>.5)])/len(df)
noR_noL = len(df_all[(df_all.scr<.5) & (df_all.scl<.5)])/len(df)
L_noL = len(df_all[(df_all.scr<.5) & (df_all.scl>.5)])/len(df)
print("Observation of phasic together with tonic: %8.2f percent. "% (R_L*100))
print("Observation of no tonic and no phasic : %8.2f percent. "% (noR_noL*100))
print("Observation of tonic without phasic: %8.2f percent. "% (L_noL*100))
print("Observation of phasic without tonic: %8.2f percent. "% (R_noL*100))

# + {"magic_args": "-i df_all  -w 5 -h 5 --units in -r 200", "language": "R"}
# columns = c('scl', 'scr', 'module')
# tmp = df_all[columns]
# head(tmp)
# ggpairs(tmp, 
#     title='Activation features per modules', 
#     columns=1:2,
#     mapping=aes(colour=module, alpha=.3), 
#     progress=FALSE) 
# -
# In 75% of the observations, a tonic activation (resp. decompression) goes together with a phasic activation (resp decompression). The correlation between probability to be tonic-activated or phasic-activated is of 0.5. 

# ### Activations against experiment context 
# #### Activations against modules and submodules

# + {"magic_args": "-i df_all -w 16 -h 6 --units in -r 200", "language": "R"}
#
# columns = c('scl', 'scr', 'module', 'submodule')
# tmp = df_all[columns]
# tmp = reshape::melt(tmp,
#                         id.vars=c('module', 'submodule'),
#                         measure.vars=c('scr', 'scl'),
#                         variable_name='feature')
#
# g1 = ggplot(tmp, aes(x=submodule, y=value, fill=feature)) +
#     geom_boxplot()+
#     geom_hline(yintercept=.5, linetype='dashed', size=1, color='black')+
#     facet_wrap(.~module, ncol=4, scales='free') +
#     labs(title="Activation amongst VR submodules", size=25) + 
#     theme(axis.text.x = element_text(angle = 45, hjust = 1), plot.title=element_text(size=25))
# g1
# -

# Here, we check that the model has not overfitted. <br>
# Hence, we fitted the model on 2 surveys and 2 waves and we clearly see that observation during space-stress game are activated (probability greater than .5) , whereas observation during baselines and relaxing modules are not (probability lower than .5). 

# let us now zoom on space-stress 
df_ss = df_all[df_all.module=='space-stress']
df_ss.loc[:, 'sequence_short'] = df_ss.sequence.str.replace('space-stress_', '', 1).str.replace('game_enemy-', '', 1)

# + {"magic_args": "-i df_ss -w 16 -h 6 --units in -r 200", "language": "R"}
#
# columns = c('scr', 'scl', 'sequence_short')
# tmp = df_ss[columns]
#
# tmp = reshape::melt(tmp,
#                         id.vars=c('sequence_short'),
#                         measure.vars=c('scr', 'scl'),
#                         variable_name='feature')
#
# g1 = ggplot(tmp, aes(x=sequence_short, y=value, fill=feature)) +
#     geom_boxplot()+
#     geom_hline(yintercept=.5, linetype='dashed', size=1, color='black')+
#     scale_x_discrete(limits=c('intro_0',
#                               'game-tutorial_0', 
#                               'wave_0', 'wave_1', 'wave_2',
#                               'survey_0','graph_0', 
#                               'breath-tutorial_0',
#                               'wave_3', 'wave_4', 'wave_5', 
#                               'survey_1', 'graph_1', 
#                               'outro_0')) + 
#     labs(title="Activation amongst Space-Stress") + 
#     theme(axis.text.x = element_text(angle = 45, hjust = 1), plot.title=element_text(size=25))
# g1
# -
# Here, we observe that, on average, the participan during space-stress are highly activated during the 6 waves, and less during the surveys, graphs and tutorials. <br>
# It should be notted that the phasic-activation seems more variable across participant than the tonic-activation. 

# +
space_stress_sequences = {
    # sequence name:                  category order
  # 'game_0':    0,
 #  'game_1':    2,
  # 'survey_0':  1,
  # 'survey_1':    3,
    'game-tutorial_0':    0,
    'wave_0':             1,
    'wave_1':             2,
    'wave_2':             3,
    'survey_0':           4,
    'breath-tutorial_0':  5,
    'wave_3':             6,
    'wave_4':             7,
    'wave_5':             8,
    'survey_1':           9,
}
df_ss = df_ss[df_ss.sequence_short.isin(space_stress_sequences)]
df_ss['context'] = df_ss.sequence_short.apply(lambda s: space_stress_sequences[s])


df_ss.head()
# -

# #### Activation against space-stress difficulty 

df_ss_merged = df_ss.merge(df_ss_beh, on=['file_id', 'sequence'], how='right', 
            left_index=False, right_index=False)
df = df_ss_merged
df_ss_merged.head()


# + {"magic_args": "-i df  -w 9 -h 3 --units in -r 200", "language": "R"}
#
# g1 = ggplot(df, aes(x=difficulty, y=scl, color=scr, alpha=.2)) +
#         geom_point() +
#         scale_color_gradient(low='gray', high='red') + 
#         theme(legend.position='up')
#     
# g2 = ggplot(df, aes(x=difficulty, y=scr, color=scl, alpha=.2)) +
# geom_point() +
# scale_color_gradient(low='gray', high='green') + 
# theme(legend.position='up')
#
# g3 = ggplot(df, aes(x=scl, y=scr, color=difficulty, alpha=.2)) +
# geom_point() +
# scale_color_gradient(low='gray', high='blue') + 
# theme(legend.position='up')
#
# cowplot::plot_grid(g1, g2, g3, nrow=1)
# -

# **Observations:**
# - scr and scl are correlated during space-stress 
# - no clear correlation pattern between scr/scl activations and the waves difficulty (at population level)
#
# ### Covariance dataset 
# Here, we construct a dataset with variance/covariance of the tonic/phasic activations and the context (difficulty of the waves and 0.0 for the surveys). <br>
# The purpose is to quantify how well a participant galvanic activations follows the context, ie. how 'apropriate' the activation is given the difficulty. 

def melt_cov(df):
    df_cov = df.cov()
    keep = np.triu(np.ones(df_cov.shape)).astype('bool').reshape(df_cov.size)
    df_cov_melt = df_cov.stack(dropna=False)[keep]
    df_cov_melt.index = df_cov_melt.index.to_list()
    return df_cov_melt


# +
df_cov = (
    df_ss_merged[['scr', 'scl', 'file_id', 'difficulty']]
    .groupby('file_id')
    .apply(lambda d: melt_cov(d))
    .dropna()
        )
df_cov.iloc[:, :] = preprocessing.StandardScaler().fit_transform(df_cov)
df_cov.columns.name = 'cov'
df_cov.columns = [level[0] + '_' + level[1] for level in  list(df_cov.columns)]

df_cov.head()


# -

# ### Exploratory plots

#

# + {"toc-hr-collapsed": false, "cell_type": "markdown"}
# ## Method
#
# We now have 4 datasets: 
# - scl: dimension 8, with phasic activations during the 6 space-stress waves and the 2 surveys 
# - scr: dimension 8, with tonic activations during the 6 space-stress waves and the 2 surveys 
# - sc: dimension 16, with both tonic and phasic activations during the 6 space-stress waves and the 2 surveys 
# - cov: dimension 6 with the variances and covariances between physio activation (SCR and SCL) and the context (space-stress difficulty). 
#
# The aim is to see if we can draw clusters/profiles out of those datasets. 
#
# <div class="alert alert-info">
# One difficulty here is to scale the dataset without disturbing the vector order (ie. for each feature, the order of the activation across the 6 space-stress periods must remain the same). <br>
# Here, we fit the standard scaler for each dimension of the entire dataset and we then transform each column with the same parameters, so that the 'shapes' of each vector is conserved.  
# </div>
#
# -

# ### Exploratory dimension reduction
#
# On this section, we would like to reduce each dataset to 2 dimensions and quantify how good this projection is. 
# If a projection in a reduced space is explicative and we can see some spatial structure (clusters, groups, ...), 
# then this would solve the problem of this notebook.

# #### PCA
#
# A PCA for the SCL dataset explains a lot of variability in two projected dimensions. 
# However, there is no evident spatial structure: one cannot clearly draw or guess that there are clusters of data.

def scale_df(df):
    ''' Scales a dataframe by fitting on all the data'''
    scaler = preprocessing.StandardScaler()
    scaler.fit(df.stack().values.reshape(-1, 1))
    df_scaled = df.apply(lambda col: scaler.transform(col.values.reshape(-1,1))[:,0])
    return df_scaled 


# +
profiling = PencilCase('train_activation')

profiling.df_ss = df_ss

profiling.scr = PencilCase('scr')
profiling.scl = PencilCase('scl')
profiling.sc = PencilCase('sc')
profiling.cov = PencilCase('cov')

sequences_short = ['survey_0', 'survey_1', 
                  # 'game_0', 'game_1']
                   'wave_0',  'wave_1',  'wave_2',
                   'wave_3',  'wave_4',  'wave_5']

for case, values in zip([profiling.scl, profiling.scr], 
                          [['scl'], ['scr']]):
    case.df = (
    df_ss[df_ss.sequence_short.isin(sequences_short)].pivot(columns='sequence_short', values=values, index='file_id')
    .dropna(how='any')
)
    case.df = scale_df(case.df)
    #case.df.iloc[:,:] = preprocessing.StandardScaler().fit_transform(case.df)
    case.X = case.df.values
    case.df.columns = [level[0] + '_' + level[1] for level in  list(case.df.columns)]

profiling.sc.df = pd.concat([profiling.scl.df, profiling.scr.df], axis=1, sort=True).dropna()
profiling.sc.df.index.name = 'file_id'
profiling.sc.X = profiling.sc.df.values

profiling.cov.df = df_cov
profiling.cov.df.iloc[:,:] = preprocessing.StandardScaler().fit_transform(df_cov)
profiling.cov.X = profiling.cov.df.values


# +
from sklearn.decomposition import PCA


def biplot(x, coefs, labels=None, k=1, ax=None):
    """Utility function to plot PCA results
    
    Inspired from https://stackoverflow.com/a/46766116/227103
    """
    if ax is None:
        ax = plt.gca()

    n = coefs.shape[0]
    if labels is None:
        labels = [f'X{i}' for i in range(n)]

    xs = x[:, 0]
    ys = x[:, 1]
    scalex = 1.0 / (xs.max() - xs.min())
    scaley = 1.0 / (ys.max() - ys.min())

    ax.scatter(xs * scalex, ys * scaley)

    for i in range(n):
        ax.arrow(0, 0, coefs[i, 0] * k, coefs[i, 1] * k, color='r', alpha=0.5)
        ax.text(coefs[i, 0] * k * 1.15, coefs[i, 1] * k * 1.15, labels[i], color='g',
                ha='center', va='center')

    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.grid()


# -

for case in [profiling.scl, profiling.scr, profiling.sc, profiling.cov]:
    X = case.X
    pca = PCA(n_components=X.shape[1]).fit(X)
    case.Xpca = pca.transform(X)
    pca_var = 100 * np.cumsum(pca.explained_variance_ratio_)
    print('Explained variance:', pca_var)
    plt.figure(figsize=(8,6))
    biplot(case.Xpca[:, 0:2], pca.components_[0:2, :].T, list(case.df.columns))
    plt.suptitle(f'{case._name} PCA explained variance: {pca_var[1]:.2f} %');
    plt.show()
    print('_'*100)

# For the SCR dataset, the PCA projection on 2 dimension explains less variability. 
# We could have expected this from the pair plot and lack of strong correlations in the previous section.
# There is still no clear spatial structure or clusters.

# #### TSNE
#
# Since PCA was inconclusive, let us try to project to a manifold using t-SNE. 
# Keep in mind that we cannot use the projected space for anything else (like clustering).
#
#
# For the SCR dataset, there *might* be two modalities, one on the top left and the other one on the bottom right of the projected space.

# +
from sklearn.manifold import TSNE

for case in [profiling.scl, profiling.scr, profiling.sc, profiling.cov]: 
#for case in [profiling.test]: 
    tsne = TSNE(n_components=2, metric='euclidean', 
                verbose=True, random_state=42)
    case.Xtsne = tsne.fit_transform(case.X)
    plt.plot(case.Xtsne[:, 0], case.Xtsne[:, 1], 'o')
    plt.title(f'TSNE {case._name}')
    plt.show()
# -

# The spatial structure of the SCR dataset is less prominent or clear, but there *might* be two groups and many points between them.

# #### Exploratory analysis concluding remarks
#
# Using pair plots and PCA, we learned that:
#
# * The SCL dataset has some strong correlations and these are captured by PCA.
# * A PCA projection in 2 dimensions does not reveals an underlying structure of the datasets.
# * Two PCA projected dimensions is not enough when considering the SCR feature.
# * There does not seem to be a strong reason to PCA-transform our datasets
#
# While on a manifold projection, we learned that:
#
# * The SCL *might* have some non-linear structure, but this could be a *false idea* introduced by tSNE magic.
# * When using SCR, the structure is less clear, yet there may be two main modes with many observations in between.
#

# ### Clustering
#
# Let's try to calculate/extract some clusters from the datasets to see if we can find some spatial structure.

# #### Preliminary verification
#
# Before we go about clustering everything, we have to check a no-go condition here:
# since we are working on high-dimensional data (6 dimensions on COV, 8 dimensions on SCL or SCR, and 16 on the combined case),
# it is important to verify that the data points are not scattered very far away from each other.
#
# Examining the histograms of all pairwise distances, we can see that:
# * There is no red flag on the distances between points; even when we added more features, the points are not exaggerately far away.



# +
from scipy.spatial.distance import pdist, squareform
from scipy.stats import skew, kurtosis

fig, axs = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(10,12))

for k, case in enumerate([profiling.scl, profiling.scr, profiling.sc, profiling.cov]):
    case.dist = pdist(case.X, metric='euclidean')
    axs[k].hist(case.dist, bins=100)
    axs[k].set_title(f'{case._name} distances')


# -



# #### Affinity propagation
#
# I decided to use affinity propagation clustering because the scikit-learn documentation has the following 
# manifest on clustering analysis:
#
# <table border="1" class="colwidths-given docutils">
# <colgroup>
# <col width="15%">
# <col width="16%">
# <col width="20%">
# <col width="27%">
# <col width="22%">
# </colgroup>
# <thead valign="bottom">
# <tr class="row-odd"><th class="head">Method name</th>
# <th class="head">Parameters</th>
# <th class="head">Scalability</th>
# <th class="head">Usecase</th>
# <th class="head">Geometry (metric used)</th>
# </tr>
# </thead>
# <tbody valign="top">
# <tr class="row-even"><td><a class="reference internal" href="#k-means"><span class="std std-ref">K-Means</span></a></td>
# <td>number of clusters</td>
# <td>Very large <code class="docutils literal"><span class="pre">n_samples</span></code>, medium <code class="docutils literal"><span class="pre">n_clusters</span></code> with
# <a class="reference internal" href="#mini-batch-kmeans"><span class="std std-ref">MiniBatch code</span></a></td>
# <td>General-purpose, even cluster size, flat geometry, not too many clusters</td>
# <td>Distances between points</td>
# </tr>
# <tr class="row-odd"><td><a class="reference internal" href="#affinity-propagation"><span class="std std-ref">Affinity propagation</span></a></td>
# <td>damping, sample preference</td>
# <td>Not scalable with n_samples</td>
# <td>Many clusters, uneven cluster size, non-flat geometry</td>
# <td>Graph distance (e.g. nearest-neighbor graph)</td>
# </tr>
# <tr class="row-even"><td><a class="reference internal" href="#mean-shift"><span class="std std-ref">Mean-shift</span></a></td>
# <td>bandwidth</td>
# <td>Not scalable with <code class="docutils literal"><span class="pre">n_samples</span></code></td>
# <td>Many clusters, uneven cluster size, non-flat geometry</td>
# <td>Distances between points</td>
# </tr>
# <tr class="row-odd"><td><a class="reference internal" href="#spectral-clustering"><span class="std std-ref">Spectral clustering</span></a></td>
# <td>number of clusters</td>
# <td>Medium <code class="docutils literal"><span class="pre">n_samples</span></code>, small <code class="docutils literal"><span class="pre">n_clusters</span></code></td>
# <td>Few clusters, even cluster size, non-flat geometry</td>
# <td>Graph distance (e.g. nearest-neighbor graph)</td>
# </tr>
# <tr class="row-even"><td><a class="reference internal" href="#hierarchical-clustering"><span class="std std-ref">Ward hierarchical clustering</span></a></td>
# <td>number of clusters or distance threshold</td>
# <td>Large <code class="docutils literal"><span class="pre">n_samples</span></code> and <code class="docutils literal"><span class="pre">n_clusters</span></code></td>
# <td>Many clusters, possibly connectivity constraints</td>
# <td>Distances between points</td>
# </tr>
# <tr class="row-odd"><td><a class="reference internal" href="#hierarchical-clustering"><span class="std std-ref">Agglomerative clustering</span></a></td>
# <td>number of clusters or distance threshold, linkage type, distance</td>
# <td>Large <code class="docutils literal"><span class="pre">n_samples</span></code> and <code class="docutils literal"><span class="pre">n_clusters</span></code></td>
# <td>Many clusters, possibly connectivity constraints, non Euclidean
# distances</td>
# <td>Any pairwise distance</td>
# </tr>
# <tr class="row-even"><td><a class="reference internal" href="#dbscan"><span class="std std-ref">DBSCAN</span></a></td>
# <td>neighborhood size</td>
# <td>Very large <code class="docutils literal"><span class="pre">n_samples</span></code>, medium <code class="docutils literal"><span class="pre">n_clusters</span></code></td>
# <td>Non-flat geometry, uneven cluster sizes</td>
# <td>Distances between nearest points</td>
# </tr>
# <tr class="row-odd"><td><a class="reference internal" href="#optics"><span class="std std-ref">OPTICS</span></a></td>
# <td>minimum cluster membership</td>
# <td>Very large <code class="docutils literal"><span class="pre">n_samples</span></code>, large <code class="docutils literal"><span class="pre">n_clusters</span></code></td>
# <td>Non-flat geometry, uneven cluster sizes, variable cluster density</td>
# <td>Distances between points</td>
# </tr>
# <tr class="row-even"><td><a class="reference internal" href="mixture.html#mixture"><span class="std std-ref">Gaussian mixtures</span></a></td>
# <td>many</td>
# <td>Not scalable</td>
# <td>Flat geometry, good for density estimation</td>
# <td>Mahalanobis distances to  centers</td>
# </tr>
# <tr class="row-odd"><td><a class="reference internal" href="#birch"><span class="std std-ref">Birch</span></a></td>
# <td>branching factor, threshold, optional global clusterer.</td>
# <td>Large <code class="docutils literal"><span class="pre">n_clusters</span></code> and <code class="docutils literal"><span class="pre">n_samples</span></code></td>
# <td>Large dataset, outlier removal, data reduction.</td>
# <td>Euclidean distance between points</td>
# </tr>
# </tbody>
# </table>

# Our knowledge on this problem is:
#
# * We do not know how many clusters there are.
# * The number of observations per cluster is probably uneven.
# * No knowledge on the geometry.
#
# Therefore, I decided to use affinity propagation. I discarded K-means because of the unknown number of clusters and uneven cluster size. Mean-shift may be a interesting perspective. All the other ones were discarded for simplicity, lack of time, a to avoid a *use them all* approach.
#
# The following sections will apply the following procedure:
#
# 1. Do a parameter search with cross validation for the sample preference model parameter 
#   (the damping parameter was not included; it was fixed since there were 
#   no convergence warnings with its default value, but this may be worth a reconsideration).
# 2. Plot the parameter search results and select a preference parameter with a good compromise/tradeoff
#   between train/test clustering metric and cluster size.
# 3. Train the model with the whole data and the preference parameter to determine the clusters
#   on the whole dataset.
# 4. Plot the points on projected spaces with the cluster results
# 5. Plot and comment on the distribution of the features of each cluster
#
# The following cell tries to avoid repeating ourselves by creating some functions that will be applied on all datasets.

# +
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import silhouette_score
from sklearn.model_selection import KFold


def poormans_gridsearch(x):
    """Grid search the `preference` parameter of AffinityPropagation"""

    # A poor man's grid search
    preferences_param = np.arange(-500, 0, 10)
    kfold = KFold(n_splits=10, random_state=42, shuffle=True)
    gs_results = []
    success = 0
    fail = 0
    for p in preferences_param:
        for i, (idx_train, idx_test) in enumerate(kfold.split(x)):
            try:
                x_train = x[idx_train]
                x_test = x[idx_test]
                model = AffinityPropagation(preference=p).fit(x_train)
                n_clusters = len(model.cluster_centers_indices_)
                labs_train = model.predict(x_train)
                score_train = silhouette_score(x_train, labs_train, metric='sqeuclidean')
                labs_test = model.predict(x_test)
                score_test = silhouette_score(x_test, labs_test, metric='sqeuclidean')
                gs_results.append({
                    'p': p,
                    'fold': i,
                    'n_clusters': n_clusters,
                    'score_train': score_train,
                    'score_test': score_test,
                })
                success += 1
            except ValueError:
                # fit/predict failed; ignore this particular fold dataset
                fail += 1
                
    if fail:
        print(f'Warning: grid search failed {fail} / {fail + success} times')
    df_gs = pd.DataFrame.from_records(gs_results)
    return df_gs


def plot_poormans(df, ref=None):
    """Plot the summarized results of :py:func:`poormans_gridsearch` """
    df_gs_group = df.groupby('p').agg({'mean', 'std'})
    plt.figure(figsize=(8*2, 6))
    plt.subplot(121)
    plt.plot(df_gs_group.index, df_gs_group[[('n_clusters', 'mean')]], label='# clusters')
    plt.fill_between(df_gs_group.index, 
                     np.ravel(df_gs_group[[('n_clusters', 'mean')]].values - df_gs_group[[('n_clusters', 'std')]].values),
                     np.ravel(df_gs_group[[('n_clusters', 'mean')]].values + df_gs_group[[('n_clusters', 'std')]].values),
                     alpha=0.5)
    if ref is not None:
        plt.axvline(ref, color='r', linestyle=':')
    plt.ylim(0, 10)
    plt.grid()
    plt.title('CV number of clusters')
    plt.legend()


    plt.subplot(122)
    plt.plot(df_gs_group.index, df_gs_group[[('score_train', 'mean')]], label='Train metric')
    plt.fill_between(df_gs_group.index, 
                     np.ravel(df_gs_group[[('score_train', 'mean')]].values - df_gs_group[[('score_train', 'std')]].values),
                     np.ravel(df_gs_group[[('score_train', 'mean')]].values + df_gs_group[[('score_train', 'std')]].values),
                     alpha=0.5)
    plt.plot(df_gs_group.index, df_gs_group[[('score_test', 'mean')]], label='Test metric')
    plt.fill_between(df_gs_group.index, 
                     np.ravel(df_gs_group[[('score_test', 'mean')]].values - df_gs_group[[('score_test', 'std')]].values),
                     np.ravel(df_gs_group[[('score_test', 'mean')]].values + df_gs_group[[('score_test', 'std')]].values),
                     alpha=0.5)
    if ref is not None:
        plt.axvline(ref, color='r', linestyle=':')
    plt.ylim(-1, 1)
    plt.grid()
    plt.title('CV scoring metric')
    plt.legend()
    
    
def train_affinity(x, preference, projections):
    af = AffinityPropagation(preference=preference).fit(x)
    cluster_centers_indices = af.cluster_centers_indices_
    labels = af.labels_
    n_clusters_ = len(cluster_centers_indices)
    counter = collections.Counter(labels)

    print('Estimated number of clusters: %d' % n_clusters_)
    print("Silhouette Coefficient: %0.3f"
          % silhouette_score(x, labels, metric='sqeuclidean'))
    print('Observations per cluster:', counter)
    
    cluster = [f'C{l}' for l in labels]
    cluster_details = [f'C{l} (n={counter[l]})' for l in labels]
    df_result = pd.DataFrame({'cluster': cluster, 'cluster_details': cluster_details})
    # Append columns with the features
    for i in range(x.shape[1]):
        df_result[f'x{i}'] = x[:, i]
        
    # also append columns of the features in a projected space
    if not isinstance(projections, dict):
        projections = {'p': projections}
    
    for k, proj in projections.items():
        for i in range(proj.shape[1]):
            df_result[f'x{k}{i}'] = proj[:, i]
    df_result.loc[:, 'cluster_master'] = False
    df_result.loc[af.cluster_centers_indices_, 'cluster_master'] = True

    return df_result

def post_cluster_merge(df_pre, df_post, df_full):
    assert df_pre.shape[0] == df_post.shape[0]
    df_tmp = df_pre.reset_index(drop=False)
    df_tmp['cluster'] = df_post['cluster']
    df_tmp['cluster_details'] = df_post['cluster_details']
    df_tmp['cluster_master'] = df_post['cluster_master']
    df_final = pd.merge(df_full, df_tmp[['file_id', 'cluster', 'cluster_details', 'cluster_master']], 
                        left_on='file_id', right_on='file_id')
    return df_final


# -

profiling.scl.preference=-400  # 2 clusters, Silhouette Coefficient: 0.685
profiling.scr.preference=-300  # 2 clusters, Silhouette Coefficient: 0.604
profiling.sc.preference=-480  #  2 clusters, Silhouette Coefficient: 0.532    
profiling.cov.preference=-360  #  2 clusters, Silhouette Coefficient: 0.399


for case in [profiling.scl, profiling.scr, profiling.sc, profiling.cov]: 
    case.df_gs = poormans_gridsearch(case.X)
    plot_poormans(case.df_gs, ref=case.preference)
    plt.suptitle(f'Affinity propagation model parameter selection by cross validation for {case._name}');
    plt.show()
    case.df_aff = train_affinity(case.X, case.preference, {'tsne':case.Xtsne, 'pca': case.Xpca})
    df_aff = case.df_aff
    case.df_complete = post_cluster_merge(case.df, case.df_aff, profiling.df_ss)
    df_complete = case.df_complete
    print('_'*100)

# Preference values are chosen to be the best compromise: 
# it is still where there is not train / test difference, the silhouette metric is high, the number of clusters is stable.

case = profiling.scl
df_aff = case.df_aff
df_complete = case.df_complete[case.df_complete.sequence_short.isin(sequences_short)]

# + {"magic_args": "-i df_complete -w 7 -h 3 --units in -r 200", "language": "R"}
#
# tmp = reshape::melt(df_complete,
#     id.vars=c('sequence', 'file_id', 'context', 'sequence_short', 'cluster', 'cluster_details', 'cluster_master'),
#     measure.vars=c('scl'))
#
# ggplot(tmp, aes(x=as.factor(context), y=value, fill=cluster_details)) +
#     geom_boxplot() +
#     geom_point(data=tmp[tmp$cluster_master,], 
#                color=2, size=2, shape=17, show.legend=F) + 
# facet_grid(variable~cluster, scales='free_y') +
# labs(title="Clusters estimated on SCL activations during Space-Stress") + 
# theme(axis.text.x = element_text(angle = 45, hjust = 1), plot.title=element_text(size=15))
#
# -

case = profiling.scr
df_aff = case.df_aff
df_complete = case.df_complete[case.df_complete.sequence_short.isin(sequences_short)]

# + {"magic_args": "-i df_complete -w 7 -h 3 --units in -r 200", "language": "R"}
#
# tmp = reshape::melt(df_complete,
#     id.vars=c('sequence', 'file_id', 'context', 'sequence_short', 'cluster', 'cluster_details', 'cluster_master'),
#     measure.vars=c('scr'))
#
# ggplot(tmp, aes(x=as.factor(context), y=value, fill=cluster_details)) +
#     geom_boxplot() +
#     geom_point(data=tmp[tmp$cluster_master,], 
#                color=2, size=2, shape=17, show.legend=F) + 
# facet_grid(variable~cluster, scales='free_y') +
# labs(title="Clusters estimated on SCR activations during Space-Stress") + 
# theme(axis.text.x = element_text(angle = 45, hjust = 1), plot.title=element_text(size=15))
#
# -

case = profiling.sc
df_aff = case.df_aff
df_complete = case.df_complete[case.df_complete.sequence_short.isin(sequences_short)]

# + {"magic_args": "-i df_aff  -w 5 -h 3 --units in -r 200", "language": "R"}
#
# g1 = ggplot(df_aff, aes(x=xtsne0, y=xtsne1, color=cluster)) +
#     geom_point() +
#     theme(legend.position='top')
#     
# g2 = ggplot(df_aff, aes(x=xpca0, y=xpca1, color=cluster)) + 
#     geom_point() +
#     theme(legend.position='top')
#
# cowplot::plot_grid(g1, g2, nrow=1)

# + {"magic_args": "-i df_complete -w 9 -h 5 --units in -r 200", "language": "R"}
#
# tmp = reshape::melt(df_complete,
#     id.vars=c('sequence', 'file_id', 'context', 'sequence_short', 'cluster', 'cluster_details', 'cluster_master'),
#     measure.vars=c('scl', 'scr'))
#
# ggplot(tmp, aes(x=as.factor(context), y=value, fill=cluster_details)) +
#     geom_boxplot() +
#     geom_point(data=tmp[tmp$cluster_master,], 
#                color=2, size=2, shape=17, show.legend=F) + 
# facet_grid(variable~cluster, scales='free_y') +
# labs(title="Clusters estimated on both SCR and SCL activations during Space-Stress") + 
# theme(axis.text.x = element_text(angle = 45, hjust = 1), plot.title=element_text(size=15))
#
# -

case = profiling.cov
df_aff = case.df_aff
df_complete = case.df_complete[case.df_complete.sequence_short.isin(sequences_short)]

# + {"magic_args": "-i df_aff  -w 5 -h 3 --units in -r 200", "language": "R"}
#
# g1 = ggplot(df_aff, aes(x=xtsne0, y=xtsne1, color=cluster)) +
#     geom_point() +
#     theme(legend.position='top')
#     
# g2 = ggplot(df_aff, aes(x=xpca0, y=xpca1, color=cluster)) + 
#     geom_point() +
#     theme(legend.position='top') 
# theme(plot.title=element_text(size=15))
# cowplot::plot_grid(g1, g2, nrow=1) 
# -

tmp = pd.concat([case.df_aff, case.df.reset_index()], axis=1)
tmp = pd.melt(tmp, id_vars=['file_id', 'cluster', 'cluster_details', 'cluster_master'], 
value_vars=['scr_scr', 'scr_scl', 'scr_difficulty', 'scl_scl', 'scl_difficulty',
      'difficulty_difficulty'], var_name='cov')

# + {"magic_args": "-i tmp -w 10 -h 5 --units in -r 200", "language": "R"}
# ggplot(tmp, aes(x=cov, y=value, fill=cluster_details)) +
#     geom_boxplot()+
# labs(title="Clusters estimated on covariances between activations and context during Space-Stress") + 
# theme(plot.title=element_text(size=15))
#
# -

# ## Conclusions
#
# **On the activation model** <br>
# I think the method is well appropriate to the DS request. <br>
# The cross-validation seems robust and the distribution, at population level across modules is comforting that we did not overfit. <br>
# <div class="alert alert-info">
# Main result is that in 75% of the observations, phasic activation (resp decompression) goes togeteher with tonic activation (resp. decompression).
# We could not identify, at population level, a clear relationship between activations degree during space-stress waves and the difficulty. 
# </div>
#
# That being said, I reflect upon:
# - the choice of the training sequences, insofar as the survey during space-stress still show activations greater than 0.5. Which leads me to a second wondering: what is the *dynamic* of GSR? ie. Is it appropriate to train on sequences outside space-stress and test on space-stress or is their some *rebound effects*?
# - the choice of the baseline-correction, insofar as we train on sequences that have been used to correct the features. 
# - the features quality, insofar as we obtain 'prettier' clusters by selecting only scl_median and scr_rate than by considering all features. Are the other features just adding noise, or is it the scaling method that is not appropriate? 
#
# **On the clustering** <br>
# I am not convinced by the resulting clustering. The result is as follow: 
# - for scr and scl  datasets, it seems that their is a 'compact mass' (~75% of the populations)  that shows the following activations: high during waves, low elsewhere. 
# - for sc dataset, I don't see any difference between the two clusters. 
# - for cov dataset, we draw two caricatural clusters: good adaptation between activation and difficulty, and no adaptation. 
#
# The reason why I am not fully convinced is because the explained variance of PCA are quitee low, the affinity model cv scorig metric also, and that the number of cluster (N=2) is very few, when the 'affinity propagation' is rather used for greater amount of clusters. 
#
# <div class="alert alert-info">
# I am disapointed that the clustering accuracy and results are worse than the with the first approach (see gsr-exploration.ipynb by DOjeda), that only included SCL_median and SCR_rate as activation features. 
# </div>
#
#
# ## Perspectives
#
# - re-run the analysis using features that have not been baseline-corrected and choose an other baseline (eg. the whole session?) 
# - choice of clustering: good thing with affinity propagation is that there is no fixed number of clusters and no assumption on their size. That beeing said, they seem to be more suitable for greater amount of clusters. When only 2 cluster can be identified, maybe we should reconsider the method? 
# - scaling of the activation vector should be reviewed. 
# - choice of the sequences to train the activation model should be reviewed. 
#
# ## Reproducibility
#
# This final cell is useful to keep track of the exact environment and packages that you used

# +
import sys
import os

print(sys.version)
is_conda = os.path.exists(os.path.join(sys.prefix, 'conda-meta'))
if is_conda:
    %conda list
else:
    %pip list
