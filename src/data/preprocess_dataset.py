# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# %%
DATA_ROOT = '../../data/raw'

# %% [markdown]
# ## LOADING DATA

# %%
print('Loading raw datasets...', flush=True)
GIT_COMMITS_PATH = f"{DATA_ROOT}/GIT_COMMITS.csv"
GIT_COMMITS_CHANGES = f"{DATA_ROOT}/GIT_COMMITS_CHANGES.csv"
SONAR_MEASURES_PATH = f"{DATA_ROOT}/SONAR_MEASURES.csv"
SZZ_FAULT_INDUCING_COMMITS = f"{DATA_ROOT}/SZZ_FAULT_INDUCING_COMMITS.csv"
JIRA_ISSUES = f"{DATA_ROOT}/JIRA_ISSUES.csv"


# %%
git_commits = pd.read_csv(GIT_COMMITS_PATH)
git_commits_changes = pd.read_csv(GIT_COMMITS_CHANGES)
sonar_measures = pd.read_csv(SONAR_MEASURES_PATH)
szz_fault_inducing_commits = pd.read_csv(SZZ_FAULT_INDUCING_COMMITS)
jira_issues = pd.read_csv(JIRA_ISSUES)


# %%
git_commits_changes[git_commits_changes['linesAdded'].isna()]


# %%
len(git_commits_changes.commitHash.unique())

# %% [markdown]
# ## FILTERING COLUMNS
print('Filtering columns...', flush=True)
# %% [markdown]
# -------------------------------------------------------------------------------------------------------------------------------

# %%
git_dates = git_commits[['commitHash','committerDate']]


# %%
agg = {
    'linesAdded': ['sum'],
    'linesRemoved': ['sum'],
    'projectID': ['count'],
}
gcg_by_commit = git_commits_changes.groupby(['projectID', 'commitHash']).agg(agg)


# %%
len(gcg_by_commit)


# %%
gcg_by_commit = gcg_by_commit.reset_index()


# %%
gcg_by_commit.columns = ['projectID', 'commitHash', 'lines_added', 'lines_removed', 'entropylike']


# %%
gcg_by_commit = pd.merge(gcg_by_commit, git_dates, on='commitHash', how='inner')


# %%
gcg_by_commit = gcg_by_commit.sort_values(by=['projectID', 'committerDate'])


# %%
print('Computing metrics...', flush=True)
total_lines = []
project = 'accumulo'
la_counter = 0
lr_counter = 0
for i, row in gcg_by_commit.iterrows():
  if project!=row['projectID']:
    project=row['projectID']
    la_counter = 0
    lr_counter = 0
  la_counter+=row['lines_added']
  lr_counter+=row['lines_removed']
  total_lines.append(la_counter-lr_counter)


gcg_by_commit['total_lines'] = total_lines


# %%
gcg_by_commit = gcg_by_commit[gcg_by_commit['total_lines']>=0] #to avoid 2 lines of wrong data in te commons-cli project


# %%
gcg_by_commit['added/total_lines'] = gcg_by_commit['lines_added']/gcg_by_commit['total_lines']


# %%
gcg_by_commit = gcg_by_commit[gcg_by_commit['added/total_lines']<=1] #to avoid 1 line of wrong data in commons-cli project 


# %%
gcg_by_commit = gcg_by_commit[['commitHash', 'entropylike', 'added/total_lines']]


# %%
jira_bugs = jira_issues[jira_issues['type'] == 'Bug']
jira_bugs = jira_bugs[['key', 'priority']]


# %%
print('Merging datasets...', flush=True)
szz_fault_inducing_commits = szz_fault_inducing_commits[['faultInducingCommitHash', 'key']]
szz_fault_inducing_commits = szz_fault_inducing_commits.rename(columns={'faultInducingCommitHash':'commitHash'})
szz_fault_inducing_commits.head()


# %%
Y = pd.merge(szz_fault_inducing_commits, jira_bugs, on='key')


# %%
def priorityToCategory(p: str):
    """
    """
    if p == 'No bug': return 0
    if p == 'Trivial': return 1
    if p == 'Minor': return 2
    if p == 'Blocker': return 3
    if p == 'Major': return 4
    if p == 'Critical': return 5


Y['priority'] = Y['priority'].apply(lambda p: priorityToCategory(p))


# %%
Y = Y[['commitHash', 'priority']]


# %%
multitarget = True #in case we are predicting multiple bugs for each commit


# %%
if not multitarget:
  Y = Y.sort_values(by='commitHash')
  Y = Y.groupby('commitHash').max().reset_index() #otherwise, we predict the one with highest priority


# %%
git_commits = git_commits[['commitHash', 'inMainBranch',	'merge']]


# %%
sonar_measures.drop(['projectID', 'SQAnalysisDate', 'functionComplexityDistribution', 'fileComplexityDistribution',                      'lastCommitDate', 'nclocLanguageDistribution', 'alertStatus', 'qualityGateDetails', 'qualityProfiles', 'files'], axis=1, inplace=True)


# %%
X = pd.merge(git_commits, sonar_measures, how='inner', on='commitHash')


# %%
X2 = pd.merge(X, gcg_by_commit, on='commitHash', how='inner')


# %%
df = pd.merge(X2, Y, on='commitHash', how='left')


# %%
df['priority'] = df['priority'].fillna(0)


# %%
df = df.fillna(df.mean()) #just for one of the multilables


# %%
print('Storing processed dataset into ../../data/processed/', flush=True)
if multitarget:
  df.to_csv('../../data/processed/bugs-multitarget.csv', index=False)
else:
  # df.to_csv('../data/processed/bugs-singletarget_with_mean.csv', index=False)
  pass


# %%



