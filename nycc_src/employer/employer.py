import pandas as pd
import re
from dataclasses import dataclass
from functools import cached_property
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def normalize(x):
    return (x - min(x)) / (max(x) - min(x))

def cleanname(s):
    return (
        s
        .astype(str)
        .apply(lambda x: re.sub(r'[^A-Za-z0-9 ]+', '', x))
        .str.strip()
        .str.lower()
    )

@dataclass
class heatmap:
    df: pd.DataFrame
    value: str = 'AMNT'

    @cached_property
    def dfemp(self):
        return (
            self.df
            .groupby(['EMPNAME','CANDNAME'])
            .agg({
                'id':'count',
                'AMNT':'sum'
            })
            .rename({
                'id':'count'
            }, axis=1)
            .reset_index()
        )

    @cached_property
    def dfpivot(self):
        employers = self.dfemp['EMPNAME'].value_counts()
        names = employers[employers>8].index

        self.dfemp = self.dfemp.loc[self.dfemp['EMPNAME'].isin(names)].copy()
        dfpivot = self.dfemp.pivot(index='EMPNAME', columns='CANDNAME', values=self.value)
        dfpivot = dfpivot.fillna(0)
        
        for row in dfpivot.index:
            dfpivot.loc[row] = normalize(dfpivot.loc[row])
        
        return dfpivot

    @cached_property
    def dfcands(self):
        # sort candidates
        pca = PCA(n_components=2)
        x = pca.fit_transform(np.array(self.dfpivot).T)
        dfcands = self.dfpivot.T
        dfcands['pca'] = x[:,0]
        return dfcands

    @cached_property
    def candidates_ordered(self):
        return ['McGuire, Raymond J', 'Yang, Andrew', 'Adams, Eric L',
       'Garcia, Kathryn A', 'Wiley, Maya D', 'Morales, Dianne',
       'Donovan, Shaun', 'Stringer, Scott M']
        # return (
        #     self.dfcands
        #     .sort_values('pca', ascending=False)
        #     .head(8)
        #     .index
        #     )

    @cached_property
    def df_heatmap(self):
        return (
            self.dfpivot[list(self.candidates_ordered)+['pca']]
            .sort_values('pca')
            .drop('pca', axis=1)
            )

    def heatmap(self):
        
        # sort employers
        pca = PCA(n_components=2)
        x = pca.fit_transform(np.array(self.dfpivot))
        self.dfpivot['pca'] = x[:,0]

        sns.set(rc={'figure.figsize':(10,20)})
        ax = sns.heatmap(
            self.df_heatmap,
            cbar=False
        )

        ax.figure.savefig(f'chansoo/employer_heatmap_{self.value}.png', bbox_inches="tight")

        return ax