import pandas as pd
import re
from dataclasses import dataclass
from functools import cached_property
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from fuzzywuzzy import process, fuzz
import networkx as nx
from tqdm import tqdm

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
        
        exclude = [
            'nan', 'selfemployed', 'self employed', 'owner', 'not employed', 'unemployed', 'retired', 'none', 'na', 
            ]

        names = self.dfemp.loc[
                (self.dfemp['count']>20) & (~self.dfemp['EMPNAME'].isin(exclude)),
                'EMPNAME'
            ].unique()

        replace = [
            ['doe', 'department of education']
        ]

        for x,y in replace:
            names = [name.replace(x,y) for name in names]

        fuzzymatches = self.get_fuzz_matches(names)

        for candidates in fuzzymatches:
            canonical = list(candidates)[0]
            self.dfemp['EMPNAME'] = [canonical if name in list(candidates) else name for name in self.dfemp['EMPNAME']]
            names = [canonical if name in list(candidates) else name for name in names]
        
        self.dfemp = self.dfemp.groupby(['EMPNAME','CANDNAME']).agg({'count':'sum', 'AMNT':'sum'}).reset_index()

        self.dfemp = self.dfemp.loc[self.dfemp['EMPNAME'].isin(names)].copy()
        dfpivot = self.dfemp.pivot(index='EMPNAME', columns='CANDNAME', values=self.value)
        dfpivot = dfpivot.fillna(0)
        
        for row in dfpivot.index:
            dfpivot.loc[row] = normalize(dfpivot.loc[row])
        
        return dfpivot

    def get_fuzz_matches(self, names):

        G = nx.Graph()

        for name in tqdm(names):
            results = process.extract(name, names, scorer=fuzz.token_sort_ratio)[1:]
            matches = [x[0] for x in results if x[1] > 80]
            for match in matches:
                G.add_edge(name, match)

        print(list(nx.connected_components(G)))
        return list(nx.connected_components(G))

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
    #     return ['McGuire, Raymond J', 'Yang, Andrew', 'Adams, Eric L',
    #    'Garcia, Kathryn A', 'Wiley, Maya D', 'Morales, Dianne',
    #    'Donovan, Shaun', 'Stringer, Scott M']
        return (
            self.dfcands
            .drop('pca',axis=0)
            .sort_values('pca', ascending=False)
            .index
            )

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

        if self.value == "AMNT":
            self.df_heatmap = self.df_heatmap.iloc[:,::-1]
        sns.set(rc={'figure.figsize':(10,20)})
        ax = sns.heatmap(
            self.df_heatmap,
            cbar=False
        )

        ax.figure.savefig(f'chansoo/employer_heatmap_{self.value}.png', bbox_inches="tight")

        return ax