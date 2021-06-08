# general libraries
import numpy as np
import pandas as pd
import csv, urllib.request
from functools import partial, cached_property
from dataclasses import dataclass
from typing import List, Any, Callable, Dict, Optional, Union
import logging
import datetime
import json
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

fp = '/home/chansoo/data/nyc_contributions'

class nyccfb:

    def __init__(self, initialize:bool = False):
        self.initialize = initialize

    @cached_property
    def files_list(self):
        page = requests.get("http://www.nyccfb.info/follow-the-money/data-library/")
        soup = BeautifulSoup(page.content, 'html.parser')
        return soup.find_all('td')

    @cached_property
    def urls(self):
        url_list = []
        for x in self.files_list:
            try:
                new_file = x.find('a')['href']
                if '.csv' in new_file:
                    url_list.append('http://www.nyccfb.info'+new_file)
            except:
                continue
        return url_list
        
    @property
    def data(self):
        # initialize dictionary
        #keys = ['contribution', 'expenditure', 'intermediar', 'payment', 'financial']        
        keys = ['contribution']
        all_dfs = {}
        for key in keys:
            all_dfs[key] = []

        for url in tqdm(self.urls):
            for key in keys:
                if (key in url.lower()) & ('key' not in url.lower()):
                    df = self.url_to_df(url)
                    all_dfs[key].append(df)

        return all_dfs

    def url_to_df(self, url):
        
        response = urllib.request.urlopen(url)
        
        lines = [
            l.decode('utf-8', errors="ignore") 
            for l in response.readlines()
        ]

        df = pd.DataFrame(
                csv.reader((line.replace('\0','') for line in lines), delimiter=",")
            )

        df.columns = df.loc[0]
        df = (
            df
            .drop(0, axis=0)
            .reset_index(drop=True)
            .assign(source=url)
            )

        return df

    def drop_null_cols(self,df_list: List[pd.DataFrame]) -> List[pd.DataFrame]:
        return [
            df[[x for x in df.columns if x != '']] 
            for df in df_list
        ]
    
    @property
    def contributions(self):
        conts = pd.concat(
            self.drop_null_cols(self.data['contribution']),
            axis=0
            )

        # fix names
        recip_notnull = conts['RECIPID'].notnull()
        conts.loc[recip_notnull,'CANDID'] = conts.loc[recip_notnull,'RECIPID']
        conts = conts.drop('RECIPID', axis=1)

        recip_name_null = conts['RECIPNAME'].isnull()
        candname = conts['CANDLAST'] + ', ' + conts['CANDFIRST'] + conts['CANDMI']
        conts.loc[recip_name_null, ['RECIPNAME']] = candname[recip_name_null]
        conts = conts.rename({'RECIPNAME':'CANDNAME'},axis=1)

        # Convert Dates 
        conts['DATE'] = pd.to_datetime(conts['DATE'], errors='coerce')        

        return conts

    @property
    def expenditures(self):
        expen = pd.concat(
            self.drop_null_cols(self.data['expenditure']),
            axis=0
            )

        expen['DATE'] = pd.to_datetime(expen['DATE'])
        expen['CANDNAME'] = expen['CANDLAST'] + ', ' + expen['CANDFIRST'] + expen['CANDMI']
        return expen

    @property
    def candidates(self):
        cand_cols = ["OFFICECD","CANCLASS","CANDNAME","COMMITTEE"]

        cands_list = [
            self.contributions[cand_cols].drop_duplicates(),
            self.expenditures[cand_cols].drop_duplicates()
        ]

        df_cands = (
            pd.concat(cands_list)
            .drop_duplicates()
            .reset_index(drop=True)
            )
        df_cands['candidate_id'] = ['nyc_ftm_' + str(x) for x in df_cands.index]

        return df_cands

    @property
    def intermediaries(self):
        return pd.concat(
                self.drop_null_cols(self.data['intermediar']),
                axis=0
            )

    def fix_payment_cols(self, df):
        payment_cols = [
            'ELECTION', 'CANDID', 'CANDNAME', 'OFFICECD', 'OFFICEBORO',
            'OFFICEDIST', 'CANCLASS', 'PRIMARYPAY', 'GENERALPAY', 
            'RUNOFFPAY','TOTALPAY', 'source'
        ]
        if 'ELECTION' not in df.columns:
            row1 = pd.Series(df.columns)
            df.columns = payment_cols
            df = pd.concat([df, row1], axis=0).drop(0,axis=1)

        return df

    @property
    def payments(self):

        payments = pd.concat(
            [
                self.fix_payment_cols(df)
                for df in self.data['payment']
            ],
            axis=0
            )

        return payments
    

    def save(self):
        
        self.contributions.to_csv(f'{fp}/contributions.csv', index=False)

        # tables = [
        #     self.candidates,
        #     self.expenditures,
        #     self.contributions,
        #     self.intermediaries,
        #     self.payments
        # ]

        # for df in tables:
        #     df.to_csv(, index=False)