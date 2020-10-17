# AUTOGENERATED! DO NOT EDIT! File to edit: nbs/02_annotate.ipynb (unless otherwise specified).

__all__ = ['load_df', 'load_completions', 'get_og_filepath', 'anno_sample_merge', 'nnAnnotations']

# Cell
from tqdm.notebook import trange, tqdm
import pandas as pd
from pandas import json_normalize
import json
import requests
import re
from glob import glob
from pathlib import Path

# Cell
def load_df(json_file):
    with open(json_file) as f:
        data = json.load(f)
        df = json_normalize(data,record_path=['completions'],meta=['data'])
       # df['result'] = df['result'].apply(lambda x: return_choice(x[0]) if len([x][0]) ==1 else x)
        df['result'] = df['result'].apply(lambda x: x[0]['value']['choices'] if len([x][0]) ==1 else x)
        return df

# Cell
def load_completions(path):
    filenames = glob(f'{path}/completions/*.json')
    dataframes = [load_df(f) for f in filenames]
    return pd.concat(dataframes)

# Cell
def _df_to_csv(df,out_fn):
    df[['data','result']].to_csv(out_fn,header=['file','label',],index=False)

# Cell
def _df_to_json(df,out_fn):
    df[['data','value.choices']].to_json(out_fn)

# Cell
def _df_to_pkl(df,out_fn):
    df.to_pickle(out_fn)

# Cell
def get_og_filepath(x):
    """
    Transforms a filepaths from processed ImageStudio format back to the Orginal Newspaper Navigator filepath  format
    """
    b, m, e = re.split('(_data_)',x)
    m = m.replace('_','/')
    e = re.split('(\d{3}_\d{1}_\d{2}.jpg)',e)
    return b+m+e[0].replace('_','/') +e[1]

# Cell
def anno_sample_merge(sample_df: pd.DataFrame, annotation_df: pd.DataFrame) -> pd.DataFrame:
    """anno_sample_merge merges a DataFrame containing a sample
    from Newspaper Navigator and a DataFrame containing annotations

    Parameters
    ----------
    sample_df : pd.DataFrame
        A Pandas DataFrame which holds a sample from Newspaper Navigator Generated by `sample.nnSample()`
    annotation_df : pd.DataFrame
        A pandas DataFrame containing annotations loaded via the `annotate.nnAnnotations` class


    Returns
    -------
    pd.DataFrame
        A new DataFrame which merges the two input DataFrames
    """
    sample_df, annotation_df = sample_df.copy(), annotation_df.copy()
    annotation_df['id'] = annotation_df['data'].map(lambda x:get_og_filepath(x))
    return sample_df.merge(annotation_df, left_on='filepath',right_on='id')

# Cell

class nnAnnotations:
    def __init__(self, df):
        self.annotation_df = df
        self.labels = df['result'].unique()
        self.label_counts = df['result'].value_counts()

    def __repr__(self):
        return (f'{self.__class__.__name__}'
                f' #annotations:{len(self.annotation_df)}')

    @classmethod
    def from_completions(cls, path, kind, drop_dupes=True, sample_df=None):
        df = load_completions(path)
        df = df.reset_index(drop=True) # add index
        df['data']= df['data'].map(lambda x: x['image'])
        df['data'] = df['data'].map(lambda x: x.split('?')[0])
        df['data'] = df['data'].apply(lambda x: Path(x).name)
        if any(df['data'].str.contains('-')): # removes labelstudio hash from data loaded via web interface
            df['data'] = df['data'].str.split('-',expand=True)[1]
        if drop_dupes:
            df = df.drop_duplicates(subset='data',keep='last')
        if kind=='classification':
            empty_rows = df[df['result'].apply(lambda x:len(x)==0)].index
            df = df.drop(empty_rows)
            df['result'] = df['result'].map(lambda x: x[0])
        if kind=='label':
            df['result'] = df['result'].map(lambda x: "|".join(map(str,x)) if len(x) >=1 else x)
            df['result'] = df['result'].map(lambda x:"" if len(x)==0 else x)
        return cls(df)

    def merge_sample(self, sample_df):
        self.merged_df = anno_sample_merge(sample_df,self.annotation_df)

    def export_merged(self, out_fn):
        self.merged_df.to_csv(out_fn)

    def export_annotations(self, out_fn):
        df = self.annotation_df
        if not Path(out_fn).exists():
            Path(out_fn).touch()
        suffix = Path(out_fn).suffix
        if suffix == '.csv':
            _df_to_csv(df, out_fn)
        if suffix == '.json':
            _df_to_json(df,out_fn)
        if suffix == '.pkl':
            _df_to_pkl(df,out_fn)