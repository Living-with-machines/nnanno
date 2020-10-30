# nnanno
> Sampling, annotating and making predictions on the Newspaper Navigator dataset  


What is Newspaper Navigator?

## Install

`pip install nnanno`

## Programming Historian Data Preparation 

## How to use

Fill me in please! Don't forget code examples:

```
get_json_url(1850, 'ads')
```




    'https://news-navigator.labs.loc.gov/prepackaged/1850_ads.json'



```
sampler = nnSampler()
df = sampler.create_sample(2, end_year=1851)
```

    


```
df['page_url'].head(1)
```




    0    https://chroniclingamerica.loc.gov/data/batches/vtu_londonderry_ver01/data/sn84023252/00200296205/1850033001/0259.jp2
    Name: page_url, dtype: category
    Categories (2, object): ['https://chroniclingamerica.loc.gov/data/batches/vtu_londonderry_ver01/data/sn84023252/00200296205/1850030201/0243.jp2', 'https://chroniclingamerica.loc.gov/data/batches/vtu_londonderry_ver01/data/sn84023252/00200296205/1850033001/0259.jp2']



## Annotate samples

# Inference

```
from fastai.vision.all import *
dls = ImageDataLoaders.from_csv('../ph/ads/', 'ads_upsampled.csv',folder='images', fn_col='file', label_col='label',
                                item_tfms=Resize(64,ResizeMethod.Squish), num_workers=0)
learn = cnn_learner(dls, resnet18, metrics=F1Score())
learn.fine_tune(1)
```

```
predictor = nnPredict(learn,try_gpu=False)
```

```
predictor.predict_sample('ads','testinference',0.01,end_year=1850)
```

    


```
df = pd.read_json('testinference/1850.json')
df.iloc[:5,-3:]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>pred_decoded</th>
      <th>illustrations_prob</th>
      <th>text-only_prob</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>0.179449</td>
      <td>0.820551</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.212609</td>
      <td>0.787392</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0.001641</td>
      <td>0.998359</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0.046870</td>
      <td>0.953130</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>0.999221</td>
      <td>0.000779</td>
    </tr>
  </tbody>
</table>
</div>


