# nnanno
> Sampling, annotating and making predictions on the Newspaper Navigator dataset  


![CI](https://github.com/davanstrien/nnanno/workflows/CI/badge.svg)

`nnanno` is a modest collection of tools to help work with the delightful [Newspaper Navigator](https://news-navigator.labs.loc.gov/) data. [Newspaper Navigator](https://news-navigator.labs.loc.gov/) is a project which extracted visual content (pictures, maps etc.) from the Library of Congress [Chronicling America](https://chroniclingamerica.loc.gov/) digitised newspaper collection. 

Newspaper Navigator has released data in a number of formats including `json` files. `nnanno` was thrown together to help work with this collection as part of the preparation of some example datasets used in a series of Programming Historian tutorials. Since this code was developed using the wonderful [nbdev](nbdev.fast.ai/) library it is possible to install it for your own use. 

## What nnanno tries to help with

### Sampling 

### nbdev notes


```
1+1
```




    2



### Disclaimer

This code was mainly written to help develop some example datasets for a series of Programming Historian tutorials. It has some tests but there are likely bugs and issues with the code. The code in this repository was all written in notebooks, some people  hate notebooks. Those people will probably hate this too 🤷‍♂️

This project is not an official Library of Congress project. 

## Install

At the moment installation is through Git. If the code gets a few more eyes on it then it may get uploaded to pip. 

`pip install nnanno` #TODO add github link

## Programming Historian Data Preparation 

## How to use

Fill me in please! Don't forget code examples:

## Creating samples

## annotation

## inference

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




    0    https://chroniclingamerica.loc.gov/data/batches/ohi_ingstad_ver01/data/sn85026051/00296027029/1850122101/0124.jp2
    Name: page_url, dtype: category
    Categories (2, object): ['https://chroniclingamerica.loc.gov/data/batches/ohi_ingstad_ver01/data/sn85026051/00296027029/1850072001/0033.jp2', 'https://chroniclingamerica.loc.gov/data/batches/ohi_ingstad_ver01/data/sn85026051/00296027029/1850122101/0124.jp2']



## Annotate samples

# Inference

```
from fastai.vision.all import *
dls = ImageDataLoaders.from_csv('../ph/ads/', 'ads_upsampled.csv',folder='images', fn_col='file', label_col='label',
                                item_tfms=Resize(64,ResizeMethod.Squish), num_workers=0)
learn = cnn_learner(dls, resnet18, metrics=F1Score())
learn.fine_tune(1)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>f1_score</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>1.003417</td>
      <td>0.810943</td>
      <td>0.779661</td>
      <td>00:11</td>
    </tr>
  </tbody>
</table>



<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: left;">
      <th>epoch</th>
      <th>train_loss</th>
      <th>valid_loss</th>
      <th>f1_score</th>
      <th>time</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.608463</td>
      <td>0.514722</td>
      <td>0.789116</td>
      <td>00:15</td>
    </tr>
  </tbody>
</table>


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
