# nnanno
> Sampling, annotating and making predictions on the Newspaper Navigator dataset  


![CI](https://github.com/davanstrien/nnanno/workflows/CI/badge.svg)

`nnanno` is a modest collection of tools to help work with the delightful [Newspaper Navigator](https://news-navigator.labs.loc.gov/) data. 

[Newspaper Navigator](https://news-navigator.labs.loc.gov/) is a project which extracted visual content (pictures, maps etc.) from the Library of Congress [Chronicling America](https://chroniclingamerica.loc.gov/) digitised newspaper collection. 

Newspaper Navigator has released data in a number of formats including `json` files which contain a range of metadata about the newspaper from which the image was taken from. `nnanno` was thrown together to help work with this collection as part of the preparation of some example datasets used in a series of Programming Historian tutorials. Since this code was developed using the wonderful [nbdev](nbdev.fast.ai/) library it is possible to install it for your own use. 

## What nnanno tries to help with

nnanno doesn't to provide an end-to-end to end 'pipeline' for using machine learning with the Newspaper Navigator data since people wanting to work with data will have different needs and interests. Instead it is a minimal collection of code that *may* help you if you want to work with the Newspaper Navigator data.

There are three particular areas where nnanno tries to help a little:
- sampling from the full Newspaper Navigator data
- annotating this sample with additional labels using [label studio](https://labelstud.io)
- inference (experimental 😬) running inference i.e making new predictions with a machine learning model on the newspaper navigator dataset using IIIF.

#### Minimal(ish) computing \ # TODO


### nbdev notes
This code was written using a tool called `nbdev`. This is a tool that helps write python libraries inside Jupyter notebooks. 
Inside the documentation you will see code cells followed by output. This is generated from a Jupyter notebook and shows the actual output of the code rather than something that has been copied and pasted for example:

```python
import datetime
print(datetime.date.today())
```

    2021-01-22


Is evaluated as Python code.  This also means all of the documentation and examples can be opened in notebooks and the code inspected, changed and run. 

### Disclaimer

This code was mainly written to help develop some example datasets for a series of Programming Historian tutorials. It has some tests but there are likely bugs and issues with the code. The code in this repository was all written in notebooks, some people  hate notebooks. Those people will probably hate this too 🤷‍♂️

## Install

At the moment installation is through Git. If the code gets a few more eyes on it then it may get uploaded to pip. 

`pip install nnanno` #TODO add github link

## Programming Historian Data Preparation 

\ # TODO add link to prep notebooks

## How to use (tl;dr)
The three main areas of nnanno are shown below. The examples section in the documentation shows this in greater detail.

### Creating samples

```python
from nnanno.sample import *
```

```python
sampler = nnSampler()
df = sampler.create_sample(1,'photos',start_year=1850, end_year=1855, step=1)
```

    


This returns a dataframe containing samples from the Newspaper Navigator data (loaded via JSON) into a Pandas DataFrame. 

```python
df.columns
```




    Index(['filepath', 'pub_date', 'page_seq_num', 'edition_seq_num', 'batch',
           'lccn', 'box', 'score', 'ocr', 'place_of_publication',
           'geographic_coverage', 'name', 'publisher', 'url', 'page_url'],
          dtype='object')



## Annotation

## Inference

```python
from nnanno.inference import *
```

```python
from fastai.vision.all import *
dls = ImageDataLoaders.from_csv('../ph/ads/', 'ads_upsampled.csv',folder='images', fn_col='file', label_col='label',
                                item_tfms=Resize(64,ResizeMethod.Squish), num_workers=0)
learn = cnn_learner(dls, resnet18, metrics=F1Score())
learn.fit(1)
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
      <td>1.025671</td>
      <td>0.924362</td>
      <td>0.736842</td>
      <td>00:12</td>
    </tr>
  </tbody>
</table>


```python
predictor = nnPredict(learn,try_gpu=False)
```

```python
predictor.predict_sample('ads','testinference',0.01,end_year=1850)
```

    


```python
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
      <td>0.014062</td>
      <td>0.985938</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0</td>
      <td>0.999923</td>
      <td>0.000078</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>0.041149</td>
      <td>0.958851</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>0.032974</td>
      <td>0.967026</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>0.018828</td>
      <td>0.981172</td>
    </tr>
  </tbody>
</table>
</div>


