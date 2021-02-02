# nnanno
> Sampling, annotating and making predictions on the Newspaper Navigator dataset  


![CI](https://github.com/davanstrien/nnanno/workflows/CI/badge.svg)

## tl;dr

`nnanno` is a modest collection of tools to help work with the delightful [Newspaper Navigator](https://news-navigator.labs.loc.gov/) data. 

[Newspaper Navigator](https://news-navigator.labs.loc.gov/) is a project which extracted visual content (pictures, maps etc.) from the Library of Congress [Chronicling America](https://chroniclingamerica.loc.gov/) digitised newspaper collection. 

Newspaper Navigator has released data in a number of formats including `json` files which contain a range of metadata about the newspaper from which the image was taken from. `nnanno` was thrown together to help work with this collection as part of the preparation of some example datasets used in a series of Programming Historian tutorials. Since this code was developed using the wonderful [nbdev](nbdev.fast.ai/) library it is possible to install it for your own use. 

## What nnanno tries to help with

nnanno doesn't to provide an end-to-end to end 'pipeline' for using machine learning with the Newspaper Navigator data since people wanting to work with data will have different needs and interests. Instead it is a minimal collection of code that *may* help you if you want to work with the Newspaper Navigator data.

There are three particular areas where nnanno tries to help a little:
- sampling from the full Newspaper Navigator data 
- annotating this sample with additional labels using [label studio](https://labelstud.io)
- inference (experimental 😬) running inference i.e making new predictions with a machine learning model on the newspaper navigator dataset using IIIF.

### Disclaimer

This code was written mainly to help develop some example datasets for a series of Programming Historian tutorials. It has some tests but there are likely bugs and issues with the code. The code in this repository was all written in notebooks, some people  hate notebooks. Those people will probably hate this too 🤷‍♂️

If you want to work with the full Newspaper Navigator dataset you will likely be better of accessing it via the proived S3 bucket see [news-navigator.labs.loc.gov/]() for more information.

## nbdev notes
This code was written using `nbdev`. This is a tool that helps use Jupyter notebooks for developing Python libraries. Inside the documentation you will see code cells followed by output. This is generated from a Jupyter notebook and shows the actual output of the code rather than something that has been copied and pasted for example:

```
import datetime
print(datetime.date.today())
```

    2021-02-02


Is evaluated as Python code. This also means all of the documentation and examples can be opened in notebooks and the code inspected, changed and run. 

## Install

At the moment installation is through Git. If the code gets a few more eyes on it then it may get uploaded to pip. 

`pip install nnanno` #TODO add github link

## Programming Historian Data Preparation 

\ # TODO add link to prep notebooks

## Functionality 
The three main areas of `nnanno` are shown below. The examples section in the documentation shows this in greater detail.

### Creating samples

```
from nnanno.sample import *
```

```
sampler = nnSampler()
df = sampler.create_sample(1,'photos',start_year=1850, end_year=1855, step=1)
```

    


This returns a dataframe containing samples from the Newspaper Navigator data (loaded via JSON) into a Pandas DataFrame. 

```
df.columns
```




    Index(['filepath', 'pub_date', 'page_seq_num', 'edition_seq_num', 'batch',
           'lccn', 'box', 'score', 'ocr', 'place_of_publication',
           'geographic_coverage', 'name', 'publisher', 'url', 'page_url'],
          dtype='object')



### Annotation
The annotation part of nnanno is mainly a little bit of documentation and a few functions to help setup annotation of a sample from Newspaper Navigator using IIIF urls and the [label studio](https://labelstud.io/) annotation tool. 

```
from nnanno.annotate import create_label_studio_json
```

### Inference

The inference section of nnanno *attempts* to show one possible way to use IIIF to run inference against samples of Newspaper Navigator using a trained [fastai](https://docs.fast.ai/) model. 

```
from nnanno.inference import *
```

```
from fastai.vision.all import *
dls = ImageDataLoaders.from_csv('../ph/ads/', 
                                'ads_upsampled.csv',
                                folder='images', 
                                fn_col='file', 
                                label_col='label',
                                item_tfms=Resize(64,ResizeMethod.Squish),
                                num_workers=0)
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
      <td>0.771532</td>
      <td>0.644032</td>
      <td>0.800000</td>
      <td>00:12</td>
    </tr>
  </tbody>
</table>


With a trained fastai model we can predict on a sample from Newspaper Navigator

```
predictor = nnPredict(learn, try_gpu=False)
```

```
predictor.predict_sample('ads','testinference',0.01,end_year=1850)
```

    


This returns a `json` file for each year from the sample containing the original newspaper navigator data plus the predictions from your model

```
df = pd.read_json('testinference/1850.json')
```

We can access the 'decoded' predictions

```
df['pred_decoded'].value_counts()
```




    text-only        70
    illustrations    18
    Name: pred_decoded, dtype: int64



or work with the probabilities directly

```
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
      <td>text-only</td>
      <td>0.149351</td>
      <td>0.850649</td>
    </tr>
    <tr>
      <th>1</th>
      <td>illustrations</td>
      <td>0.722150</td>
      <td>0.277850</td>
    </tr>
    <tr>
      <th>2</th>
      <td>illustrations</td>
      <td>0.998170</td>
      <td>0.001830</td>
    </tr>
    <tr>
      <th>3</th>
      <td>text-only</td>
      <td>0.421549</td>
      <td>0.578451</td>
    </tr>
    <tr>
      <th>4</th>
      <td>text-only</td>
      <td>0.062635</td>
      <td>0.937365</td>
    </tr>
  </tbody>
</table>
</div>


