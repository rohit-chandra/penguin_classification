## Penguin Classification

Main aim of this project to is implement end-to-end ML pipelines on AWS sagemaker :target:.


## 1: Training Pipeline

- In this session we’ll run Exploratory Data Analysis on the [Penguins dataset](https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data) and we’ll build a simple [SageMaker Pipeline](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-sdk.html) with one step to split and transform the data.

<p align="left">
<img src="program/images/training.png"/>
</p>

- We’ll use a Scikit-Learn Pipeline for the transformations, and a Processing Step with a SKLearnProcessor to execute a preprocessing script. Check the SageMaker Pipelines Overview for an introduction to the fundamental components of a SageMaker Pipeline.

### Step 1: EDA

- Let’s run Exploratory Data Analysis on the dataset. The goal of this section is to understand the data and the problem we are trying to solve.

- Let’s load the Penguins dataset:

```
import pandas as pd
import numpy as np

penguins = pd.read_csv(DATA_FILEPATH)
penguins.head()
```
<p align="left">
<img src="program/images/eda1.PNG"/>
</p>

- We can see the dataset contains the following columns:

    - `species`: The species of a penguin. This is the column we want to predict.
    - `island`: The island where the penguin was found
    - `culmen_length_mm`: The length of the penguin’s culmen (bill) in millimeters
    - `culmen_depth_mm`: The depth of the penguin’s culmen in millimeters
    - `flipper_length_mm`: The length of the penguin’s flipper in millimeters
    - `body_mass_g`: The body mass of the penguin in grams
    - `sex`: The sex of the penguin

- If you are curious, here is the description of a penguin’s culmen:

<p align="left">
<img src="program/images/culmen.jpeg"/>
</p>










## Running the code

Before running the project, follow the [Setup instructions](https://program.ml.school/setup.html). After that, you can test the code by running the following command:

```
$ nbdev_test --path program/cohort.ipynb
```

This will run the notebook and make sure everything runs. If you have any problems, it's likely there's a configuration issue in your setup.

## Resources

* [Serving a TensorFlow model from a Flask application](program/serving/flask/README.md): A simple Flask application that serves a multi-class classification TensorFlow model to determine the species of a penguin.
