## :penguin: :penguin: Penguin Classification :penguin: :penguin:

Main aim of this project to is implement end-to-end `ML pipelines on AWS sagemaker` to predict the species of the Penguins.


## 1. Training Pipeline

- In this session we’ll run Exploratory Data Analysis on the [Penguins dataset](https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data) and we’ll build a simple [SageMaker Pipeline](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-sdk.html) with one step to split and transform the data.

<p align="left">
<img src="program/images/training.png"/>
</p>

- We’ll use a [Scikit-Learn Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html) for the transformations, and a [Processing Step](https://docs.aws.amazon.com/sagemaker/latest/dg/build-and-manage-steps.html#step-type-processing) with a [SKLearnProcessor](https://sagemaker.readthedocs.io/en/stable/frameworks/sklearn/sagemaker.sklearn.html#scikit-learn-processor) to execute a preprocessing script. Check the [SageMaker Pipelines Overview](https://docs.aws.amazon.com/sagemaker/latest/dg/pipelines-sdk.html) for an introduction to the fundamental components of a SageMaker Pipeline.

### Step 1: EDA

- `Note`: This step has nothing to do with the pipeline

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

- Now, let’s get the `summary statistics` for the features in our dataset.

```
penguins.describe(include="all")
```

<p align="left">
<img src="program/images/eda2.PNG"/>
</p>

- Let’s now display the distribution of values for the three categorical columns in our data:

```
species_distribution = penguins["species"].value_counts()
island_distribution = penguins["island"].value_counts()
sex_distribution = penguins["sex"].value_counts()

print(species_distribution)
print()
print(island_distribution)
print()
print(sex_distribution)
```
<p align="left">
<img src="program/images/eda3.PNG"/>
</p>

- The distribution of the categories in our data are:

    - `species`: There are 3 species of penguins in the dataset: Adelie (152), Gentoo (124), and Chinstrap (68).
    - `island`: Penguins are from 3 islands: Biscoe (168), Dream (124), and Torgersen (52).
    - `sex`: We have 168 male penguins, 165 female penguins, and 1 penguin with an ambiguous gender (.).

- Let’s replace the ambiguous value in the sex column with a null value:

```
penguins["sex"] = penguins["sex"].replace(".", np.nan)
sex_distribution = penguins["sex"].value_counts()
sex_distribution
```
<p align="left">
<img src="program/images/eda4.PNG"/>
</p>

- Next, let’s check for any missing values in the dataset.

```
penguins.isnull().sum()
```
<p align="left">
<img src="program/images/eda5.PNG"/>
</p>

- Let’s get rid of the missing values. For now, we are going to replace the missing values with the most frequent value in the column. Later, we’ll use a different strategy to replace missing numeric values.

```
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="most_frequent")
penguins.iloc[:, :] = imputer.fit_transform(penguins)
penguins.isnull().sum()
```
<p align="left">
<img src="program/images/eda6.PNG"/>
</p>

- Let’s visualize the distribution of `categorical features`.

```
import matplotlib.pyplot as plt

fig, axs = plt.subplots(3, 1, figsize=(6, 10))

axs[0].bar(species_distribution.index, species_distribution.values)
axs[0].set_ylabel("Count")
axs[0].set_title("Distribution of Species")

axs[1].bar(island_distribution.index, island_distribution.values)
axs[1].set_ylabel("Count")
axs[1].set_title("Distribution of Island")

axs[2].bar(sex_distribution.index, sex_distribution.values)
axs[2].set_ylabel("Count")
axs[2].set_title("Distribution of Sex")

plt.tight_layout()
plt.show()
```
<p align="left">
<img src="program/images/eda7.png"/>
</p>

- Let’s visualize the distribution of `numerical columns`.

```
fig, axs = plt.subplots(2, 2, figsize=(8, 6))

axs[0, 0].hist(penguins["culmen_length_mm"], bins=20)
axs[0, 0].set_ylabel("Count")
axs[0, 0].set_title("Distribution of culmen_length_mm")

axs[0, 1].hist(penguins["culmen_depth_mm"], bins=20)
axs[0, 1].set_ylabel("Count")
axs[0, 1].set_title("Distribution of culmen_depth_mm")

axs[1, 0].hist(penguins["flipper_length_mm"], bins=20)
axs[1, 0].set_ylabel("Count")
axs[1, 0].set_title("Distribution of flipper_length_mm")

axs[1, 1].hist(penguins["body_mass_g"], bins=20)
axs[1, 1].set_ylabel("Count")
axs[1, 1].set_title("Distribution of body_mass_g")

plt.tight_layout()
plt.show()
```

<p align="left">
<img src="program/images/eda8.png"/>
</p>

- Let’s display the covariance matrix of the dataset. The “covariance” measures how changes in one variable are associated with changes in a second variable. In other words, the covariance measures the degree to which two variables are linearly associated.

```
penguins.cov(numeric_only=True)
```
<p align="left">
<img src="program/images/eda9.PNG"/>
</p>

- Here are three examples of what we get from interpreting the covariance matrix below:

    - Penguins that weight more tend to have a larger culmen.
    - The more a penguin weights, the shallower its culmen tends to be.
    - There’s a small variance between the culmen depth of penguins.

- Let’s now display the correlation matrix. “Correlation” measures both the strength and direction of the linear relationship between two variables.

```
penguins.corr(numeric_only=True)
```
<p align="left">
<img src="program/images/eda10.PNG"/>
</p>

- Here are three examples of what we get from interpreting the correlation matrix below:

    - Penguins that weight more tend to have larger flippers.
    - Penguins with a shallower culmen tend to have larger flippers.
    - The length and depth of the culmen have a slight negative correlation.


- Let’s display the distribution of species by island.

```
unique_species = penguins["species"].unique()

fig, ax = plt.subplots(figsize=(6, 6))
for species in unique_species:
    data = penguins[penguins["species"] == species]
    ax.hist(data["island"], bins=5, alpha=0.5, label=species)

ax.set_xlabel("Island")
ax.set_ylabel("Count")
ax.set_title("Distribution of Species by Island")
ax.legend()
plt.show()
```
<p align="left">
<img src="program/images/eda11.png"/>
</p>

- Let’s display the distribution of species by sex.

```
fig, ax = plt.subplots(figsize=(6, 6))

for species in unique_species:
    data = penguins[penguins["species"] == species]
    ax.hist(data["sex"], bins=3, alpha=0.5, label=species)

ax.set_xlabel("Sex")
ax.set_ylabel("Count")
ax.set_title("Distribution of Species by Sex")

ax.legend()
plt.show()
```
<p align="left">
<img src="program/images/eda12.png"/>
</p>


### Step 2: Creating the Preprocessing Script

- Fetch the data from S3 bucket on AWS and send it to a `processing job` (job running on AWS)
- Processing Job splits the data into 3 sets and transforms and the output of this job gets stored back on S3 location called `Dataset splits`:
    - `Training set`
    - `Validation set`
    - `Test set`



## Running the code

Before running the project, follow the [Setup instructions](https://program.ml.school/setup.html). After that, you can test the code by running the following command:

```
$ nbdev_test --path program/cohort.ipynb
```

This will run the notebook and make sure everything runs. If you have any problems, it's likely there's a configuration issue in your setup.

## Resources

* [Serving a TensorFlow model from a Flask application](program/serving/flask/README.md): A simple Flask application that serves a multi-class classification TensorFlow model to determine the species of a penguin.
