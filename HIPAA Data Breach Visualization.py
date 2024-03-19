# -*- coding: utf-8 -*-
"""
HIPAA Data Breach Visualization: 2022 - 2024

@author: Osi
"""

import os
import pandas as pd
import numpy as np
import matplotlib
from sklearn.feature_extraction.text import TfidVectorizer


## Read in HIPAA breach report from 2022 to 2024
hipaa = "A:/Documents/Python Scripts/Security/breach_report_2022_to_2024.csv"
df = pd.read_csv(hipaa)
df.dtypes


## Breach size distribution
def_fig_size = (15, 6)
df["Individuals Affected"].plot(
    kind="hist", figsize=def_fig_size, log=True, title="Breach Size Distribution")

## Average breach size by entity
# change dtype from object to char
df.groupby("Covered Entity Type").mean().plot(
    kind="bar", figsize=def_fig_size, title="Average Breach Size by Entity Type")

## Number of individuals affected by state
df.groupby("State").sum().nlargest(20, "Individuals Affected").plot.pie(
    y="Individuals Affected", figsize=def_fig_size, legend=False)

## Average breach size by type of breach
# change dtype from object to char
df.groupby("Type of Breach").mean().plot(
    kind="bar", figsize=def_fig_size, title="Average Breach Size by Entity Type")

## TF-IDF Vectorizer
vectorizer = TfidVectorizer()

# fit vectorizer to the breach descriptions and vectorize
df["Web Description"] = df["Web Description"].str.replace("\r", "")
X = df["Web Description"].values
X_transformed = vectorizer.fit_transform(X)

# Select the 15 most imprtant features in the breach descriptions based on TF-IDF
feature_array = np.array(vectorizer.get_feature_names())
tfidf_sorting = np.argsort(X_transformed.toarray()).flatten()[::-1]
n = 15
top_n = feature_array[tfidf_sorting][:n]
print(top_n)

# Print breach descriptions containing a specified keyword

k = 2
i = 0
for x in df["Web Description"].values:
    if "review" in x:   #change keyword here
        i += 1
        print(x)
        print()
        if i == k:
            break
        
