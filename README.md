# Black Friday Customer Segmentation

An unsupervised machine learning project segmenting 5,891 customers based on their 
Black Friday purchasing behavior and demographic attributes. Unlike the previous two 
projects in this portfolio, which involved predicting a known target column, this project 
has no right answer to check against. The goal is to discover natural groupings in the 
data that describe meaningfully different types of customers a business could act on.

## Project Structure

.

├── README.md

├── notebooks/

│   └── black_friday_clustering.ipynb

├── data/

│   └── train.csv

└── requirements.txt

## Project Overview

The Black Friday Sales dataset contains 550,068 transaction records across 5,891 unique 
customers. A key preprocessing step was aggregating the transaction-level data into a 
single row per customer before clustering could be applied, since clustering individual 
purchases rather than customers would answer the wrong question entirely.

Three customer segments were identified using K-Means clustering with K=3, selected 
using both the elbow method and silhouette scoring.

## Results

| Cluster | Size | Avg Total Purchase | Avg Transactions | Key Characteristics |
|---|---|---|---|---|
| 0 | 802 | 2.77M | 312 | Heavy spenders, City B concentrated |
| 1 | 1,289 | 539K | 57 | Older, mostly married, City C |
| 2 | 3,800 | 574K | 60 | Younger, typical behavior, City C |

Cluster 0 is the standout finding, a small group of power users spending roughly five 
times more and transacting roughly five times more often than the rest of the customer 
base, disproportionately concentrated in City B.

## Methodology

Key decisions and techniques used throughout:

- Aggregated 550,068 transaction rows to 5,891 customer-level rows before clustering
- Target encoded Occupation (21 categories) to avoid overwhelming K-Means distance 
  calculations with 21 sparse one-hot columns
- Label encoded Age to preserve its natural ordinal progression
- Applied RobustScaler instead of standard scaling to reduce the influence of extreme 
  spending outliers without removing those customers from the analysis
- Used both the elbow method and silhouette scoring to select K=3, with honest 
  acknowledgment that the silhouette score (0.267) indicates real but soft cluster 
  boundaries rather than sharply distinct groups

## Key Takeaway

The clustering found one clearly defined segment, the heavy shoppers in Cluster 0, and 
one age and marital status based split between Clusters 1 and 2. A silhouette score of 
0.267 confirmed real but moderate separation, meaning the segments represent gradual 
differences in customer behavior rather than sharply distinct customer types. Cluster 0 
is a genuinely actionable finding for a business, the kind of power user group worth 
prioritizing for retention and loyalty programs.

## Dataset

[Black Friday Sales Dataset](https://www.kaggle.com/datasets/sdolezel/black-friday) on 
Kaggle, 550,068 transaction records across 5,891 customers and 3,631 products.

## Requirements

pandas
numpy
matplotlib
seaborn
scikit-learn
scipy

Install with:

pip install -r requirements.txt

## Running the Notebook

1. Clone this repository
2. Install the requirements above
3. Open `notebooks/black_friday_clustering.ipynb` and run all cells in order