# Data Visualisation and Clustering Analysis of HDB Resale Prices Across Singapore Towns
## Project Overview

This analysis examines Housing & Development Board (HDB) resale flat prices across 26 residential towns in Singapore to identify affordability clusters and pricing patterns. Using publicly available data from Singapore's Open Data Portal spanning January 2017 onwards, the study applies machine learning techniques to segment towns based on their housing price characteristics. Results are presented in a **marimo** notebook.

## Methodology

The analysis employs a systematic approach to understand price dynamics and affordability segmentation:

**Data Processing and Charting with SQL, Pandas and Altair**
- Examined distribution of flat types and total number of flats resold across all towns
- Monthly average resale prices were calculated for each town from the raw transaction data
- A 12-month rolling average was computed (starting December 2017) to smooth short-term fluctuations and reveal underlying trends
- Two key statistical measures were derived for each town: mean resale price and standard deviation across the analysis period

**Clustering Analysis with Scikit-learn**
- Feature standardization was performed using Scikit-learn's StandardScaler to normalize the mean and standard deviation values, ensuring equal weighting in the clustering algorithm
- The Elbow Method was applied to determine the optimal number of clusters (K) for K-means clustering
- K-means clustering algorithm was used to group the 26 towns based on their price characteristics

## Visualisation
View the data and charts as a static html here (https://static.marimo.app/static/hdb-resale-prices-jppx). Access the full **marimo**  notebook via the **molab** badge.
[![Open in molab](https://marimo.io/molab-shield.png)](https://molab.marimo.io/notebooks/nb_9MfjuGro9sjMX19Do5PdKZ)

## Key Insights

The clustering analysis reveals distinct affordability segments among Singapore's residential towns, with groupings determined by both average price levels and price volatility. These clusters provide insights into:

- **Market segmentation**: Towns naturally group into affordability tiers based on historical pricing patterns
- **Price stability**: The standard deviation component captures which towns experience more volatile pricing versus those with stable markets
- **Geographic and demographic patterns**: The clustering reflects underlying factors such as location, maturity of estates, and amenity access

## Applications

This analysis offers practical value for multiple stakeholders:

- **Homebuyers**: Understanding which towns fall into similar affordability clusters aids in expanding housing search options
- **Policy makers**: Identifying pricing patterns across towns can inform housing policy and subsidy targeting
- **Market analysts**: The clustering provides a data-driven framework for market segmentation and trend analysis
- **Researchers**: The methodology demonstrates reproducible techniques for housing market analysis using open data

## Data Source

Analysis based on HDB Resale Flat Prices dataset from Singapore's Open Data Portal, covering monthly transactions by town and flat type from January 2017 onwards.
Housing & Development Board. (2021). Resale flat prices based on registration date from Jan-2017 onwards (2025) [Dataset]. data.gov.sg. Retrieved October 05, 2025 from https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view
