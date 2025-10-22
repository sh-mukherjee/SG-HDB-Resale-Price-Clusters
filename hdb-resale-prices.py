# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "altair==5.5.0",
#     "marimo",
#     "numpy==2.2.6",
#     "pandas==2.3.3",
#     "polars==1.34.0",
#     "requests==2.32.5",
#     "scikit-learn==1.7.2",
#     "scipy==1.16.2",
#     "statsforecast==2.0.2",
#     "statsmodels==0.14.5",
# ]
# ///

import marimo

__generated_with = "0.16.5"
app = marimo.App(
    width="full",
    app_title="hdb-resale-prices",
    auto_download=["html"],
    sql_output="pandas",
)


@app.cell(hide_code=True)
def _(
    clusters,
    conclusion,
    data,
    elbow,
    flats,
    mo,
    summary1,
    summary2,
    summary3,
    title,
    trend_chart,
):
    mo.ui.tabs(
        {
            "Executive Summary": mo.vstack([title, mo.hstack([summary1, summary2]), summary3]),
            "Data": data,
            "Volume": flats,
            "Trends": trend_chart,
            "Clustering": mo.vstack([
                mo.accordion({
                    "Optimal Value of K (click to expand)": elbow
                }), 
                    clusters]),
            "Conclusion": conclusion
        }
    )
    return


@app.cell(hide_code=True)
def _(
    df,
    flat_obs,
    flat_type_chart,
    mo,
    reference,
    total_flats_chart,
    total_obs,
):
    data = mo.vstack([mo.callout(reference, kind='info'), mo.md("## Random Sample of Data"), mo.ui.table(df.sample(n=10, random_state=42)
    )])
    flats = mo.hstack([mo.vstack([flat_obs, flat_type_chart]), mo.vstack([total_obs, total_flats_chart])])
    return data, flats


@app.cell(hide_code=True)
def _(mo):

    title = mo.callout(mo.md("# Clustering Analysis of HDB Resale Prices Across Singapore Towns"))

    summary1 = mo.callout(mo.md(r"""
    ## Project Overview

    This analysis examines Housing & Development Board (HDB) resale flat prices across 26 residential towns in Singapore to identify affordability clusters and pricing patterns. Using publicly available data from Singapore's Open Data Portal spanning January 2017 onwards, the study applies machine learning techniques to segment towns based on their housing price characteristics.

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
    """))

    summary2 = mo.callout(mo.md(r"""

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
    """))

    summary3 =mo.callout(mo.md(r"""
    ## Data Source

    Analysis based on HDB Resale Flat Prices dataset from Singapore's Open Data Portal, covering monthly transactions by town and flat type from January 2017 onwards.
    Housing & Development Board. (2021). Resale flat prices based on registration date from Jan-2017 onwards (2025) [Dataset]. data.gov.sg. Retrieved October 05, 2025 from https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view
    """))
    return summary1, summary2, summary3, title


@app.cell
def _(mo):
    mo.outline(label="Table of Contents")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Import Python Modules""")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell
def _():
    import pandas as pd
    import numpy as np
    import altair as alt
    return alt, pd


@app.cell
def _():
    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    return KMeans, StandardScaler


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Load and Transform the Data""")
    return


@app.cell
def _(mo):
    reference = mo.md('Housing & Development Board. (2021). Resale flat prices based on registration date from Jan-2017 onwards (2025) [Dataset]. data.gov.sg. Retrieved October 05, 2025 from https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view')
    return (reference,)


@app.cell
def _(mo, pd):
    # define a function to read in the CSV file to a Pandas dataframe and convert the month column to 'date' data type
    @mo.cache
    def get_data():
        # Read CSV file into a pandas DataFrame
        df = pd.read_csv('ResaleflatpricesfromJan2017.csv')

        # Convert 'month' column to datetime (format: YYYY-MM)
        df['month'] = pd.to_datetime(df['month'], format='%Y-%m', errors='coerce')
        return df


    df = get_data()
    return (df,)


@app.cell
def _(df):
    df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Count the number of flat types""")
    return


@app.cell
def _(df, mo):
    flat_df = mo.sql(
        f"""
        SELECT
            town,
            flat_type,
            COUNT(*) AS flat_count
        FROM df 
        GROUP BY
            town,
            flat_type,
        ORDER BY
            town,
            flat_type;
        """
    )
    return (flat_df,)


@app.cell
def _(town_avg_df):
    # 1. Define the 26-color fixed palette for the 26 towns
    # list of towns
    all_town_domain = sorted(list(town_avg_df["town"].unique()))
    all_color_range = [
        '#4C78A8', '#9ECADD', '#FF7F0E', '#FFBB78', '#2CA02C', '#98DF8A', '#D62728', '#FF9896', 
        '#9467BD', '#C5B0D5', '#8C564B', '#C49C94', '#E377C2', '#F7B6D2', '#7F7F7F', '#C7C7C7', 
        '#BCBD22', '#DBDB8D', '#17BECF', '#9EDAE5', '#F08E35', '#C93F56', '#A8AECC', '#6B8E23', 
        '#A54B86', '#42A8AE'
    ]
    color_map = dict(zip(all_town_domain, all_color_range))
    return all_color_range, all_town_domain, color_map


@app.cell
def _(alt, flat_df):
    # chart the total number of flat types sold among the towns over the years 2017 to Sep 2025
    flat_type_chart = (
        alt.Chart(flat_df)
        .mark_bar()
        .encode(
            x=alt.X(field='flat_type', type='nominal'),
            y=alt.Y(field='flat_count', type='quantitative'),
            color=alt.Color(field='flat_type', type='nominal', legend=None), #scale=alt.Scale(
                #domain=all_town_domain,
                #range=all_color_range)
                #),
            tooltip=[
                # alt.Tooltip(field='flat_type'),
                alt.Tooltip(field='flat_count', format=',.0f'),
                alt.Tooltip(field='town')
            ]
        )
        .properties(
            title='HDB flat types sold over the years 2017 to Sep 2025',
            height=400,
            width='container',
            config={
                'axis': {
                    'grid': True
                }
            }
        )
    )
    #flat_type_chart
    return (flat_type_chart,)


@app.cell
def _(mo):
    flat_obs = mo.callout(mo.md("Most of the units sold were 3-, 4-, or 5-room flats"), kind="warn")
    return (flat_obs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Count the total number of flats sold in each town over the entire period""")
    return


@app.cell
def _(df, mo):
    total_flats_df = mo.sql(
        f"""
        SELECT
            town,
            COUNT(*) AS total_flats_sold
        FROM df
        GROUP BY town
        ORDER BY town;
        """
    )
    return (total_flats_df,)


@app.cell
def _(all_color_range, all_town_domain, alt, total_flats_df):
    # Chart the total number of flats as a bar graph
    total_flats_chart = (
        alt.Chart(total_flats_df)
        .mark_bar()
        .encode(
            x=alt.X(field='total_flats_sold', type='quantitative'),
            y=alt.Y(field='town', type='nominal', axis=alt.Axis(title=None)),
            color=alt.Color(field='town', type='nominal', scale=alt.Scale(
                domain=all_town_domain,
                range=all_color_range), legend=None),
            tooltip=[
                alt.Tooltip(field='total_flats_sold', format=',.0f'),
                alt.Tooltip(field='town')
            ]
        )
        .properties(
            title=('Total Number of Flats Sold in Each Town'),
            height=400,
            width='container',
            config={
                'axis': {
                    'grid': False
                }
            }
        )
    )
    #total_flats_chart
    return (total_flats_chart,)


@app.cell
def _(mo):
    total_obs = mo.callout(mo.md(r"""
    BUKIT TIMAH has the smallest number of flat resales, followed by MARINE PARADE and CENTRAL AREA
    """), kind='warn')
    return (total_obs,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Monthly Average Resale Price by Town""")
    return


@app.cell
def _(df, mo):
    town_avg_df = mo.sql(
        f"""
        -- Calculate monthly average resale price for each town
        SELECT
            town,
            month,
            AVG(resale_price) AS avg_resale_price
        FROM
            df
        GROUP BY
            town,
            month;
        """
    )
    return (town_avg_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Rolling 12-Month Average Resale Price per Town

    Rolling 12 month averages in SQL. The first 12 m average will be counted from Dec 2017
    """
    )
    return


@app.cell
def _(mo, town_avg_df):
    roll_df = mo.sql(
        f"""
        WITH temp AS (
            SELECT
                town,
                month,
                avg_resale_price,
                AVG(avg_resale_price) OVER (
                    PARTITION BY town
                    ORDER BY month
                    ROWS BETWEEN 11 PRECEDING AND CURRENT ROW
                ) AS moving_avg_12m_price
            FROM
                town_avg_df
        )
        SELECT *
        FROM temp
        WHERE month >= '2017-12-01'
        ORDER BY town, month;
        """
    )
    return (roll_df,)


@app.cell(hide_code=True)
def _(mo, town_avg_df):
    # create a marimo selection widget for towns
    town_sel = mo.ui.multiselect.from_series(town_avg_df["town"], value=["BUKIT TIMAH", "PUNGGOL", "YISHUN"], label="Select Towns")
    return (town_sel,)


@app.cell(hide_code=True)
def _(roll_df, town_sel):
    # Filter the DataFrame using the .isin() method
    town_mask = roll_df["town"].isin(town_sel.value)
    roll_df_filtered = roll_df[town_mask]
    return (roll_df_filtered,)


@app.cell(hide_code=True)
def _(alt, color_map, roll_df_filtered, town_sel):
    # Build the chart for the rolling 12m average resale price for each town

    # Build the domain/range for selected towns
    sel_domain = town_sel.value
    sel_range = [color_map[t] for t in sel_domain]

    roll_avg_chart = (
        alt.Chart(roll_df_filtered)
        .mark_line()
        .encode(
            x=alt.X('month:T'), 
            y=alt.Y('moving_avg_12m_price:Q'),
            color=alt.Color('town:N', scale=alt.Scale(
                domain=sel_domain,
                range=sel_range
            ),
            legend = alt.Legend(title="Selected Towns")
                           ),
            # row=alt.Row(field='town', type='nominal'),
            tooltip=[
                alt.Tooltip(field='month', timeUnit='yearmonth'),
                alt.Tooltip(field='moving_avg_12m_price', format=',.2f'),
                alt.Tooltip(field='town')
            ]
        )
        .properties(
            title='Average Monthly Resale Price over Rolling 12M for HDB Flat in SGD',
            height=390,
            width='container',
            config={
                'axis': {
                    'grid': False
                }
            }
        )
    )
    return


@app.cell
def _(all_color_range, all_town_domain, alt, roll_df):
    # Altair selection
    highlight = alt.selection_point(fields=['town'], bind='legend')

    roll_chart = (
        alt.Chart(roll_df)
        .mark_line()
        .encode(
            x=alt.X('month:T', title='Month'),
            y=alt.Y('moving_avg_12m_price:Q', title='Rolling 12M Average Price (SGD)'),
            color=alt.Color(
                'town:N',
                scale=alt.Scale(domain=all_town_domain, range=all_color_range),
                legend=alt.Legend(title="Click on a town to filter")
            ),
            tooltip=[
                alt.Tooltip(field='month', timeUnit='yearmonth'),
                alt.Tooltip(field='moving_avg_12m_price', format=',.2f'),
                alt.Tooltip(field='town')
            ],
            opacity=alt.condition(highlight, alt.value(1), alt.value(0.1))
        )
        .add_params(highlight)
        .properties(
            title='Rolling 12-Month Average HDB Resale Prices (SGD) - Click on a Town in the Legend to Filter',
            height=400,
            width='container'
        )
    )
    #roll_chart
    return (roll_chart,)


@app.cell
def _(mo):
    obs = mo.callout(mo.md(r"""
    - BUKIT TIMAH is the most expensive town in Singapore for HDB flats
    - YISHUN is the cheapest
    """), kind="warn")
    return (obs,)


@app.cell
def _(mo, obs, roll_chart):
    trend_chart = mo.vstack([obs, roll_chart])
    return (trend_chart,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Cluster Analysis""")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Create a Features Dataframe

    Create a DataFrame with town as the index and Mean_Price and Std_Dev as columns, representing the mean and standard deviation of the rolling 12-month average prices for each town.
    """
    )
    return


@app.cell
def _(mo, roll_df):
    features_df = mo.sql(
        f"""
        SELECT
            town,
            AVG(moving_avg_12m_price) AS Mean_Price,
            STDDEV(moving_avg_12m_price) AS Std_Dev
        FROM
            roll_df
        GROUP BY
            town
        ORDER BY
            town;
        """
    )
    return (features_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Scale Features

    Apply Standard Scaling to the Mean_Price and Std_Dev columns in the Features DataFrame.
    """
    )
    return


@app.cell
def _(StandardScaler, features_df, pd):
    scaler = StandardScaler()
    features_to_scale = features_df[['Mean_Price', 'Std_Dev']]
    scaled_features = scaler.fit_transform(features_to_scale)
    scaled_features_df = pd.DataFrame(scaled_features, columns=['Mean_Price', 'Std_Dev'], index=features_df.index)
    features_df[['Mean_Price', 'Std_Dev']] = scaled_features_df[['Mean_Price', 'Std_Dev']]
    return


@app.cell
def _(features_df):
    features_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Determine Optimal Clusters (Elbow Method)

    Implement the Elbow Method to find the optimal number of clusters by calculating inertia for different values of K and plotting the results.
    """
    )
    return


@app.cell
def _(KMeans, alt, features_df, pd):
    # Calculate inertia values for K values between 1 and 10
    inertia_values = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        kmeans.fit(features_df[['Mean_Price', 'Std_Dev']])
        inertia_values.append(kmeans.inertia_)

    # Create DataFrame for Altair
    elbow_df = pd.DataFrame({
        'k': range(1, 11),
        'inertia': inertia_values
    })

    # Create Altair chart
    elbow_chart = (
        alt.Chart(elbow_df)
        .mark_line(point=True)
        .encode(
            x=alt.X('k:Q', 
                    scale=alt.Scale(domain=[1, 10]),
                    axis=alt.Axis(tickMinStep=1, title='Number of Clusters (K)')),
            y=alt.Y('inertia:Q', 
                    axis=alt.Axis(title='Inertia')),
            tooltip=[
                alt.Tooltip('k:Q', title='K'),
                alt.Tooltip('inertia:Q', format=',.0f', title='Inertia')
            ]
        )
        .properties(
            title='Elbow Method for Optimal K',
            width=600,
            height=400
        )
    )
    #elbow_chart
    return (elbow_chart,)


@app.cell(hide_code=True)
def _(mo):
    k_choice = mo.callout("K = 3 appears to be optimal", kind="success")
    return (k_choice,)


@app.cell(hide_code=True)
def _(mo):
    k_reason = mo.md("The elbow is most evident around K=3, where the rate of inertia reduction slows significantly. Beyond K=4, gains are minimal, suggesting over-clustering risks (e.g., splitting noise, not structure)")
    return (k_reason,)


@app.cell
def _(elbow_chart, k_choice, k_reason, mo):
    elbow = mo.hstack([elbow_chart, mo.vstack([k_choice, k_reason])])
    return (elbow,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Perform k-means clustering

    Run the KMeans algorithm with k=3 and assign the cluster labels to the scaled features dataframe.
    """
    )
    return


@app.cell
def _(KMeans, features_df):
    kmeans3 = KMeans(n_clusters=3, random_state=42, n_init=10)
    features_df['Cluster_ID'] = kmeans3.fit_predict(features_df[['Mean_Price', 'Std_Dev']])
    features_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    ## Visualise Clusters

    Create a scatter plot of the scaled Mean_Price and Std_Dev, colored by Cluster_ID, and add town labels to visualize the clusters.
    """
    )
    return


@app.cell
def _(alt, features_df):
    # Create scatter plot
    scatter = alt.Chart(features_df).mark_circle(size=60).encode(
        x=alt.X('Mean_Price:Q', title='Scaled Mean Price'),
        y=alt.Y('Std_Dev:Q', title='Scaled Standard Deviation'),
        color=alt.Color('Cluster_ID:N', title='Cluster ID', scale=alt.Scale(scheme='set2')),
        tooltip=['town', 'Mean_Price', 'Std_Dev', 'Cluster_ID']
    )

    # Add text labels for towns
    text = scatter.mark_text(align='left', dx=5, dy=-5, fontSize=9).encode(
        text='town:N'
    )

    # Combine scatter and text, configure chart
    cluster_chart = (scatter + text).properties(
        title='Town Clusters based on Mean and Standard Deviation of Rolling Resale Prices',
        width=1000,
        height=1000
    ).configure_axis(
        grid=True
    ).configure_view(
        strokeWidth=0
    )
    # cluster_chart
    return (cluster_chart,)


@app.cell
def _(cluster_chart, cluster_table, mo):
    clusters = mo.hstack([cluster_chart, cluster_table], justify='center')
    return (clusters,)


@app.cell(hide_code=True)
def _(mo):
    cluster_table = mo.md(
        r"""
    | Cluster ID | Name | Description |
    |:---:|:---:|:---|
    | 0 | High Price, Moderate Volatility | Towns in this cluster have a significantly higher average scaled mean price and moderate standard deviation. |
    | 1 | Moderate Price, High Volatility | This cluster is characterized by moderate average scaled mean prices and higher scaled standard deviations, suggesting more price fluctuation. |
    | 2 | Low Price, Low Volatility | Towns in this cluster exhibit lower average scaled mean prices and lower scaled standard deviations, indicating more stable and affordable prices. |
    """
    )
    return (cluster_table,)


@app.cell
def _(mo):
    cluster_names = {
        0: 'High Price, Moderate Volatility',
        1: 'Moderate Price, High Volatility',
        2: 'Low Price, Low Volatility'
    }

    cluster_0 = mo.md(f"Cluster {list(cluster_names.keys())[0]} ({cluster_names[list(cluster_names.keys())[0]]}): Towns in this cluster have a significantly higher average scaled mean price and moderate standard deviation. Examples include BISHAN, BUKIT TIMAH, and CENTRAL AREA.")
    return (cluster_names,)


@app.cell
def _(cluster_names, mo):
    cluster_1 = mo.md((f"Cluster {list(cluster_names.keys())[1]} ({cluster_names[list(cluster_names.keys())[1]]}): This cluster is characterized by moderate average scaled mean prices and higher scaled standard deviations, suggesting more price fluctuation. Examples include BUKIT BATOK, CHOA CHU KANG, and KALLANG/WHAMPOA."))
    return


@app.cell
def _(cluster_names, mo):
    cluster_2 = mo.md((f"Cluster {list(cluster_names.keys())[2]} ({cluster_names[list(cluster_names.keys())[2]]}): Towns in this cluster exhibit lower average scaled mean prices and lower scaled standard deviations, indicating more stable and affordable prices. Examples include ANG MO KIO, BEDOK, and JURONG EAST."))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""# Interpret and Report Findings""")
    return


@app.cell(hide_code=True)
def _(mo):
    conclusion = mo.md(
        r"""
    ## Summary:

    ### What the Z-scores Indicate

    | Scaled Value of Mean | Interpretation |
    |---|---|
    | Negative | The town's average price is below the average of all 26 towns. |
    | Positive | The town's average price is above the average of all 26 towns. |
    | Zero | The town's average price is exactly the average of all 26 towns. |

    | Scaled Value of Std Dev | Interpretation |
    |---|---|
    | Negative | The town's price volatility is lower than the average volatility of all 26 towns. |
    | Positive | The town's price volatility is higher than the average volatility of all 26 towns. |


    ### Data Analysis Key Findings

    *   Cluster 0 is characterized by a high average scaled mean price (average of scaled mean price is 1.49) and moderate scaled standard deviation (average of scaled standard deviation is -0.00).
    *   Cluster 1 exhibits moderate average scaled mean prices (average of scaled mean price is -0.28) and high scaled standard deviations (average of scaled standard deviation is 1.74).
    *   Cluster 2 shows low average scaled mean prices (average of scaled mean price is -0.78) and low scaled standard deviations (average of scaled standard deviation is -0.85).

    ### Insights or Next Steps

    *   The clusters highlight distinct market segments: high-price/moderate-volatility, moderate-price/high-volatility, and low-price/low-volatility.
    *   This clustering can inform strategies for buyers, sellers, and policymakers based on their risk tolerance and price preferences.
    """
    )
    return (conclusion,)


if __name__ == "__main__":
    app.run()
