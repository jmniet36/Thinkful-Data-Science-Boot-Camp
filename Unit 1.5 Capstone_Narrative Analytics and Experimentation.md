

```python
from IPython.display import HTML

HTML('''<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>''')
```




<script>
code_show=true; 
function code_toggle() {
 if (code_show){
 $('div.input').hide();
 } else {
 $('div.input').show();
 }
 code_show = !code_show
} 
$( document ).ready(code_toggle);
</script>
<form action="javascript:code_toggle()"><input type="submit" value="Click here to toggle on/off the raw code."></form>




```python
import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
import pandas as pd
import ipywidgets as widgets
from scipy import special
from pandas import Series
import plotly as py
import plotly.graph_objs as go
import math
from IPython.display import Image

py.offline.init_notebook_mode(connected=True)
```


<script type="text/javascript">window.PlotlyConfig = {MathJaxConfig: 'local'};</script><script type="text/javascript">if (window.MathJax) {MathJax.Hub.Config({SVG: {font: "STIX-Web"}});}</script><script>requirejs.config({paths: { 'plotly': ['https://cdn.plot.ly/plotly-latest.min']},});if(!window._Plotly) {require(['plotly'],function(plotly) {window._Plotly=plotly;});}</script>


<font style = 'font-family:Book Antiqua'><i><font size = 4>Thinkful Data Science Boot Camp: Unit 1.5 Capstone</i>
<p></p>
    <p style='line-height:1'><font size = 8><b> U.S.A. Minimum Wage and Median Home Price<b></font></p>
<br>
<br>

<font style = 'font-family:Book Antiqua'><font size = 6><b><u>The Background</u></b>

<font style = 'font-family:Book  Antiqua'><font size = 5><i><b>Exploring Minimum Wages by State from 1968 to 2017</b></i>

<p><font style = 'font-family:Book Antiqua'><font size = 4>To better analyze the minimum wage, lets consolidate all the states and the Federal Minimum wage to one mean value for each year.
<p> The chart preview below calculates the mean instance for the values <i>High_Value, Low_Value, CPI_Average, High_2018</i>, and <i>Low_ 2018</i> for all states and D.C. then groups them by year.
<p> Note that the value for <i>CPI_Average</i> remains the same because the value displayed in the original data set was already the average calculated value for that given year, repeated every year for each state. 


```python
# Importing file with wage data
wage = pd.read_csv('Minimum Wage Data.csv', encoding = "Windows-1252" )

# Converting the State column from series to string data
wage['State'].to_string

# Removing all rows that do not contain the 50 U.S. States but leaving the Federal minimum wage (FLSA).
wage = wage[~wage['State'].isin(['Guam','District of Columbia','Puerto Rico','U.S. Virgin Islands'])]

# Dropping Footnote and Table_Data column from data set.
wage = wage.drop(columns = ['Footnote','Table_Data'])


# Computing the yearly mean wage each year in the United States.
mean_yearly_wage = wage.groupby(['Year'], as_index=False).mean()
mean_yearly_wage.head()
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
      <th>Year</th>
      <th>High_Value</th>
      <th>Low_Value</th>
      <th>CPI_Average</th>
      <th>High_2018</th>
      <th>Low_2018</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1968</td>
      <td>0.889534</td>
      <td>0.834299</td>
      <td>34.783333</td>
      <td>6.404510</td>
      <td>6.006863</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1969</td>
      <td>0.889534</td>
      <td>0.834299</td>
      <td>36.683333</td>
      <td>6.072157</td>
      <td>5.695490</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1970</td>
      <td>0.972745</td>
      <td>0.917020</td>
      <td>38.825000</td>
      <td>6.273137</td>
      <td>5.913333</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1971</td>
      <td>0.972745</td>
      <td>0.917020</td>
      <td>40.491667</td>
      <td>6.014314</td>
      <td>5.669804</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1972</td>
      <td>1.171765</td>
      <td>1.109176</td>
      <td>41.816667</td>
      <td>7.016471</td>
      <td>6.641569</td>
    </tr>
  </tbody>
</table>
</div>



<p><font style = 'font-family:Book Antiqua'><font size = 4>The graph below displays the mean wage over time, rounded to the closest penny. We can see that wage has increased over time from \$.89 cents in 1968 to \$7.50 in 2017, which is \$.25 higher than the <a href='https://www.dol.gov/whd/minimumwage.htm'> current Federal Minimum Wage</a>. It is likely that this discrepency is due to the high minimum wage in states like Washington, even though some states do not report their own minimum wage.

<p>The year 1991 sees a jump in mean wage from the steady increase over time. This is likely due to the high increase of the minimum wage in Washington, Oregon, and Alaska that happened that same year (see graph <i>Wage Over Time: by State and Year</i>).</p> 

<p>Though the Federal Minimum Wage has not increased since 2009, we can see that the mean wage in the U.S. has continued to increase. This is likely in relation to states adapting their own minimum wage that is higher than the federal mandated one. In the graph <i>Wage Over Time: by State and Year</i>, we can see that most of the states are clusterd closer together in the earlier years between 1968-1975 and 1980-1990 and as wages begin to vary and increase, so does the mean wage in the U.S.</p>


<p>We can compare the mean wage in each given year to the 2018 equivalent wage, <i>High_2018</i>. The value of any given wage if it was earned in 2018. So the wage of \$0.89 cents is is the 2018 equivalant of \$6.40.</p>

<p>Note that over time the mean minimum wage and the 2018 equivalent get closer together as wages no longer need to be adjusted for inflation. As we have seen on the above graphs, the minimum wage increases at a steady pace with some short periods of no growth. Looking at the <i>2018 Wage Equivalent</i> overlayed, we can see that there are some points on the line where the 2018 wage equivalent drops down closer to the real wage.</p>

<p>Generally, the declines in equivalent value seems to coincide with a <a href='https://www.thebalance.com/the-history-of-recessions-in-the-united-states-3306011'> recession</a>, with the exception of 2008's <a href='https://www.thebalance.com/the-great-recession-of-2008-explanation-with-dates-4056832'>Great Recession</a>. The reason for the drop is probably because the value of the U.S. dollar at those points in time had decreased in value and the purchasing power dropped, even though minimum wages continued to rise. In terms of 2018 dollars, that money did not go as far.</p>


```python
layout = dict(title = 'Mean Wage Over Time: A Comparison To 2018 Value',
              xaxis = dict(title = 'Year'),
              yaxis = dict(title = 'Wage in U.S. $'),
              annotations=[
                dict(
                    x=1973,
                    y=6.61,
                    xref='x',
                    yref='y',
                    text='OPEC Oil Embargo',
                    showarrow=True,
                    bgcolor= 'yellow',
                    arrowhead=4,
                    ax=0,
                    ay=70
                ),
                dict(
                    x=1980,
                    y=6.37,
                    xref='x',
                    yref='y',
                    text='Interest Rate Hike',
                    showarrow=True,
                    bgcolor= 'yellow',
                    arrowhead=4,
                    ax=0,
                    ay=-50
                ),             
                dict(
                    x=1990,
                    y=4.96,
                    xref='x',
                    yref='y',
                    text='Savings/Loan Crisis',
                    showarrow=True,
                    bgcolor= 'yellow',
                    arrowhead=4,
                    ax=0,
                    ay=30
                ),        
                dict(
                    x=2008,
                    y=6.88,
                    xref='x',
                    yref='y',
                    text='Great Recession',
                    showarrow=True,
                    bgcolor= 'yellow',
                    arrowhead=4,
                    ax=0,
                    ay=-45,
                ),
                dict(
                    x=1970,
                    y=6.27,
                    xref='x',
                    yref='y',
                    text='1970 Recession',
                    showarrow=True,
                    bgcolor= 'yellow',
                    arrowhead=4,
                    ax=0,
                    ay=-45,
                ),                                       
                dict(
                    x=2001,
                    y=6.25,
                    xref='x',
                    yref='y',
                    text='Dot-Com Crash',
                    showarrow=True,
                    bgcolor= 'yellow',
                    arrowhead=4,
                    ax=0,
                    ay=-45,
                ),                         
              ]            
        )


# Create a trace
trace5 = go.Scatter(
     x = mean_yearly_wage['Year'],
     y = round(mean_yearly_wage['High_Value'],2),
     name = 'High Yearly Wage',
     line = dict(
         color = ('rgb(102, 183, 163)')
     )
)     

trace6 = go.Scatter(
     x = mean_yearly_wage['Year'],
     y = round(mean_yearly_wage['High_2018'],2),
     name = '2018 Wage Equivelant',
     line = dict(
         color = ('rgb(184, 232, 16)')
     )
)

data = [trace5,trace6]

fig = dict(data=data, layout=layout)
py.offline.iplot(fig, filename='styled-line')
```


<div id="2200689e-a542-46ac-a810-8388b9c9cbbe" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";
if (document.getElementById("2200689e-a542-46ac-a810-8388b9c9cbbe")) {
    Plotly.newPlot("2200689e-a542-46ac-a810-8388b9c9cbbe", [{"line": {"color": "rgb(102, 183, 163)"}, "name": "High Yearly Wage", "x": [1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017], "y": [0.89, 0.89, 0.97, 0.97, 1.17, 1.17, 1.17, 1.17, 1.61, 1.61, 1.61, 1.96, 2.1, 2.24, 2.24, 2.24, 2.24, 2.24, 2.24, 2.24, 2.59, 2.59, 2.59, 3.17, 3.44, 3.44, 3.55, 3.55, 3.59, 3.85, 4.16, 4.16, 4.34, 4.42, 4.61, 4.67, 4.73, 4.78, 5.0, 5.55, 5.91, 6.23, 6.58, 6.61, 6.66, 6.69, 6.84, 7.1, 7.29, 7.5], "type": "scatter", "uid": "0fc18bce-0cea-4f11-877f-f8d76a5e8224"}, {"line": {"color": "rgb(184, 232, 16)"}, "name": "2018 Wage Equivelant", "x": [1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017], "y": [6.4, 6.07, 6.27, 6.01, 7.02, 6.61, 5.95, 5.45, 7.06, 6.63, 6.16, 6.78, 6.37, 6.16, 5.81, 5.62, 5.39, 5.21, 5.11, 4.93, 5.47, 5.22, 4.96, 5.83, 6.14, 5.97, 6.0, 5.84, 5.73, 6.0, 6.4, 6.26, 6.32, 6.25, 6.41, 6.36, 6.27, 6.12, 6.21, 6.7, 6.88, 7.27, 7.55, 7.36, 7.26, 7.19, 7.24, 7.5, 7.6, 7.66], "type": "scatter", "uid": "03ecdd3a-4d78-4bd5-be83-f24423e27aa1"}], {"annotations": [{"arrowhead": 4, "ax": 0, "ay": 70, "bgcolor": "yellow", "showarrow": true, "text": "OPEC Oil Embargo", "x": 1973, "xref": "x", "y": 6.61, "yref": "y"}, {"arrowhead": 4, "ax": 0, "ay": -50, "bgcolor": "yellow", "showarrow": true, "text": "Interest Rate Hike", "x": 1980, "xref": "x", "y": 6.37, "yref": "y"}, {"arrowhead": 4, "ax": 0, "ay": 30, "bgcolor": "yellow", "showarrow": true, "text": "Savings/Loan Crisis", "x": 1990, "xref": "x", "y": 4.96, "yref": "y"}, {"arrowhead": 4, "ax": 0, "ay": -45, "bgcolor": "yellow", "showarrow": true, "text": "Great Recession", "x": 2008, "xref": "x", "y": 6.88, "yref": "y"}, {"arrowhead": 4, "ax": 0, "ay": -45, "bgcolor": "yellow", "showarrow": true, "text": "1970 Recession", "x": 1970, "xref": "x", "y": 6.27, "yref": "y"}, {"arrowhead": 4, "ax": 0, "ay": -45, "bgcolor": "yellow", "showarrow": true, "text": "Dot-Com Crash", "x": 2001, "xref": "x", "y": 6.25, "yref": "y"}], "title": {"text": "Mean Wage Over Time: A Comparison To 2018 Value"}, "xaxis": {"title": {"text": "Year"}}, "yaxis": {"title": {"text": "Wage in U.S. $"}}}, {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly"}); 
}
});</script><script type="text/javascript">window.addEventListener("resize", function(){if (document.getElementById("2200689e-a542-46ac-a810-8388b9c9cbbe")) {window._Plotly.Plots.resize(document.getElementById("2200689e-a542-46ac-a810-8388b9c9cbbe"));};})</script>


<font style = 'font-family:Book  Antiqua'><font size = 5><i><b>Exploring Median Home Prices Between 1963 to 2011</b></i>

<font style = 'font-family:Book Antiquai'><font size = 4> The <a href='https://www.census.gov/construction/nrs/historical_data/index.html'>United States Census Bureau</a> keeps records of home sales across the country. The below data set is for  new, single-family homes only from 1963 to 2011. The Survey of Construction does not collect sales information for multifamily buildings or for existing homes.

<br>By taking the yearly median home value we can better analyze the the trends in home prices. 


```python
# Importing file with median price of new homes.
median_home = pd.read_csv('median_home_price.csv')
median_home['Adjusted Home'] = median_home['Median_Home_Price']/1000

# Computing the yearly mean wage each year in the United States.
mean_yearly_home = median_home.groupby(['Year'], as_index = False).mean()
mean_yearly_home.head()
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
      <th>Year</th>
      <th>Median_Home_Price</th>
      <th>Adjusted Home</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1963</td>
      <td>17966.666667</td>
      <td>17.966667</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1964</td>
      <td>18983.333333</td>
      <td>18.983333</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1965</td>
      <td>20150.000000</td>
      <td>20.150000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1966</td>
      <td>21525.000000</td>
      <td>21.525000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1967</td>
      <td>22691.666667</td>
      <td>22.691667</td>
    </tr>
  </tbody>
</table>
</div>



<br><font style = 'font-family:Book Antiqua'><font size = 4><p>In the below graph CPI and home prices are again displayed, however, the slope of home prices is more discernable. In this graph we can see that home prices actually stabalized in 2006, two years before the housing bubble burst. We can see that 2009 was the lowest point in value for new, single-family homes. Again we see that home values are significantly higher than CPI from 2003 to 2008. Though CPI was increasing and seemed healthy, it was not able to predict the housing market crash. This is likely because of the way housing is calculated, where it calculates the monthly cost of owning a home based on rent prices. As we saw during the Great Recession, the number of vacant homes increased, causing the rental prices to also increase as new renters flooded the market. This means that the CPI was given a false low reading during the housing boom, pre-Great Recession.</p>


```python
layout = dict(title = 'Yearly Median Home Value (<i>New, Single-Family Homes</i>) and CPI Over Time',
              xaxis = dict(title = 'Year'),
              yaxis = dict(title = '$ U.S. '),
            annotations=[
                dict(
                    x=2006,
                    y=244,
                    xref='x',
                    yref='y',
                    text='Home prices stabalize',
                    showarrow=True,
                    bgcolor= 'yellow',
                    arrowhead=1,
                    ax=10,
                    ay=-70
                ),
                dict(
                    x=2009,
                    y=213,
                    xref='x',
                    yref='y',
                    text='Lowest home values',
                    showarrow=True,
                    bgcolor= 'yellow',
                    arrowhead=1,
                    ax=10,
                    ay=70
                ),
            ]
    )


# Create a trace
trace7 = go.Scatter(
     x = mean_yearly_wage['Year'],
     y = mean_yearly_wage['CPI_Average'],
     name = 'Average CPI',
     line = dict(
         color = ('rgb(102, 183, 163)')
     )
)     

trace8 = go.Scatter(
     x = mean_yearly_home['Year'],
     y = round(mean_yearly_home['Adjusted Home'],2),
     name = 'Home Price Displayed in $1000s',
     line = dict(
         color = 'red'
     )
)

data = [trace7,trace8]

fig = dict(data=data, layout=layout)
py.offline.iplot(fig, filename='styled-line')

```


<div id="ad6741c0-6ca6-460b-a4c2-512040f104f5" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";
if (document.getElementById("ad6741c0-6ca6-460b-a4c2-512040f104f5")) {
    Plotly.newPlot("ad6741c0-6ca6-460b-a4c2-512040f104f5", [{"line": {"color": "rgb(102, 183, 163)"}, "name": "Average CPI", "x": [1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017], "y": [34.78333333, 36.68333333000004, 38.82500000000003, 40.49166667000003, 41.81666666999996, 44.400000000000034, 49.30833333000005, 53.81666666999995, 56.90833333, 60.608333329999965, 65.23333332999995, 72.57499999999993, 82.40833332999999, 90.92500000000007, 96.5, 99.60000000000002, 103.88333330000015, 107.56666669999993, 109.60833329999996, 113.625, 118.2583333, 123.96666670000012, 130.6583332999999, 136.19166669999984, 140.31666669999984, 144.45833330000002, 148.22500000000014, 152.38333330000015, 156.85000000000014, 160.51666670000014, 163.00833330000015, 166.5749999999999, 172.19999999999993, 177.06666669999984, 179.875, 183.95833329999994, 188.88333330000015, 195.29166670000015, 201.5916667000001, 207.34241669999994, 215.30249999999987, 214.53700000000012, 218.05550000000022, 224.9391667, 229.59391669999982, 232.95708329999982, 236.7361667, 237.01699999999985, 240.00716669999986, 245.11958330000013], "type": "scatter", "uid": "69595e0d-8354-4d7b-bfdf-3d85fa54575b"}, {"line": {"color": "red"}, "name": "Home Price Displayed in $1000s", "x": [1963, 1964, 1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011], "y": [17.97, 18.98, 20.15, 21.52, 22.69, 24.83, 25.58, 23.53, 25.22, 27.55, 32.71, 36.02, 39.24, 44.28, 48.98, 55.79, 62.75, 64.71, 68.82, 69.3, 75.46, 80.02, 84.28, 92.23, 104.68, 113.35, 120.38, 122.28, 120.02, 121.3, 126.14, 130.41, 133.43, 139.77, 145.05, 151.98, 159.84, 166.54, 172.61, 185.02, 191.38, 217.82, 234.21, 243.07, 243.74, 230.41, 214.5, 221.24, 224.07], "type": "scatter", "uid": "7b8700dd-2884-4cfb-8a41-94d34c6ee00a"}], {"annotations": [{"arrowhead": 1, "ax": 10, "ay": -70, "bgcolor": "yellow", "showarrow": true, "text": "Home prices stabalize", "x": 2006, "xref": "x", "y": 244, "yref": "y"}, {"arrowhead": 1, "ax": 10, "ay": 70, "bgcolor": "yellow", "showarrow": true, "text": "Lowest home values", "x": 2009, "xref": "x", "y": 213, "yref": "y"}], "title": {"text": "Yearly Median Home Value (<i>New, Single-Family Homes</i>) and CPI Over Time"}, "xaxis": {"title": {"text": "Year"}}, "yaxis": {"title": {"text": "$ U.S. "}}}, {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly"}); 
}
});</script><script type="text/javascript">window.addEventListener("resize", function(){if (document.getElementById("ad6741c0-6ca6-460b-a4c2-512040f104f5")) {window._Plotly.Plots.resize(document.getElementById("ad6741c0-6ca6-460b-a4c2-512040f104f5"));};})</script>


<font style = 'font-family:Book Antiqua'><font size = 5><i><b>Comparing Wage and Median Home Prices</b></i>

<font style = 'font-family:Book Antiqua'><font size = 4> By combining the created data sets <i>Median Wage</i> and <i>Median Home Prices</i> we can create two new columns with the percent change of values from <i>High_Value</i>(median minimum wage in a given year) and <i>Median_Home_Price</i> (new, single-family home prices). See columns in data set below <b>PercentHome_Change</b> and <b>PercentWage_Change</b>.
<p>
<font size = 3><ul><u><b>Added Chart Values:</u></b>
<p class='tab'><p><b>PercentHome_Change</b>: Percent change from previous year for median minimum wage.
<p><b>PercentWage_Change</b>: Percent change from previous year for median prices of new, single-family homes. 



```python
# Merging both data sets wage and median home.
wage_and_medianhome = mean_yearly_wage.merge(mean_yearly_home, on = 'Year')


wage_and_medianhome['PercentHome_Change'] = wage_and_medianhome['Median_Home_Price'].pct_change(1)
wage_and_medianhome['PercentWage_Change'] = wage_and_medianhome['High_Value'].pct_change(1)

print(wage_and_medianhome.shape)
wage_and_medianhome.head()
```

    (44, 10)
    




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
      <th>Year</th>
      <th>High_Value</th>
      <th>Low_Value</th>
      <th>CPI_Average</th>
      <th>High_2018</th>
      <th>Low_2018</th>
      <th>Median_Home_Price</th>
      <th>Adjusted Home</th>
      <th>PercentHome_Change</th>
      <th>PercentWage_Change</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1968</td>
      <td>0.889534</td>
      <td>0.834299</td>
      <td>34.783333</td>
      <td>6.404510</td>
      <td>6.006863</td>
      <td>24833.333333</td>
      <td>24.833333</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1969</td>
      <td>0.889534</td>
      <td>0.834299</td>
      <td>36.683333</td>
      <td>6.072157</td>
      <td>5.695490</td>
      <td>25575.000000</td>
      <td>25.575000</td>
      <td>0.029866</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1970</td>
      <td>0.972745</td>
      <td>0.917020</td>
      <td>38.825000</td>
      <td>6.273137</td>
      <td>5.913333</td>
      <td>23533.333333</td>
      <td>23.533333</td>
      <td>-0.079831</td>
      <td>0.093544</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1971</td>
      <td>0.972745</td>
      <td>0.917020</td>
      <td>40.491667</td>
      <td>6.014314</td>
      <td>5.669804</td>
      <td>25216.666667</td>
      <td>25.216667</td>
      <td>0.071530</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1972</td>
      <td>1.171765</td>
      <td>1.109176</td>
      <td>41.816667</td>
      <td>7.016471</td>
      <td>6.641569</td>
      <td>27550.000000</td>
      <td>27.550000</td>
      <td>0.092531</td>
      <td>0.204596</td>
    </tr>
  </tbody>
</table>
</div>



<br><font style = 'font-family:Book Antiqua'><font size = 4><p>The graph below, <i>Minimum Wage and Median Single-Family Home Price: Yearly Percent Change</i>, is a better visual representation of the changes in each year. Here, we can compare the changes in wages and homes each year. As we can see, percent wage change varies over the years, but never dips below zero. This supports the continual upward trend seen in graph <i>Wage Over Time: By Year</i>. In contrast, percent change for new, single-family home values dip below zero on four seperate occasions. These occasions align with the 1970's Recession, savings/loan crisis, and the Great Recession.</p> 
<br>
<p>Notably, there seems to be greater increase in wages before 1995. After 1995, the changes are more moderate with the majority being less than a .1% change. At the same time, home values seem to be increasing at a higher percent than wages. Between the years 2001 and 2005, home values increased by at least twice or more percentage points than wages. Then the Great Recession happened and home prices plummetted before increasing again in 2010 and 2011.</p>


```python
layout = dict(title = 'Minimum Wage and Median Single-Family Home Prices: Yearly Percent Change',
              xaxis = dict(title = 'Year'),
              yaxis = dict(title = 'Percent Change')
             )

    
data = [
    go.Bar(
        x = wage_and_medianhome['Year'],
        y = round(wage_and_medianhome['PercentHome_Change'],2),
        base = 0,
        marker = dict(
          color = ('rgb(184, 232, 16)')
        ),
        name = '% Home Price Change'
    ),
    go.Bar(
        x = wage_and_medianhome['Year'],
        y = round(wage_and_medianhome['PercentWage_Change'],2),
        base = 0,
        marker = dict(
          color = ('rgb(102, 183, 163)')
        ),
        name = '% Wage Change'
    )
]


fig = go.Figure(layout=layout,data=data)
py.offline.iplot(fig, filename='base-bar')
```


<div id="31430041-0236-4576-ad92-427ba4cfb496" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";
if (document.getElementById("31430041-0236-4576-ad92-427ba4cfb496")) {
    Plotly.newPlot("31430041-0236-4576-ad92-427ba4cfb496", [{"base": 0, "marker": {"color": "rgb(184, 232, 16)"}, "name": "% Home Price Change", "x": [1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011], "y": [null, 0.03, -0.08, 0.07, 0.09, 0.19, 0.1, 0.09, 0.13, 0.11, 0.14, 0.12, 0.03, 0.06, 0.01, 0.09, 0.06, 0.05, 0.09, 0.13, 0.08, 0.06, 0.02, -0.02, 0.01, 0.04, 0.03, 0.02, 0.05, 0.04, 0.05, 0.05, 0.04, 0.04, 0.07, 0.03, 0.14, 0.08, 0.04, 0.0, -0.05, -0.07, 0.03, 0.01], "type": "bar", "uid": "0719d2f3-c862-4872-9f06-b8ab2367a657"}, {"base": 0, "marker": {"color": "rgb(102, 183, 163)"}, "name": "% Wage Change", "x": [1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975, 1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986, 1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997, 1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011], "y": [null, 0.0, 0.09, 0.0, 0.2, 0.0, 0.0, 0.0, 0.37, 0.0, 0.0, 0.22, 0.07, 0.07, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.16, 0.0, 0.0, 0.23, 0.09, 0.0, 0.03, 0.0, 0.01, 0.07, 0.08, 0.0, 0.04, 0.02, 0.04, 0.01, 0.01, 0.01, 0.05, 0.11, 0.07, 0.05, 0.05, 0.01], "type": "bar", "uid": "e7b5580e-19ec-47ef-94c3-670908372a42"}], {"title": {"text": "Minimum Wage and Median Single-Family Home Prices: Yearly Percent Change"}, "xaxis": {"title": {"text": "Year"}}, "yaxis": {"title": {"text": "Percent Change"}}}, {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly"}); 
}
});</script><script type="text/javascript">window.addEventListener("resize", function(){if (document.getElementById("31430041-0236-4576-ad92-427ba4cfb496")) {window._Plotly.Plots.resize(document.getElementById("31430041-0236-4576-ad92-427ba4cfb496"));};})</script>


<font style = 'font-family:Book Antiqua'><font size = 4><p> Overall we can see that wages and new, single-family home prices have a similar trend of continual growth, despite some moments of decline. Further research is needed to truly assess if home values are increasing at a greater rate then wages. It would be beneficial to examine home prices by state rather than month. It might also be interesting to examine home prices for all types of homes, rather than just new, single-family homes as we did in this analysis. It would be intresting to note how each state varied in regards to its wages and home prices.</p>

<hr width=50%>

<font style = 'font-family:Book Antiqua'><i><font size = 4>Experimental Design:</i>
<p></p>
    <p style='line-height:1'><font size = 8><b> Will Increased Minimum Wage in the Most Expensive Cities in the USA Improve Home Affordability?</b></font></p>


<font style = 'font-family:Book Antiqua'><font size = 4><p> According to the article published by Dan Burrows, <i><a href='https://www.kiplinger.com/slideshow/real-estate/T006-S001-most-expensive-u-s-cities-to-live-in-2019/index.html'>20  Most Expensive U.S. Cities to Live In</a></i>, the cities of Honolulu, Seattle and Manhattan are the top three most expensive cities to live in the United States with a median home value of \\$626k to \\$927k. Though most of these cities do have a higher wage than the federal minimum, Seattle and Manhattan as high as \\$12.</p>
<br>
<p>Through the analysis above we can see that overall, the wage in the United States has increased through the years, the data cannot truly represent what is happening in this cities with high priced homes and high cost of living. For example, Manahattan's median home price is \$915,300 but Albany, a county north of Manhattan, has a median home price of <a href='https://www.tax.ny.gov/research/property/assess/sales/resmedian.htm'>\$210,000</a>. Large distributions among two different counties in the same state can create a skewed picture of median home prices in the whole state.</p>
<br>
<p>In the experiment below, we will focus on these high price cities and see if a wage increase will make homeownership, a pillar of the American Dream, come true for more Americans in those cities.</p>

<font style = 'font-family:Book Antiqua'><font size = 6><b><u>Experimental Design</u></b>

<font style = 'font-family:Book Antiqua'><font size = 5><i><b><b>Hypothesis:</b></i> 
<font style = 'font-family:Book Antiqua'><font size = 4><p>Will an increase in minimum wage to \\$20 an hour in the cities of Honolulu, Hawaii; Seattle, Washington and Manhattan, New York improve home affordability for low-income families? 
    <br>
    <br>
<font style = 'font-family:Book  Antiqua'><font size = 5><p><i><b>Sample:</b></i>
<font style = 'font-family:Book  Antiqua'><font size = 4><p>We will focus on three major cities:
- Honolulu, Hawaii
- Seattle, Washington
- Manhattan, New York</p>

<p>Our analysis will also focus one the affects of a wage increase of families that make 80 percent or less of Area Median Income (AMI):

<b><a href='https://dbedt.hawaii.gov/hhfdc/files/2018/04/2018-HUD-Income-Limits-All-Counties.pdf'>Honolulu, Hawaii:</a></b>

<img src='Honolulu AMI.png'>

<br>
<br>
<b><a href='https://www.seattle.gov/Documents/Departments/Housing/PropertyManagers/IncomeRentLimits/2018%20Rent%20and%20Income%20Limits.pdf'>Seattle, Washington:</a></b>
<img src='Seattle AMI.png'>

<br>
<br>
<b><a href='https://www1.nyc.gov/site/hpd/renters/area-median-income.page'>Manhattan, New York:</a></b>
<img src='New York AMI.png'>

<font style = 'font-family:Book Antiqua'><font size = 6><b><u>
    Rollout & Evaluation Plan</u></b>

<font style = 'font-family:Book Antiqua'><font size = 5><i><b>Impact Window:</b></i> 
<font style = 'font-family:Book Antiqua'><font size = 4><p>The first 5 years after the minimum wage is increased to \\$20 an hour.</p>
<br>
<br>
<font style = 'font-family:Book Antiqua'><font size = 5><i><b>Success Metric:</b></i> 
<font style = 'font-family:Book Antiqua'><font size = 4><p>An increase of 2% in home ownership.
Increased home ownership in the cities of Honolulu, Seattle and San Francisco among households that make 80\% percent or less of AMI.</p> 
<br>
<br>
<font style = 'font-family:Book Antiqua'><font size = 5><i><b>Secondary Metrics:</b></i> 
<font style = 'font-family:Book Antiqua'><font size = 4><p>There are other metrics that can be viewed to measure the success of the wage increase:    
- Number of home foreclosures.
- Amount of families lifted out of poverty.
 - This can be counted by the number of households that no longer rely on the following programs:
   - Supplemental Nutrition Assistance Program (SNAP)
   - The Special Supplemental Nutrition Program for Women, Infants, and Children (WIC)
   - Temporary Assistance for Needy Families (TANF)
   - Medicaid
   - Housing Choice Voucher Program/Section 8 Housing</p>

<br>
<br>
<font style = 'font-family:Book Antiqua'><font size = 5><i><b>Data Collection Approach:</b></i>
<font style = 'font-family:Book Antiqua'><font size = 4><p>To measure the affects of a wage increase on any of the metrics mentioned above, we use state and federal records to gather data and make observations about the population we have chosen to focus on. Specifically, we will use data from <a href='https://www.census.gov/programs-surveys/ahs.html'>American Housing Survey (AHS)</a> to calculate a change in home ownership for Manhattan and Seattle. Obtaining housing information for Honolulu may be more difficult as Hawaii does not participate in AHS surveys. Because our research will focus more on the low income population it might be better to narrow our scope and use data sets from HUD. The Resident Characteristics Report is released once a month and summarizes general information about households who reside in Public Housing or participate in Section 8.</p>
<p>Obtaining data to measure public assistance programs may be more difficult as most data is gathered by county or at the state level and rarely by metropolitian area. However, if the data allowed, we can break down the information by zip code and focus on those within Honolulu, Seattle, and Manhattan.
<br>
<br>
<font style = 'font-family:Book Antiqua'><font size = 5><i><b>Limitations:</b></i>
<font style = 'font-family:Book Antiqua'><font size = 4><p>There are a number of limitaions in this study:</p>

- The study assumes that all those who work in the the cities of Honolulu, Seattle, and Manhattan will also reside in those cities. This study there for does not take into account the residency of all workers in the city.
- This analysis does not take into account fluctiation of home prices and rents nor the effects a wage incresae could have on both.
- It is likely that with the increase among low-income earners, those in more senior and mamagement roles can also see a change in their owm waged as they increase to maintain pay scale. However this "wage spill" will not be reviewed in this paper.
- The time frame of this study can present problems and create variances within the data. A recession can cause higher unemployement and increase the popuation of households receiving government assistance as well as increase mortgage defaults, decreasing homeowenership rates. 

<br>
<br>
<font style = 'font-family:Book Antiqua'><font size = 5><i><b>Success Criteria:</b></i>
<font style = 'font-family:Book Antiqua'><font size = 4><p>I would not consider any increase in homeownership a failure. A 2% increase is aggressive, especially considering the relative short time frame of 5 years. When looking at anything related to policy changes, it will take time to see real changes. Though agressive, a 2% increase in homeownership is still feasible. Information from the <a href='https://fred.stlouisfed.org/series/RHORUSQ156N#0'>US Census Bureau shows</a> that a 2% increase is possible. The below chart shows a 1.9% increase in homeownership in the United States between 4/1/2016 and 10/1/2018.</p>


```python
homeownership = pd.read_csv('RHORUSQ156N (1).csv')
homeownership.head()

layout = dict(title = 'U.S. Homeownership Over Time',
              xaxis = dict(title = 'Year'), 
              yaxis = dict(title = 'Homeownership %'),
            annotations=[
                dict(
                    x='4/1/2016',
                    y=63.4,
                    xref='x',
                    yref='y',
                    text='62.9%',
                    showarrow=True,
                    bgcolor= 'yellow',
                    arrowhead=3,
                    ax=-10,
                    ay=-80
                ),
                dict(
                    x='10/1/2018',
                    y=64.8,
                    xref='x',
                    yref='y',
                    text='64.8%',
                    showarrow=True,
                    bgcolor= 'yellow',
                    arrowhead=3,
                    ax=10,
                    ay=-70
                ),
            ]
    )


# Create a trace
trace13 = go.Scatter(
     x = homeownership['DATE'],
     y = homeownership['Home_Ownership'],
     name = 'Average CPI',
     line = dict(
         color = ('rgb(102, 183, 163)')
     )
)     


data = [trace13]

fig = dict(data=data, layout=layout)
py.offline.iplot(fig, filename='styled-line')
```


<div id="ee4a7ae9-89d7-4ce3-8156-7071b0cf7696" style="height: 525px; width: 100%;" class="plotly-graph-div"></div><script type="text/javascript">require(["plotly"], function(Plotly) { window.PLOTLYENV=window.PLOTLYENV || {};window.PLOTLYENV.BASE_URL="https://plot.ly";
if (document.getElementById("ee4a7ae9-89d7-4ce3-8156-7071b0cf7696")) {
    Plotly.newPlot("ee4a7ae9-89d7-4ce3-8156-7071b0cf7696", [{"line": {"color": "rgb(102, 183, 163)"}, "name": "Average CPI", "x": ["1/1/1965", "4/1/1965", "7/1/1965", "10/1/1965", "1/1/1966", "4/1/1966", "7/1/1966", "10/1/1966", "1/1/1967", "4/1/1967", "7/1/1967", "10/1/1967", "1/1/1968", "4/1/1968", "7/1/1968", "10/1/1968", "1/1/1969", "4/1/1969", "7/1/1969", "10/1/1969", "1/1/1970", "4/1/1970", "7/1/1970", "10/1/1970", "1/1/1971", "4/1/1971", "7/1/1971", "10/1/1971", "1/1/1972", "4/1/1972", "7/1/1972", "10/1/1972", "1/1/1973", "4/1/1973", "7/1/1973", "10/1/1973", "1/1/1974", "4/1/1974", "7/1/1974", "10/1/1974", "1/1/1975", "4/1/1975", "7/1/1975", "10/1/1975", "1/1/1976", "4/1/1976", "7/1/1976", "10/1/1976", "1/1/1977", "4/1/1977", "7/1/1977", "10/1/1977", "1/1/1978", "4/1/1978", "7/1/1978", "10/1/1978", "1/1/1979", "4/1/1979", "7/1/1979", "10/1/1979", "1/1/1980", "4/1/1980", "7/1/1980", "10/1/1980", "1/1/1981", "4/1/1981", "7/1/1981", "10/1/1981", "1/1/1982", "4/1/1982", "7/1/1982", "10/1/1982", "1/1/1983", "4/1/1983", "7/1/1983", "10/1/1983", "1/1/1984", "4/1/1984", "7/1/1984", "10/1/1984", "1/1/1985", "4/1/1985", "7/1/1985", "10/1/1985", "1/1/1986", "4/1/1986", "7/1/1986", "10/1/1986", "1/1/1987", "4/1/1987", "7/1/1987", "10/1/1987", "1/1/1988", "4/1/1988", "7/1/1988", "10/1/1988", "1/1/1989", "4/1/1989", "7/1/1989", "10/1/1989", "1/1/1990", "4/1/1990", "7/1/1990", "10/1/1990", "1/1/1991", "4/1/1991", "7/1/1991", "10/1/1991", "1/1/1992", "4/1/1992", "7/1/1992", "10/1/1992", "1/1/1993", "4/1/1993", "7/1/1993", "10/1/1993", "1/1/1994", "4/1/1994", "7/1/1994", "10/1/1994", "1/1/1995", "4/1/1995", "7/1/1995", "10/1/1995", "1/1/1996", "4/1/1996", "7/1/1996", "10/1/1996", "1/1/1997", "4/1/1997", "7/1/1997", "10/1/1997", "1/1/1998", "4/1/1998", "7/1/1998", "10/1/1998", "1/1/1999", "4/1/1999", "7/1/1999", "10/1/1999", "1/1/2000", "4/1/2000", "7/1/2000", "10/1/2000", "1/1/2001", "4/1/2001", "7/1/2001", "10/1/2001", "1/1/2002", "4/1/2002", "7/1/2002", "10/1/2002", "1/1/2003", "4/1/2003", "7/1/2003", "10/1/2003", "1/1/2004", "4/1/2004", "7/1/2004", "10/1/2004", "1/1/2005", "4/1/2005", "7/1/2005", "10/1/2005", "1/1/2006", "4/1/2006", "7/1/2006", "10/1/2006", "1/1/2007", "4/1/2007", "7/1/2007", "10/1/2007", "1/1/2008", "4/1/2008", "7/1/2008", "10/1/2008", "1/1/2009", "4/1/2009", "7/1/2009", "10/1/2009", "1/1/2010", "4/1/2010", "7/1/2010", "10/1/2010", "1/1/2011", "4/1/2011", "7/1/2011", "10/1/2011", "1/1/2012", "4/1/2012", "7/1/2012", "10/1/2012", "1/1/2013", "4/1/2013", "7/1/2013", "10/1/2013", "1/1/2014", "4/1/2014", "7/1/2014", "10/1/2014", "1/1/2015", "4/1/2015", "7/1/2015", "10/1/2015", "1/1/2016", "4/1/2016", "7/1/2016", "10/1/2016", "1/1/2017", "4/1/2017", "7/1/2017", "10/1/2017", "1/1/2018", "4/1/2018", "7/1/2018", "10/1/2018", "1/1/2019"], "y": [62.9, 62.9, 62.9, 63.4, 63.5, 63.2, 63.3, 63.8, 63.3, 63.9, 63.8, 63.5, 63.6, 64.1, 64.1, 63.6, 64.1, 64.4, 64.4, 64.4, 64.3, 64.0, 64.4, 64.0, 64.0, 64.1, 64.4, 64.5, 64.3, 64.5, 64.3, 64.4, 64.9, 64.4, 64.4, 64.4, 64.8, 64.8, 64.6, 64.4, 64.4, 64.9, 64.6, 64.5, 64.6, 64.6, 64.9, 64.8, 64.8, 64.5, 65.0, 64.9, 64.8, 64.4, 65.2, 65.4, 64.8, 64.9, 65.8, 65.4, 65.5, 65.5, 65.8, 65.5, 65.6, 65.3, 65.6, 65.2, 64.8, 64.9, 64.9, 64.5, 64.7, 64.7, 64.8, 64.4, 64.6, 64.6, 64.6, 64.1, 64.1, 64.1, 63.9, 63.5, 63.6, 63.8, 63.8, 63.9, 63.8, 63.8, 64.2, 64.1, 63.7, 63.7, 64.0, 63.8, 63.9, 63.8, 64.1, 63.8, 64.0, 63.7, 64.0, 64.1, 63.9, 63.9, 64.2, 64.2, 64.0, 63.9, 64.3, 64.4, 64.2, 64.4, 64.7, 64.6, 63.8, 63.8, 64.1, 64.2, 64.2, 64.7, 65.0, 65.1, 65.1, 65.4, 65.6, 65.4, 65.4, 65.7, 66.0, 65.7, 65.9, 66.0, 66.8, 66.4, 66.7, 66.6, 67.0, 66.9, 67.1, 67.2, 67.7, 67.5, 67.5, 67.7, 68.1, 68.0, 67.8, 67.6, 68.0, 68.3, 68.0, 68.0, 68.4, 68.6, 68.6, 69.2, 69.0, 69.2, 69.1, 68.6, 68.8, 69.0, 68.5, 68.7, 69.0, 68.9, 68.4, 68.2, 68.2, 67.8, 67.8, 68.1, 67.9, 67.5, 67.3, 67.4, 67.6, 67.2, 67.1, 66.9, 66.9, 66.5, 66.4, 65.9, 66.3, 66.0, 65.4, 65.5, 65.5, 65.4, 65.0, 65.0, 65.3, 65.2, 64.8, 64.7, 64.4, 64.0, 63.7, 63.4, 63.7, 63.8, 63.5, 62.9, 63.5, 63.7, 63.6, 63.7, 63.9, 64.2, 64.2, 64.3, 64.4, 64.8, 64.2], "type": "scatter", "uid": "9a8c94d0-7c4e-4f23-84ae-8a7fac64186f"}], {"annotations": [{"arrowhead": 3, "ax": -10, "ay": -80, "bgcolor": "yellow", "showarrow": true, "text": "62.9%", "x": "4/1/2016", "xref": "x", "y": 63.4, "yref": "y"}, {"arrowhead": 3, "ax": 10, "ay": -70, "bgcolor": "yellow", "showarrow": true, "text": "64.8%", "x": "10/1/2018", "xref": "x", "y": 64.8, "yref": "y"}], "title": {"text": "U.S. Homeownership Over Time"}, "xaxis": {"title": {"text": "Year"}}, "yaxis": {"title": {"text": "Homeownership %"}}}, {"showLink": false, "linkText": "Export to plot.ly", "plotlyServerURL": "https://plot.ly"}); 
}
});</script><script type="text/javascript">window.addEventListener("resize", function(){if (document.getElementById("ee4a7ae9-89d7-4ce3-8156-7071b0cf7696")) {window._Plotly.Plots.resize(document.getElementById("ee4a7ae9-89d7-4ce3-8156-7071b0cf7696"));};})</script>


<font style = 'font-family:Book Antiqua'><font size = 4><p>If there is no increase in homeownership, there are other ways in which the success of a minimum wage increase can be measured. An overall decrease in the level of households living at or below 80% of AMI and a decrease in households relying on government assistance programs can be considered a success. Historically, owning a home has become one of the American milestones by which success is measured but maybe this view is dated and should change. The majority of a households income is in the value of their home, however this is not the only investment that is available. If an increase in wages can lift households out of poverty and give families more disposable income, they may be more likely to take part in other investment avenues that can increase their net worth.</p>
