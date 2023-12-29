import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn import metrics
import plotly.express as px

def PredictArrests(crimes, class_weight=None, max_depth=None):
  x = crimes[['Arrest', 'Beat', 'District', 'Ward', 'Community Area']]

  x['Beat'].replace('', np.nan, inplace=True)
  x['District'].replace('', np.nan, inplace=True)
  x['Ward'].replace('', np.nan, inplace=True)
  x['Community Area'].replace('', np.nan, inplace=True)

  x = x.dropna(subset=['Beat', 'District', 'Ward', 'Community Area'])

  y = x['Arrest']
  
  x = x.drop(columns='Arrest')

  y =y.astype('int')
  x =x.astype('int')

  if class_weight is not None:
    clf = tree.DecisionTreeClassifier(max_depth=max_depth, class_weight=class_weight)
  else:
    clf = tree.DecisionTreeClassifier(max_depth=4)

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=True)

  clf = clf.fit(x_train, y_train)

  y_pred = clf.predict(x_test)

  x = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
  
  print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

  return x

def PredictArrestsByLocation(crimes, class_weight=None, max_depth=None):
  crimes = crimes.dropna()
  x = crimes[['Arrest', 'Latitude', 'Longitude']]

  y = x['Arrest']
  
  x = x.drop(columns='Arrest')

  y = y.astype('int')
  x = x.astype('int')
  
  clf = tree.DecisionTreeClassifier(class_weight=class_weight, max_depth=max_depth)

  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

  clf = clf.fit(x_train, y_train)

  y_pred = clf.predict(x_test)

  x = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
  
  print("Accuracy:", metrics.accuracy_score(y_test, y_pred))

  return x

def ArrestsMade(crimes):
    crimes['cnt'] = 1
    return crimes.groupby(['Arrest']).agg({'cnt':'sum'})

def BaselinePrediction(crimes):
    mode = 0
    crimes = crimes.dropna()
    x = crimes['Arrest']
    x = x.astype('int')
    vals, counts = np.unique(x, return_counts=True)
    mode = np.argmax(counts)
    res = np.ndarray(shape=x.shape[0])
    res.fill(mode)
    return metrics.accuracy_score(x, res)

def ComputeClassWeights(crimes):
   return compute_class_weight(class_weight = 'balanced', classes = np.unique(crimes), y = crimes)
#
# Returns the crime data after performing all the filtering/cleaning
# All filtering should be done by other functions and called inside this
# function
#
def CrimesAfterFiltering(crimes):
  _crimes = FilterLocation(crimes)
  
  _crimes = CrimeTypeNotListed(_crimes)
 
  _crimes = FilterDescription(_crimes)

  # Drop all values of empty in location description
  _crimes['Location Description'].replace('', np.nan, inplace=True)
  _crimes.dropna(subset=['Location Description'], inplace=True)

  # Filter years
  _crimes = _crimes[((_crimes['Year'] >= 2001) & (_crimes['Year'] <= 2022))]

  # Returns number of rows in the dataframe
  # return len(_crimes.index)

  # Return the dataframe
  return _crimes

def FilterDescription(crimes):
  return crimes[(crimes['Description'].isnull() == False)]

def FilterLocation(crimes):
  return crimes[(crimes['Location'].isnull() == False)]

def CrimeByYears(crimes):
  return crimes.groupby(['Year']).size()

def CrimeTypeNotListed(crimes):
  return crimes[(crimes['Primary Type'].isnull() == False)]

def get_sorted_data(crimes):
  crimes['cnt'] = 1
  crimes = crimes[((crimes['Year'] >= 2004) & (crimes['Year'] < 2023)) & ((crimes['Primary Type'] == "THEFT")| 
                                              (crimes['Primary Type'] == "BATTERY")|
                                              (crimes['Primary Type'] == "CRIMINAL DAMAGE")|
                                              (crimes['Primary Type'] == "NARCOTICS")|
                                              (crimes['Primary Type'] == "ASSAULT")
                                              )]
  return crimes.groupby(['Year', 'Primary Type']).agg({'cnt':'sum'})
  
def get_total_crimes(crimes):
  crimes['cnt'] = 1
  filter1 = crimes["Year"]>2003
  crimes.where(filter1, inplace = True)
  return crimes.groupby(['Primary Type']).agg({'cnt':'sum'})


def Plot_crimes_overyears(crimes):
    a = get_sorted_data(crimes)
    file3 = a.reset_index()
    ax = sns.lineplot(data=file3, x="Year", y="cnt", hue="Primary Type", style="Primary Type")
    b = ax.set_xticks(range(2004,2023,2))
    c = ax.set_ylabel("Number of Crimes Committed")
    
def CrimesnearUIC(crimes):
    crimes['count'] = 1
    crimes1 = crimes
    crimes1 = crimes1.where(crimes1["Community Area"] == 28.0, inplace = True)
    crimes1 = crimes.groupby(['Latitude', 'Longitude'], group_keys = True).agg({'count':'sum'})
    crimes1.reset_index(inplace=True)
    graph = px.density_mapbox(crimes1, lat='Latitude', lon='Longitude', z='count',zoom = 13 ,radius = 25,color_continuous_midpoint = 1240, mapbox_style="stamen-terrain")
    return graph

def plot_crime_areas(crimes):
  crimes['count'] = 1
  locations = crimes.groupby(['Block']).agg({'count':'sum'})
  locations = locations.sort_values(['Block', 'count'])
  locations = locations.head(100)
  return locations
  # debugging needed
  ax = sns.barplot(data=locations, x='Block', y='count', hue="Primary Type", order=locations.sort_values('Block').cnt)
  yLab = ax.set_ylabel("Number of Crimes Committed")
  xLab = ax.set_xlabel(locations['Block'])


def sort_crimes(crimes):
  return crimes.sort_values(['cnt'], ascending=[False]).head(5)

def filterCommunityAreas(crimes, comAreas):
    crimes['cnt'] = 1
    merged = pd.merge(crimes, comAreas, left_on='Community Area', right_on='Community Area Number', how='left')
    merged.groupby('COMMUNITY AREA NAME').count()['cnt'].sort_values(ascending=False)
    filtered = merged.filter(['ID', 'Primary Type', 'Location Description', 'COMMUNITY AREA NAME', 'cnt', 'Year', 'Arrest'])
    gb1 = filtered.groupby(["COMMUNITY AREA NAME"]).agg({'cnt':'sum'})
    gb2 = filtered.groupby(["COMMUNITY AREA NAME", 'Arrest']).agg({'cnt':'sum'})
    gb2 = gb2.reset_index()
    filt = gb2['Arrest'] == True
    gb2 = gb2.where(filt)
    gb2 = gb2.dropna()

    gb1.rename(columns = {'cnt':'TrueVals'}, inplace = True)
    gb2 = gb2.reset_index()
    gb2 = gb2.drop(['Arrest', 'index'], axis=1)
    gb2.rename(columns = {'cnt':'Total'}, inplace = True)

    data = pd.merge(gb1, gb2, on="COMMUNITY AREA NAME", how='inner')
    data = data.sort_values('Total', ascending=False)
    data = data.head(20)
    return data

def plotArrestsOnCA(data):
    data = data[(data['COMMUNITY AREA NAME'] != "CHICAGO")]
    data = data.sample(frac = 1)
    sns.set_color_codes("pastel")
    sns.barplot(x="TrueVals", y="COMMUNITY AREA NAME", data=data,
                label="Total Crime Aggregate", color="b")

    sns.set_color_codes("muted")
    sns.barplot(x="Total", y="COMMUNITY AREA NAME", data=data,
                label="Total Arrest Made", color="b")

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)
    
def plotHeatmap(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df['Hour'] = df['Date'].dt.hour
    plt.figure(figsize=(10, 6))
    crime_by_location_time = df.groupby(['Location Description', 'Hour']).size().unstack()
    sns.heatmap(crime_by_location_time, cmap='YlGnBu')
    plt.title('Heat Map of Crime Locations and Time of Day')
    plt.show()

def socioEcon(crimes, comAreas):
    crimes['cnt'] = 1
    gb = crimes.groupby(['Community Area']).agg({'cnt':'sum'})
    gb.reset_index()
    merged = pd.merge(gb, comAreas, left_on='Community Area', right_on='Community Area Number', how='left')
    merged = merged.rename(columns={"cnt": "Total Crime"})
    merged = merged.drop(merged.index[0])
    merged.drop('Community Area Number', inplace=True, axis=1)
    merged = merged.sort_values(by='Total Crime', ascending=False)
    merged = merged.head(10)
    del merged['Total Crime']
    return merged

def plotBelowPovertyHouseholdRelation(merged):
    ax = sns.barplot(x="COMMUNITY AREA NAME", y="PERCENT HOUSEHOLDS BELOW POVERTY", data=merged,
                label="Portfolio", color="g")
    a = ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")

def plotHSDrelation(merged):
    ax = sns.barplot(x="COMMUNITY AREA NAME", y="PERCENT HOUSEHOLDS BELOW POVERTY", data=merged,
                label="Portfolio", color="g")
    a = ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")