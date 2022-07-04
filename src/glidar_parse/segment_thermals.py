
from sklearn.cluster import DBSCAN
import pandas as pd


def segment_thermals(dataFrame: pd.DataFrame):
    """
    Clusters a point dataset into thermals
    @param dataFrame: pandas data frame with the flight data
    @ returns data frame with labels for each thermal
    """

    positiveVario = dataFrame[dataFrame['vario'] >= 0]

    positiveVario['time_sec'] = (dataFrame.time - dataFrame.time.min()).dt.total_seconds()

    fit = pd.DataFrame(positiveVario[['x', 'y', 'time_sec']])

    fit['time_sec'] = fit['time_sec'] * 0.5
    clustering = DBSCAN(eps=15, min_samples=3).fit(fit)
    positiveVario = pd.DataFrame(positiveVario)
    positiveVario['labels'] = clustering.labels_

    return positiveVario