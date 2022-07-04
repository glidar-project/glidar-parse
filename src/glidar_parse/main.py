import pandas as pd
import numpy as np

import os
import sys
import argparse

from pyproj import Proj
from tqdm.cli import tqdm

# from glidar_parse.cesium_projection import CesiumProjection
from glidar_parse.kml_parser import KmlParser
from glidar_parse.igc_parser import IgcParser

def parse_files(listOfFiles):

    flights = []
    print('Parsing...')
    for f in tqdm(listOfFiles):
        if f[-3:].lower() == 'kml':
            flights.append(KmlParser(f))

    frames = []
    for flight in flights:
        df = flight.track
        frames.append(df)

    return pd.concat(frames)


def search_folder(folder):

    # Get the list of all files in directory tree at given path
    listOfFiles = list()
    for (dirpath, dirnames, filenames) in os.walk(folder):
        listOfFiles += [os.path.join(dirpath, file) for file in filenames]

    return listOfFiles

    
def parse_file(filename):
    
    if not os.path.isfile(filename):
        raise ValueError(filename)
    
    track = None
    f_ext = filename[-4:].lower()
    if f_ext == '.kml':
        track = KmlParser(filename)
    elif f_ext == '.igc':
        track = IgcParser(filename)

    if track is None:
        raise ValueError('Unknown file format:', f_ext)

    return track

def main():

    parser = argparse.ArgumentParser(description='Optional app description')
    # Optional positional argument
    parser.add_argument('file', type=str, nargs='+',
                        help='Name of the tracklog file to parse. Supported types: .kml, .igc')

    # Switch
    parser.add_argument('-r', action='store_true',
                        help='Recurrent search of tracklog files in a folder')

    parser.add_argument("-o", type=str,
                        help="Output file name")

    args = parser.parse_args()

    print(args)
    
    file_list = filter(lambda f: os.path.isfile(f), args.file)
    folder_list = filter(lambda f: os.path.isdir(f), args.file)
    
    if args.r: # If recurrent search enabled
        for folder in folder_list:
            file_list += search_folder(folder_list)

    data = parse_files(file_list)

    o_file = 'data.csv'
    if parser.o is not None:
        o_file = parser.o

    print(data)
    data.to_csv(o_file)


if __name__ == '__main__':

    main()

