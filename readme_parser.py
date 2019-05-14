import os
import re
import pandas as pd


# Code adapted from https://github.com/arsen-movsesyan/springboard_WESAD/blob/master/parsers/readme_parser.py
VALUE_EXTRACT_KEYS = {
    "age": {
        'search_key': 'Age',
        'delimiter': ':'
    },
    "height": {
        'search_key': 'Height',
        'delimiter': ':'
    },
    "weight": {
        'search_key': 'Weight',
        'delimiter': ':'
    },
    "gender": {
        'search_key': 'Gender',
        'delimiter': ':'
    },
    "dominant_hand": {
        'search_key': 'Dominant',
        'delimiter': ':'
    },
    "coffee_today": {
        'search_key': 'Did you drink coffee today',
        'delimiter': '? '
    },
    "coffee_last_hour": {
        'search_key': 'Did you drink coffee within the last hour',
        'delimiter': '? '
    },
    "sport_today": {
        'search_key': 'Did you do any sports today',
        'delimiter': '? '
    },
    "smoker": {
        'search_key': 'Are you a smoker',
        'delimiter': '? '
    },
    "smoke_last_hour": {
        'search_key': 'Did you smoke within the last hour',
        'delimiter': '? '
    },
    "feel_ill_today": {
        'search_key': 'Do you feel ill today',
        'delimiter': '? '
    }
}


def parse_readme(subject_id):
    with open(readme_locations[subject_id] + subject_id + parse_file_suffix, 'r') as f:

        x = f.read().split('\n')

    readme_dict = {}

    for item in x:
        for key in VALUE_EXTRACT_KEYS.keys():
            search_key = VALUE_EXTRACT_KEYS[key]['search_key']
            delimiter = VALUE_EXTRACT_KEYS[key]['delimiter']
            if item.startswith(search_key):
                d, v = item.split(delimiter)
                readme_dict.update({key: v})
                break
    return readme_dict


def parse_all_readmes():
    DATA_PATH = 'data/WESAD/'
    parse_file_suffix = '_readme.txt'

    readme_locations = {subject_directory: DATA_PATH + subject_directory + '/' 
                      for subject_directory in os.listdir(DATA_PATH)
                          if re.match('^S[0-9]{1,2}$', subject_directory)}

    dframes = []

    for subject_id, path in readme_locations.items():
        readme_dict = parse_readme(subject_id)
        df = pd.DataFrame(readme_dict, index=[subject_id])
        dframes.append(df)

    df = pd.concat(dframes)
    df.to_csv(DATA_PATH + 'readmes.csv')
    
    
def parser_main():
    
    if not os.path.isfile('data/WESAD/readmes.csv'):
        parse_all_readmes()

    df = pd.read_csv('data/WESAD/readmes.csv', index_col=0)

    dummy_df = pd.get_dummies(df)

    feat_df = pd.read_csv('may14_feats4.csv', index_col=0)

    dummy_df['subject'] = dummy_df.index.str[1:].astype(int)

    dummy_df = dummy_df[['age', 'height', 'weight', 'gender_ female', 'gender_ male',
                       'coffee_today_YES', 'sport_today_YES', 'smoker_NO', 'smoker_YES',
                       'feel_ill_today_YES', 'subject']]

    merged_df = pd.merge(feat_df, dummy_df, on='subject')

    merged_df.to_csv('m14_merged.csv')