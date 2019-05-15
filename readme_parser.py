import os
import re
import pandas as pd


class rparser:
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
    
    DATA_PATH = 'data/WESAD/'
    parse_file_suffix = '_readme.txt'
    
    
    def __init__(self):
        
        self.readme_locations = {subject_directory: self.DATA_PATH + subject_directory + '/' 
                          for subject_directory in os.listdir(self.DATA_PATH)
                              if re.match('^S[0-9]{1,2}$', subject_directory)}
        
        # Check if parsed readme file is available ( should be as it is saved above )
        if not os.path.isfile('data/readmes.csv'):
            print('Parsing Readme files')
            self.parse_all_readmes()
        else:
            print('Files already parsed.')
            
        self.merge_with_feature_data()
        
        
    def parse_readme(self, subject_id):
        with open(self.readme_locations[subject_id] + subject_id + self.parse_file_suffix, 'r') as f:

            x = f.read().split('\n')

        readme_dict = {}

        for item in x:
            for key in self.VALUE_EXTRACT_KEYS.keys():
                search_key = self.VALUE_EXTRACT_KEYS[key]['search_key']
                delimiter = self.VALUE_EXTRACT_KEYS[key]['delimiter']
                if item.startswith(search_key):
                    d, v = item.split(delimiter)
                    readme_dict.update({key: v})
                    break
        return readme_dict


    def parse_all_readmes(self):
        
        dframes = []

        for subject_id, path in self.readme_locations.items():
            readme_dict = self.parse_readme(subject_id)
            df = pd.DataFrame(readme_dict, index=[subject_id])
            dframes.append(df)

        df = pd.concat(dframes)
        df.to_csv(self.DATA_PATH + 'readmes.csv')

        
    def merge_with_feature_data(self):
        # Confirm feature files are available
        if os.path.isfile('data/may14_feats4.csv'):
            feat_df = pd.read_csv('data/may14_feats4.csv', index_col=0)
        else:
            print('No feature data available. Exiting...')
            return
           
        # Combine data and save
        df = pd.read_csv(f'{self.DATA_PATH}readmes.csv', index_col=0)

        dummy_df = pd.get_dummies(df)
        
        dummy_df['subject'] = dummy_df.index.str[1:].astype(int)

        dummy_df = dummy_df[['age', 'height', 'weight', 'gender_ female', 'gender_ male',
                           'coffee_today_YES', 'sport_today_YES', 'smoker_NO', 'smoker_YES',
                           'feel_ill_today_YES', 'subject']]

        merged_df = pd.merge(feat_df, dummy_df, on='subject')

        merged_df.to_csv('data/m14_merged.csv')
        