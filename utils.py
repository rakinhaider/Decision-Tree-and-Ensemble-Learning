import pandas as pd

categorical = ['gender',
               'race',
               'race_o',
               'field']

psParticipants = ['attractive_important',
                  'sincere_important',
                  'intelligence_important',
                  'funny_important',
                  'ambition_important',
                  'shared_interests_important']

psPartners = ['pref_o_attractive',
              'pref_o_sincere',
              'pref_o_intelligence',
              'pref_o_funny',
              'pref_o_ambitious',
              'pref_o_shared_interests']

notContinuousValuedColumns = ['gender',
                              'race',
                              'race_o',
                              'samerace',
                              'field',
                              'decision']

rPP = ['attractive_partner',
       'sincere_partner',
       'intelligence_partner',
       'funny_partner',
       'ambition_partner',
       'shared_interests_partner']

final = True
binCount = 5
target_attribute = 'decision'


def readFile(filename, index_col=0):
    data = pd.read_csv(filename, index_col=index_col)
    return list(data.columns), data
