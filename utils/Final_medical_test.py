### Import Libraries
import numpy as np              
import pandas as pd             
from scipy import stats         
import re

### Input info
min_age = 18                        # at least
max_age = 100                       # at most
false_positive_percent = 15         # smaller
false_negative_percent = -100       # smaller & negative is removed.
fixation_loss_percent = 33          # smaller

number_of_rows_to_delete = 1    
vf_test_number = 5              
minimum_days = 28               
group_period = 365*2            
number_of_rows_to_delete = 2    


### Make Code
def make_Exam_Age(dataframe):
    dataframe['DOB'] = pd.to_datetime(dataframe['DOB'],format='mixed')
    dataframe['Exam Date'] = pd.to_datetime(dataframe['Exam Date'].str.split(" ", expand=True)[0])
    dataframe['Exam Age'] = dataframe['Exam Date'].dt.year - dataframe['DOB'].dt.year
    return dataframe

def make_lid(dataframe):
    data = dataframe.copy()

    _, id = np.unique(data['PID'].astype(str) + data['SchoolName'], return_inverse=True)
    data['id'] = id
    
    _, lid = np.unique(data['PID'].astype(str) + data['Eye'] + data['SchoolName'], return_inverse=True)
    data['lid'] = lid

    return data

def good_patients(dataframe,min_age,max_age,false_positive_percent,false_negative_percent,fixation_loss_percent):
    print('# init data shape : {0}'.format(dataframe.shape))
    print('# lid(안구) : {0}'.format(len(dataframe['lid'].unique())))
    print('# id(환자) : {0}'.format(len(dataframe['id'].unique())))
    
    dataframe = dataframe.drop_duplicates(['lid', 'Exam Date'],keep='first')
    print('# Drop_duplicates : {0}'.format(dataframe.shape[0]))
    
    dataframe = dataframe[(dataframe['Exam Age'] >= min_age) & (dataframe['Exam Age'] <= max_age)]
    print('# Age range {0} to {1} : {2}'.format(min_age,max_age,dataframe.shape[0]))
        
    dataframe = dataframe[dataframe['False positive percent'] < false_positive_percent]
    print('# False positive percent < {0} : {1}'.format(false_positive_percent,dataframe.shape[0]))

    dataframe = dataframe[dataframe['False negative percent'] > false_negative_percent]
    print('# False negative percent > {0} : {1}'.format(false_negative_percent,dataframe.shape[0]))

    dataframe.insert(loc=19,
            column='Fixation loss percent',
            value=100*dataframe['Fixation loss'] / dataframe['Fixation trial'])
    dataframe = dataframe[dataframe['Fixation loss percent'] < fixation_loss_percent]
    print('# Fixation loss percent < {0} : {1}'.format(fixation_loss_percent,dataframe.shape[0]))
        
    print('# good_patients data: ',dataframe.shape[0])
    return dataframe

def drop_rows(dataframe, number_of_rows_to_delete):
    dataframe.sort_values(by=('lid'), inplace=True)
    drop_indices = []
    print('** unique lid : ', len(dataframe.lid.unique()))
    for idx in dataframe.lid.unique():
        drop_indices.append(dataframe[dataframe.lid == idx].index[0])    
    dataframe.drop(drop_indices, inplace=True)
    return dataframe

def make_days(dataframe):
    df = dataframe.sort_values(by=['lid', 'Exam Date']).copy()
    df['Exam Date'] = pd.to_datetime(df['Exam Date'])
    first_exam_dates = df.groupby('lid')['Exam Date'].transform('min')
    df['days'] = (df['Exam Date'] - first_exam_dates).dt.days
    return df

def filter_lids_by_test_count(dataframe, vf_test_number):
    lid_counts = dataframe['lid'].value_counts()
    keep_lids = lid_counts[lid_counts >= vf_test_number].index
    return dataframe[dataframe['lid'].isin(keep_lids)]

def filter_5(dataframe):
    lid_counts = dataframe['lid'].value_counts()
    keep_lids = lid_counts[lid_counts > 5].index
    pp = dataframe[dataframe['lid'].isin(keep_lids)]
    print('# 5번 초과 시야검사 받은 사람 : ', pp.id.unique().shape[0])
    print('이들의 시야검사 평균 :', round(pp.groupby('id').size().mean(),3))
    return dataframe

def remove_close_days(dataframe, minimum_days=28):
    df = dataframe.copy()
    df = df.sort_values(by=['lid', 'days'])

    filtered_rows = []

    for lid, group in df.groupby('lid'):
        days = group['days'].tolist()
        filtered_days = [days[0]]
        for d in days[1:]:
            if d - filtered_days[-1] >= minimum_days:
                filtered_days.append(d)

        filtered_group = group[group['days'].isin(filtered_days)]
        filtered_rows.append(filtered_group)

    remove_close_days_df = pd.concat(filtered_rows).reset_index(drop=True)

    final_lid = []
    for lid, group in remove_close_days_df.groupby('lid'):
        if len(group) >= 5:
            final_lid.append(group)
    remove_close_days_df = pd.concat(final_lid).reset_index(drop=True)
    return remove_close_days_df


def divide_dataframe(dataframe, group_period=365*2):
    df = dataframe.copy()
    df = df.sort_values(by=['lid', 'days'])
    new_df = pd.DataFrame()

    for lid, group in df.groupby('lid'):
        group = group.reset_index(drop=True)
        days = group['days'].tolist()

        start_idx = 0
        sub_id = 0  

        for j in range(len(days) - 1):
            if days[j + 1] - days[j] > group_period:
                temp = group.iloc[start_idx:j + 1].copy()
                start_idx = j + 1

                if temp.shape[0] >= 5:
                    min_day = temp['days'].min()
                    temp['days'] = temp['days'] - min_day
                    temp['eye_episode'] = f"{lid}{sub_id}"
                    sub_id += 1
                    new_df = pd.concat([new_df, temp], ignore_index=True)

        temp = group.iloc[start_idx:].copy()
        if temp.shape[0] >= 5:
            min_day = temp['days'].min()
            temp['days'] = temp['days'] - min_day
            temp['eye_episode'] = f"{lid}{sub_id}"
            sub_id += 1
            new_df = pd.concat([new_df, temp], ignore_index=True)
    return new_df


def medical_test(dataframe, agis_test=True, cigts_test=True, md_slope=True, vfi_slope=True, plr_test=True):
    
    def AGIS_test(dataframe):
        dataframe['AGIS_test'] = 0
        for lid_num in dataframe.lid.unique():
            temp = dataframe[dataframe.lid == lid_num].AGIS
            baseline_agis = (temp.iloc[0] + temp.iloc[1]) / 2
            for jdx in range(len(temp) - 2):
                if (temp.iloc[jdx + 2] - baseline_agis >= 4) & (temp.iloc[jdx + 1] - baseline_agis >= 4) & (temp.iloc[jdx] - baseline_agis >= 4):
                    dataframe.loc[temp.index, 'AGIS_test'] = 1
                    break
        return dataframe

    def CIGTS_test(dataframe):
        dataframe['CIGTS_test'] = 0
        for lid_num in dataframe.lid.unique():
            temp = dataframe[dataframe.lid == lid_num].CIGTS
            baseline_CIGTS = (temp.iloc[0] + temp.iloc[1]) / 2
            for jdx in range(len(temp) - 2):
                if (temp.iloc[jdx + 2] - baseline_CIGTS >= 3) & (temp.iloc[jdx + 1] - baseline_CIGTS >= 3) & (temp.iloc[jdx] - baseline_CIGTS >= 3):
                    dataframe.loc[temp.index, 'CIGTS_test'] = 1
                    break
        return dataframe

    def MD_slope(dataframe):
        dataframe['MD_slope_value'] = 0.0
        dataframe['MD_slope_test'] = 0
        for lid_num in dataframe.lid.unique():
            temp = dataframe[dataframe.lid == lid_num]
            y_variation = temp.MD
            x_variation = np.array(temp.days) / 365.25
            model = stats.linregress(x_variation, y_variation)
            if (model.slope < 0.0) & (model.pvalue < 0.05):
                dataframe.loc[temp.index, 'MD_slope_test'] = 1
                dataframe.loc[temp.index, 'MD_slope_value'] = model.slope
        return dataframe

    def VFI_slope(dataframe):
        dataframe['VFI_slope_value'] = 0.0
        dataframe['VFI_slope_test'] = 0
        for lid_num in dataframe.lid.unique():
            temp = dataframe[dataframe.lid == lid_num]
            y_variation = temp.VFI
            x_variation = np.array(temp.days) / 365.25
            model = stats.linregress(x_variation, y_variation)
            if (model.slope < 0.0) & (model.pvalue < 0.05):
                dataframe.loc[temp.index, 'VFI_slope_test'] = 1
                dataframe.loc[temp.index, 'VFI_slope_value'] = model.slope
        return dataframe

    def PLR_test(dataframe):
        dataframe['PLR_test'] = 0
        for lid_num in dataframe.lid.unique():
            temp = dataframe[dataframe.lid == lid_num]            
            TDV_collect = temp.filter(regex='TDV_').copy()
            TDV_collect['TDV_26'] = np.NaN
            TDV_collect['TDV_35'] = np.NaN
            count = 0
            x_variation = np.array(temp.days) / 365.25
            for col_idx in range(TDV_collect.shape[1]):
                y_variation = TDV_collect.iloc[:, col_idx].tolist()
                model = stats.linregress(x_variation, y_variation)
                if (model.slope <= -1.0) & (model.pvalue < 0.01):
                    count += 1
                    if count >= 3:
                        dataframe.loc[temp.index, 'PLR_test'] = 1
                        break
        return dataframe

    if agis_test:
        dataframe = AGIS_test(dataframe)
    
    if cigts_test:
        dataframe = CIGTS_test(dataframe)
    
    if md_slope:
        dataframe = MD_slope(dataframe)
    
    if vfi_slope:
        dataframe = VFI_slope(dataframe)
    
    if plr_test:
        dataframe = PLR_test(dataframe)
    
    return dataframe

def wiggs_test(dataframe, event_based=True, trend_based_MD=True, trend_based_PDV=True):
    def Event_based(dataframe):
        df = dataframe.copy()
        df['Event_based_label'] = 0
        for lid, group in df.groupby('lid'):
            temp = group.filter(like='Wg_')
            for col in temp.columns:
                s = temp[col]
                baseline_zero = (s.iloc[0] == 0)
                last_two_one = (s.iloc[-2:].eq(1).all(axis=0))
                if (baseline_zero & last_two_one).all():
                    df.loc[df['lid']==lid, 'Event_based_label'] = 1
                    break
        return df

    def Trend_based_MD(dataframe):
        df = dataframe.copy()
        df['Trend_based_MD_label'] = 0
        for lid, group in df.groupby('lid'):
            temp = group.MD
            if np.mean(temp.iloc[-2:]) - np.mean(temp.iloc[:2]) <= -3:
                df.loc[df['lid']==lid, 'Trend_based_MD_label'] = 1
        return df
    
    def Trend_based_PDV(dataframe):
        Wg_map = ['Wg_SN', 'Wg_SB', 'Wg_ST', 'Wg_IN', 'Wg_IB', 'Wg_IT', 'Wg_SP', 'Wg_IP']
        Wg_SN = ['05','11','12','19','20','21']
        Wg_SB = ['06','07','08','09','13','14','15','16']
        Wg_ST = ['10','17','18','27']
        Wg_IN = ['28','29','30','37','38','45']
        Wg_IB = ['39','40','41','42','46','47','48','49','51','52','53','54']
        Wg_IT = ['36','43','44','50']
        Wg_SP = ['13','14','15','16','22','23','24','25']
        Wg_IP = ['31','32','33','34','39','40','41','42']

        code_map = {
        'Wg_SN': Wg_SN, 'Wg_SB': Wg_SB, 'Wg_ST': Wg_ST, 'Wg_IN': Wg_IN,
        'Wg_IB': Wg_IB, 'Wg_IT': Wg_IT, 'Wg_SP': Wg_SP, 'Wg_IP': Wg_IP
        }
    
        df = dataframe.copy()
        df['Trend_based_pdv_label'] = 0
        for lid, group in df.groupby('lid'):
            for c in Wg_map:
                temp = group.filter(like=c)
                if (temp.iloc[-2].eq(1) & temp.iloc[-1].eq(1)).all():
                    pat = r'^PDV_(?:' + '|'.join(map(re.escape, code_map[c])) + r')$'
                    pdv_ls = group.filter(regex=pat)
                    first = (10**(pdv_ls.iloc[:2]*0.1)).stack().mean()
                    last =  (10**(pdv_ls.iloc[-2:]*0.1)).stack().mean()
                    if 10*np.log10(last) - 10*np.log10(first) <= -3:
                        df.loc[df['lid']==lid, 'Trend_based_pdv_label'] = 1
                        break
        return df
    if event_based:
        dataframe = Event_based(dataframe)
    
    if trend_based_MD:
        dataframe = Trend_based_MD(dataframe)
    
    if trend_based_PDV:
        dataframe = Trend_based_PDV(dataframe)
    
    return dataframe

def fix_length(dataframe, length=5):
    df = dataframe.copy()
    df = df.sort_values(by=['eye_episode', 'days'])
    new_id = 0
    
    filtered_rows = []
    for eye_episode, group in df.groupby('eye_episode'):
        if len(group) == length:
            group['eye_episode'] = f"{eye_episode}{new_id}"
            new_id += 1
            filtered_rows.append(group)
        elif len(group) > length:
            original_length = len(group)
            chunk = original_length // length
            for i in range(chunk):
                start_idx = original_length - (i*length)
                end_idx = start_idx - length
                temp = group.iloc[end_idx:start_idx].copy()
                if len(temp) == length:
                    min_day = temp.days.min()
                    temp['days'] = temp['days'] - min_day
                    temp['eye_episode'] = f"{eye_episode}{new_id}"
                    new_id += 1
                    filtered_rows.append(temp)

    fixed_length_df = pd.concat(filtered_rows).reset_index(drop=True)
    return fixed_length_df

def sublid_less_than_2y(dataframe):
    df = dataframe.copy()
    df['days'] = df['days'].astype(int)
    df = df.sort_values(by=['eye_episode', 'days'])

    valid_indices = []

    for eye_episode, group in df.groupby('eye_episode'):
        days = group['days']
        if days.max() - days.min() > 730:
            valid_indices.extend(group.index)
    return df.loc[valid_indices].reset_index(drop=True)

def consensus_deterioration(dataframe):
    test_cols = dataframe.filter(like='_test')
    temp = test_cols.sum(axis=1) >= 1
    dataframe['Consensus_label'] = temp.astype('int')
    return dataframe

def wiggs_deterioration(dataframe):
    test_cols = dataframe.filter(like='based')
    test_cols = test_cols.fillna(0)    
    temp = test_cols.sum(axis=1) >= 1       
    dataframe['Wiggs_label'] = temp.astype('int')    
    return dataframe

def mean_diff(dataframe):
    results = []
    columns_to_diff = dataframe.filter(regex='PDV_|PDP_|TDV_|TDP_|THV_').columns.tolist() + ['MD', 'VFI', 'AGIS', 'CIGTS', 'Exam Age', 'days']
    
    grouped = dataframe.groupby('eye_episode')
    
    for idx, temp in grouped:
        half = len(temp) // 2
        diff = round(temp[columns_to_diff].iloc[half:].mean() - temp[columns_to_diff].iloc[:half].mean(), 3)
        
        last_temp = temp.iloc[-1].copy()      
        last_temp.update(diff)

        last_temp['Exam Age'] = temp['Exam Age'].mean()
        results.append(last_temp)
    
    mean_diff_df = pd.DataFrame(results).reset_index(drop=True)
    
    return mean_diff_df


