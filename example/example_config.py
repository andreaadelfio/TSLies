'''
Configuration file to identify features and targets between the dataframe columns.
Defines column mappings, units, and visualization settings for time series analysis.
'''

h_names = ['hist_top', 'hist_Xpos', 'hist_Xneg', 'hist_Ypos', 'hist_Yneg']
# h_names = ['histNorm_top', 'histNorm_Xpos', 'histNorm_Xneg', 'histNorm_Ypos', 'histNorm_Yneg']

y_cols_raw = ['top', 'Xpos', 'Xneg', 'Ypos', 'Yneg']
y_cols_raw = ['top_low', 'top_middle', 'top_high', 'Xpos_low', 'Xpos_middle', 'Xpos_high', 'Xneg_low', 'Xneg_middle', 'Xneg_high', 'Ypos_low', 'Ypos_middle', 'Ypos_high', 'Yneg_low', 'Yneg_middle', 'Yneg_high']
# y_cols_raw = ['top_low']
# y_cols_raw = ['top_middle', 'top_high', 'Xpos_middle', 'Xpos_high', 'Xneg_middle', 'Xneg_high', 'Ypos_middle', 'Ypos_high', 'Yneg_middle', 'Yneg_high']
# y_cols_raw = ['Xpos_middle']
# y_cols_raw = [f'{col}_middle' for col in y_cols_raw]
y_smooth_cols = [f'{col}_smooth' for col in y_cols_raw]
# y_cols_raw = ['histNorm_top', 'histNorm_Xpos', 'histNorm_Xneg', 'histNorm_Ypos', 'histNorm_Yneg']
# y_smooth_cols = ['histNorm_top_smooth', 'histNorm_Xpos_smooth', 'histNorm_Xneg_smooth', 'histNorm_Ypos_smooth', 'histNorm_Yneg_smooth']
y_cols = y_cols_raw
y_pred_cols = [col + '_pred' for col in y_cols_raw]

# thresholds = {'top': 7, 'Xpos': 10, 'Ypos': 7, 'Xneg': 7, 'Yneg': 7}
# thresholds = {'top_low': 5, 'top_middle': 7, 'top_high': 7, 'Xpos_low': 5, 'Xpos_middle': 7, 'Xpos_high': 7, 'Xneg_low': 5, 'Xneg_middle': 7, 'Xneg_high': 7, 'Ypos_low': 5, 'Ypos_middle': 7, 'Ypos_high': 7, 'Yneg_low': 5, 'Yneg_middle': 7, 'Yneg_high': 7}
thresholds = {'top_low': 5, 'top_middle': 5, 'top_high': 5, 'Xpos_low': 5, 'Xpos_middle': 5, 'Xpos_high': 5, 'Xneg_low': 5, 'Xneg_middle': 5, 'Xneg_high': 5, 'Ypos_low': 5, 'Ypos_middle': 5, 'Ypos_high': 5, 'Yneg_low': 5, 'Yneg_middle': 5, 'Yneg_high': 5}


x_cols = ['SC_POSITION_0', 'SC_POSITION_1', 'SC_POSITION_2', 'LAT_GEO', 'LON_GEO',
            'RAD_GEO', 'RA_ZENITH', 'DEC_ZENITH', 'B_MCILWAIN', 'L_MCILWAIN', 
            'GEOMAG_LAT', 'LAMBDA', 'RA_SCZ', 'START', 'STOP', 'MET',
            'LAT_MODE', 'LAT_CONFIG', 'DATA_QUAL', 'LIVETIME', 'DEC_SCZ', 'RA_SCX',
            'DEC_SCX', 'RA_NPOLE', 'DEC_NPOLE', 'ROCK_ANGLE',
            'QSJ_1', 'QSJ_2', 'QSJ_3', 'QSJ_4', 'RA_SUN', 'DEC_SUN',
            'SC_VELOCITY_0', 'SC_VELOCITY_1', 'SC_VELOCITY_2',
            'GOES_XRSA_HARD', 'GOES_XRSB_SOFT', 'GOES_XRSA_HARD_EARTH_OCCULTED',
            'GOES_XRSB_SOFT_EARTH_OCCULTED', 'TIME_FROM_SAA', 'SAA_EXIT']
x_cols_excluded = ['LIVETIME', 'GOES_XRSA_HARD_EARTH_OCCULTED', 'GOES_XRSB_SOFT_EARTH_OCCULTED', 'STOP', 'MET']

#units for each y column and x column
units = {'SC_POSITION_0': 'm',
        'SC_POSITION_1': 'm',
        'SC_POSITION_2': 'm',
        'LAT_GEO': 'deg',
        'LON_GEO': 'deg',
        'RAD_GEO': 'deg',
        'RA_ZENITH': 'deg',
        'DEC_ZENITH': 'deg',
        'B_MCILWAIN': 'Gauss',
        'L_MCILWAIN': 'Earth_Radii', 
        'GEOMAG_LAT': 'deg',
        'LAMBDA': 'deg',
        'RA_SCZ': 'deg',
        'START': 's',
        'STOP': 's',
        'MET': 's',
        'LAT_MODE': 'NA',
        'LAT_CONFIG': 'NA',
        'DATA_QUAL': 'NA',
        'LIVETIME': 's',
        'DEC_SCZ': 'deg',
        'RA_SCX': 'deg',
        'DEC_SCX': 'deg',
        'RA_NPOLE': 'deg',
        'DEC_NPOLE': 'deg',
        'ROCK_ANGLE': 'deg',
        'QSJ_1': 'NA',
        'QSJ_2': 'NA',
        'QSJ_3': 'NA',
        'QSJ_4': 'NA',
        'RA_SUN': 'deg',
        'DEC_SUN': 'deg',
        'SC_VELOCITY_0': 'm/s',
        'SC_VELOCITY_1': 'm/s',
        'SC_VELOCITY_2': 'm/s',
        'GOES_XRSA_HARD': r'W/m$^2$/s',
        'GOES_XRSB_SOFT': r'W/m$^2$/s',
        'GOES_XRSA_HARD_EARTH_OCCULTED': r'W/m$^2$/s',
        'GOES_XRSB_SOFT_EARTH_OCCULTED': r'W/m$^2$/s',
        'TIME_FROM_SAA': 's',
        'SAA_EXIT': 'NA',
        }

latex_y_cols = {}
for col in y_cols:
    face = col.split('_')[0]
    energy = col.split('_')[1][0].upper()
    if 'pos' in col:
        coord = col.split('pos')[0]
        latex_y_cols[col] = f'{coord}^+' + r'_{%s}'%energy
    elif 'neg' in col:
        coord = col.split('neg')[0]
        latex_y_cols[col] = f'{coord}^-' + r'_{%s}'%energy
    elif 'top' in col:
        latex_y_cols[col] = 'T' + r'_{%s}'%energy
    units[col] = r'count/m^2/s'

if __name__ == '__main__':
    vars_copy = vars().copy()
    filtered_vars = [var for var in vars_copy if '__' not in var]
    max_len = max(map(len, filtered_vars))

    for var in filtered_vars:
        print(f"{var:<{max_len}} {vars_copy[var]}")
