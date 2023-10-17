import pandas as pd
import numpy as np
from models.utils import *
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.figure_factory as ff

import warnings
warnings.filterwarnings('ignore')


def calculate_R2(model, type, input=None, complete_r=None):
    portfolio_ret = pd.read_pickle('data/portfolio_ret.pkl')
    oos_ret = portfolio_ret.loc[(portfolio_ret['DATE'] >= OOS_start) & (portfolio_ret['DATE'] <= OOS_end)]

    if not isinstance(input, np.ndarray):
        # print('type: ', type)
        if isinstance(model, str):
            output_path = f'results/{type}/{model}_{type}.csv'
        else:
            output_path = f'results/{type}/{model.name}_{type}.csv'
        # print('path : ', output_path)
        model_output = pd.read_csv(output_path)
    else:
        model_output = input
        model_output = pd.DataFrame(model_output, columns=CHARAS_LIST)
        model_output['DATE'] = oos_ret['DATE'].to_list()
    names = model_output.columns.values
    names[0] = 'DATE'
    model_output.columns = names

    for col in model_output.columns: # hard code for format error
        if col!='DATE':
            model_output[col] = model_output[col].apply(lambda x: float(str(x).replace('[', '').replace(']', '')))

    residual_square = ((oos_ret.set_index('DATE') - model_output.set_index('DATE'))**2).dropna()
    residual_square = (1 - (residual_square == np.inf) * 1.0) * residual_square # drop Inf outliers
    
    total_square = oos_ret.set_index('DATE')**2
    total_square = (1 - (total_square == np.inf) * 1.0) * total_square # drop Inf outliers
    
    model_output_R2 = 1 - np.sum(residual_square.values)/np.sum(total_square.values)
    
    if not isinstance(input, np.ndarray):
        return model_output_R2
    
    else:
        no_omit_output = complete_r
        no_omit_output = pd.DataFrame(no_omit_output, columns=CHARAS_LIST)
        no_omit_output['DATE'] = oos_ret['DATE'].to_list()
        
        no_omit_residual_square = ((oos_ret.set_index('DATE') - no_omit_output.set_index('DATE'))**2).dropna()
        no_omit_residual_square = (1 - (no_omit_residual_square == np.inf) * 1.0) * no_omit_residual_square # drop Inf outliers
        
        no_omit_model_output_R2 = 1 - np.sum(no_omit_residual_square.values)/np.sum(total_square.values)
        
        return no_omit_model_output_R2 - model_output_R2 # the difference of R^2, i.e. the importance of characteristics



def alpha_plot(model, type, save_dir='imgs'):
    if 'alpha' not in os.listdir(save_dir):
        os.mkdir(f'{save_dir}/alpha')
    
    portfolio_ret = pd.read_pickle('data/portfolio_ret.pkl')
    oos_result = portfolio_ret.loc[(portfolio_ret['DATE'] >= OOS_start) & (portfolio_ret['DATE'] <= OOS_end)].set_index('DATE')
    
    output_path = f'results/{type}/{model.name}_{type}.csv'
    inference_result = pd.read_csv(output_path)
    names = inference_result.columns.values
    names[0] = 'DATE'
    inference_result.columns = names

    inference_result = inference_result.set_index('DATE')
    
    pricing_error_analysis = []
    for col in CHARAS_LIST:
        raw_return = oos_result[col].mean()
        error = oos_result[col] - inference_result[col]
        alpha = error.mean()
        t_stat = abs(error.mean()/error.std()) * np.sqrt(oos_result.shape[0])
        pricing_error_analysis.append([raw_return, alpha, t_stat])

    pricing_error_analysis = pd.DataFrame(pricing_error_analysis, columns = ['raw ret', 'alpha', 't_stat'], index=CHARAS_LIST)
    
    lower_point = min(np.min(pricing_error_analysis['raw ret']), np.min(pricing_error_analysis['alpha'])) * 1.15
    upper_point = max(np.max(pricing_error_analysis['raw ret']), np.max(pricing_error_analysis['alpha'])) * 1.15

    significant_mask = pricing_error_analysis['t_stat'] > 3

    plt.scatter(pricing_error_analysis.loc[significant_mask]['raw ret'], pricing_error_analysis.loc[significant_mask]['alpha'], marker='^', color='r', alpha=0.6, label=f'#Alphas(|t|>3.0)={np.sum(significant_mask*1.0)}')
    plt.scatter(pricing_error_analysis.loc[~significant_mask]['raw ret'], pricing_error_analysis.loc[~significant_mask]['alpha'], marker='o', color='b', alpha=0.6, label=f'#Alphas(|t|<3.0)={94-np.sum(significant_mask*1.0)}')
    plt.plot(np.linspace(lower_point, upper_point, 10), np.linspace(lower_point, upper_point, 10), color='black')

    plt.ylabel('Alpha (%)')
    plt.xlabel('Raw Return (%)')
    plt.legend()

    plt.title(model.name)
    plt.savefig(f'{save_dir}/alpha/{model.name}_inference_alpha_plot.png')
    plt.close()
    

def plot_R2_bar(R_df, type):
    
    R_df['Model'] = R_df[0].apply(lambda x: x.split('_')[0])

    labels = ['K=1', 'K=2', 'K=3', 'K=4', 'K=5', 'K=6']
    FF = (R_df.loc[R_df['Model']=='FF'][1]*100).to_list()
    PCA = (R_df.loc[R_df['Model']=='PCA'][1]*100).to_list()
    IPCA = (R_df.loc[R_df['Model']=='IPCA'][1]*100).to_list()
    CA0 = (R_df.loc[R_df['Model']=='CA0'][1]*100).to_list()
    CA1 = (R_df.loc[R_df['Model']=='CA1'][1]*100).to_list()
    CA2 = (R_df.loc[R_df['Model']=='CA2'][1]*100).to_list()
    CA3 = (R_df.loc[R_df['Model']=='CA3'][1]*100).to_list()
    CVAE0 = (R_df.loc[R_df['Model']=='CVAE0'][1]*100).to_list()
    CVAE1 = (R_df.loc[R_df['Model']=='CVAE1'][1]*100).to_list()
    CVAE2 = (R_df.loc[R_df['Model']=='CVAE2'][1]*100).to_list()
    CVAE3 = (R_df.loc[R_df['Model']=='CVAE3'][1]*100).to_list()


    x = np.arange(len(labels))  # 标签位置
    width = 0.11

    fig, ax = plt.subplots(figsize=(15, 5))
    ax.bar(x - width*3 , FF, width, label='FF', color=plt.get_cmap('OrRd')(np.linspace(0, 1, 8)[1]))
    ax.bar(x - width*2 , PCA, width, label='PCA', color=plt.get_cmap('OrRd')(np.linspace(0, 1, 8)[2]))
    ax.bar(x - width , IPCA, width, label='IPCA', color=plt.get_cmap('OrRd')(np.linspace(0, 1, 8)[3]))
    ax.bar(x + 0.00, CA0, width, label='CA0', color=plt.get_cmap('OrRd')(np.linspace(0, 1, 8)[4]))
    ax.bar(x + width , CA1, width, label='CA1', color=plt.get_cmap('OrRd')(np.linspace(0, 1, 8)[5]))
    ax.bar(x + width*2 , CA2, width, label='CA2', color=plt.get_cmap('OrRd')(np.linspace(0, 1, 8)[6]))
    ax.bar(x + width*3 , CA3, width, label='CA3', color=plt.get_cmap('OrRd')(np.linspace(0, 1, 8)[7]))
    ax.bar(x + width*4 , CVAE0, width, label='CVAE0', color=plt.get_cmap('OrRd')(np.linspace(0, 1, 8)[7]))
    ax.bar(x + width*5 , CVAE1, width, label='CVAE1', color=plt.get_cmap('OrRd')(np.linspace(0, 1, 8)[7]))
    ax.bar(x + width*6 , CVAE2, width, label='CVAE2', color=plt.get_cmap('OrRd')(np.linspace(0, 1, 8)[7]))
    ax.bar(x + width*7 , CVAE3, width, label='CVAE3', color=plt.get_cmap('OrRd')(np.linspace(0, 1, 8)[7]))


    ax.set_ylabel(f'Portfolio {type} R^2 (%)')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    fig.tight_layout()

    plt.savefig(f'imgs/{type}_R2.png')
    plt.close()



def plot_R2_table(R_df, type):
    plt.figure(dpi=200)
    
    for col in R_df.columns:
        R_df[col] = R_df[col].apply(lambda x: round_number(x))

    R_df = R_df.reset_index()
    R_df.columns = ['Model', 'K=1', 'K=2', 'K=3', 'K=4', 'K=5', 'K=6']


    fig_total =  ff.create_table(R_df,
                        colorscale=[[0, 'white'],
                                    [0.01, 'lightgrey'],
                                    [1.0, 'white']],
                        font_colors=['#000000', '#000000',
                                    '#000000'])
    fig_total.update_layout(
        autosize=False,
        width=500,
        height=200,
    )
    fig_total.write_image(f"imgs/R2_{type}_table.png", scale=4)
    


def round_number(num):
    num = str(round(num*100, 2))
    while len(num.split('.')[1]) < 2:
        num = num + '0'
    return num


    
if __name__=="__main__":
    CAs = ["CA0_1", "CA0_2", "CA0_3", "CA0_4", "CA0_5", "CA0_6", "CA1_1", "CA1_2", "CA1_3", "CA1_4", "CA1_5", "CA1_6", "CA2_1", "CA2_2", "CA2_3", "CA2_4", "CA2_5", "CA2_6", "CA3_1", "CA3_2", "CA3_3", "CA3_4", "CA3_5", "CA3_6"]
    FFs = ["FF_1", "FF_2", "FF_3", "FF_4", "FF_5", "FF_6"]
    PCAs = ["PCA_1", "PCA_2", "PCA_3", "PCA_4", "PCA_5", "PCA_6"]
    IPCAs = ["IPCA_1", "IPCA_2", "IPCA_3", "IPCA_4", "IPCA_5", "IPCA_6"]
    CVAEs = [f'CVAE{prefix}_{suffix+1}' for prefix, suffix in zip(range(4),range(6))]
    models = FFs + PCAs + IPCAs + CAs + CVAEs
    
    ## Plot R^2 bars
    total_R2 = []
    for m in models:
        total_R2.append(calculate_R2(m, 'inference'))
    R_total = pd.DataFrame([models, total_R2]).T

    predict_R2 = []
    for m in models:
        predict_R2.append(calculate_R2(m, 'predict'))
    R_pred = pd.DataFrame([models, predict_R2]).T
    
    plot_R2_bar(R_total, 'total')
    plot_R2_bar(R_pred, 'pred')
    
    ## Save R^2 tables
    R_total_df = pd.DataFrame(np.array(total_R2).reshape(-1, 6), columns = ['K=1', 'K=2', 'K=3', 'K=4', 'K=5', 'K=6'], index=['FF', 'PCA', 'IPCA', 'CA0', 'CA1', 'CA2', 'CA3'])
    R_pred_df = pd.DataFrame(np.array(predict_R2).reshape(-1, 6), columns = ['K=1', 'K=2', 'K=3', 'K=4', 'K=5', 'K=6'], index=['FF', 'PCA', 'IPCA', 'CA0', 'CA1', 'CA2', 'CA3'])
    
    plot_R2_table(R_total_df, 'total')
    plot_R2_table(R_pred_df, 'pred')  
    
    
    ## Plot characteristics importance heatmap
    # models = ["IPCA", "CA0_5", "CA1_5", "CA2_5", "CA3_5"]
    # #TODO: paste results from R_squares/
    # R2_omit = []
    # R_minus = pd.DataFrame(np.array(R2_omit).reshape(-1, 94)*100, index=models, columns=CHARAS_LIST).T
    # char_ranks = R_minus.T.sum().argsort().argsort().index.to_list()
    # char_ranks.reverse()
    
    # plt.figure(figsize=(8, 15), dpi=200)
    # sns.heatmap(R_minus.T[char_ranks].T, cmap='Blues', linewidths=0.6)
    # plt.savefig('imgs/omit_char_R2_bias.png', bbox_inches='tight')
    # plt.close()
    