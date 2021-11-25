import pandas as pd
import pickle
import sklearn
import xgboost
import scipy
import streamlit as st
import numpy as np
import base64
import matplotlib

# loaded_model= pickle.load(open('latest_final_model.pkl','rb'))
loaded_model= xgboost.Booster()
loaded_model.load_model('latest_final_model.json')
# config= pickle.load(open("/content/drive/MyDrive/PrudentialData/Booster_final_model_config.pkl",'rb'))
# xgboost.Booster.feature_names=config
# loaded_model.load_config(config)
train = pd.read_csv('train.csv')
test =pd.read_csv("test.csv")

def get_table_download_link_csv(df):
    #csv = df.to_csv(index=False)
    csv = df.to_csv(index=False).encode()
    #b64 = base64.b64encode(csv.encode()).decode() 
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="test.csv" target="_blank">Download csv file</a>'
    return href

def get_xgb_imp(xgb, feat_names):
    from numpy import array
    imp_vals = xgb.get_fscore()
    imp_vals= dict(sorted(imp_vals.items(), key= lambda x: x[1], reverse= True))
    # imp_dict = {feat_names[i]:float(imp_vals.get('f'+str(i),0.)) for i in range(len(feat_names))}
    total = array(imp_vals.values()).sum()

    def div_d(my_dict):
        sum_p = sum(my_dict.values())
        for i in my_dict:
            my_dict[i] = float(my_dict[i]*100/sum_p)

        return my_dict
    
    return div_d(imp_vals)


def eval_wrapper(yhat, y):  
    y = np.array(y)
    y = y.astype(int)
    yhat = np.array(yhat)
    yhat = np.clip(np.round(yhat), np.min(y), np.max(y)).astype(int)   
    return sklearn.metrics.cohen_kappa_score(yhat, y,weights='quadratic')

def apply_offset(data, bin_offset, sv, scorer=eval_wrapper):
    # data has the format of pred=0, offset_pred=1, labels=2 in the first dim
    data[1, data[0].astype(int)==sv] = data[0, data[0].astype(int)==sv] + bin_offset
    score = scorer(data[1], data[2])
    return score


def final_fun_1(X):
    
    '''This function takes details about a potential customers as input and returns a prediction of the risk level of Customer.
       The details include: product info, family info, employment info, general health measurements , 
       medical history info, and medical keyword(yes/No).'''
    global loaded_model, train
    
    columns_to_drop = ['Id', 'Response']
    xgb_num_rounds = 800
    num_classes = 8
    eta_list = [0.05] * 200 
    eta_list = eta_list + [0.02] * 500
    eta_list = eta_list + [0.01] * 100
    test= X.copy()
    # create any new variables    
    train['Product_Info_2_char'] = train.Product_Info_2.str[0]
    train['Product_Info_2_num'] = train.Product_Info_2.str[1]
    test['Product_Info_2_char'] = test.Product_Info_2.str[0]
    test['Product_Info_2_num'] = test.Product_Info_2.str[1]

    # factorize categorical variables
    train['Product_Info_2'] = pd.factorize(train['Product_Info_2'])[0]
    train['Product_Info_2_char'] = pd.factorize(train['Product_Info_2_char'])[0]
    train['Product_Info_2_num'] = pd.factorize(train['Product_Info_2_num'])[0]

    train['BMI_Age'] = train['BMI'] * train['Ins_Age']

    med_keyword_columns = train.columns[train.columns.str.startswith('Medical_Keyword_')]
    train['Med_Keywords_Count'] = train[med_keyword_columns].sum(axis=1)
    
    
    test['Product_Info_2'] = pd.factorize(test['Product_Info_2'])[0]
    test['Product_Info_2_char'] = pd.factorize(test['Product_Info_2_char'])[0]
    test['Product_Info_2_num'] = pd.factorize(test['Product_Info_2_num'])[0]

    test['BMI_Age'] = test['BMI'] * test['Ins_Age']

    med_keyword_columns = test.columns[test.columns.str.startswith('Medical_Keyword_')]
    test['Med_Keywords_Count'] = test[med_keyword_columns].sum(axis=1)

    print('Eliminate missing values')
    # Use -1 for any others
    train.fillna(-1, inplace=True)
    test.fillna(-1, inplace=True)

    # fix the dtype on the label column
    train['Response'] = train['Response'].astype(int)
    # test['Response'] = test['Response'].astype(int)
    xgtrain = xgboost.DMatrix(train.drop(columns_to_drop, axis=1), label=train['Response'].values)
    xgtest = xgboost.DMatrix(test.drop(['Id'], axis=1), label=None)
    
    train_preds = loaded_model.predict(xgtrain, ntree_limit=loaded_model.num_boosted_rounds())
    test_preds = loaded_model.predict(xgtest, ntree_limit=loaded_model.num_boosted_rounds())
    train_preds = np.clip(train_preds, -0.99, 8.99)
    test_preds = np.clip(test_preds, -0.99, 8.99)

    # train offsets 
    offsets = np.array([0.1, -1, -2, -1, -0.8, 0.02, 0.8, 1])
    data = np.vstack((train_preds, train_preds, train['Response'].values))
    for j in range(num_classes):
        data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j] 
    for j in range(num_classes):
        train_offset = lambda x: -apply_offset(data, x, j)
        offsets[j] = scipy.optimize.fmin_powell(train_offset, offsets[j])  

    # apply offsets to test
    data = np.vstack((test_preds, test_preds, np.zeros(test.shape[0],)))
    for j in range(num_classes):
        data[1, data[0].astype(int)==j] = data[0, data[0].astype(int)==j] + offsets[j] 

    final_test_preds = np.round(np.clip(data[1], 1, 8)).astype(int)
    #     all_predictions= pd.DataFrame(final_test_preds, index= test['Id'], columns=['Response'])
    return final_test_preds, get_xgb_imp(loaded_model,train.drop(columns_to_drop, axis=1).columns)



def main():
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Life Insurance Risk Prediction </h2>
    </div>
    <br>
    """
    
    st.markdown(html_temp,unsafe_allow_html=True)
    st.write('Created by: **Sumit Gulati**')
    st.write("Fill in the details in below csv file to predict the risk.")
    st.write("")
    
    download=st.button('Download template')
    if download:
        'Download Started!'
        st.markdown(get_table_download_link_csv(test), unsafe_allow_html=True)


    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        X = pd.read_csv(uploaded_file)
        st.write(X)
    
    
    res, prob = "", ""
    if st.button("Predict"):
        y_pred, importances =final_fun_1(X)
        st.write('On a scale of 1-8 customer insurance risk level is ',y_pred)
        st.write("The top 10 important Features are as follows:\n")
        fig, ax = matplotlib.pyplot.subplots()
        ax = matplotlib.pyplot.bar(list(importances.keys())[:15], list(importances.values())[:15])
        matplotlib.pyplot.xticks(rotation= 90)
        st.pyplot(fig)
if __name__=='__main__':
    main()


