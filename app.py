
import numpy as np
import pandas as pd
import streamlit as st
import pickle 

with open('final_model.pkl','rb') as file:
    model=pickle.load(file)
    
with open ('transformer.pkl','rb') as file:
    transformer=pickle.load(file)
    
    
def prediction(input_list):
    input_list=np.array(input_list,dtype=object)
    
    pred=model.predict_proba([input_list])[:,1][0]
    
    if pred>0.5:
        return f'This Booking is more likely to get cancelled, with chances {round(pred,2)}'
    
    else:
        return f'This Booking is less likely to get cancelled, with chances {round(pred,2)}'
    
    
def main():
    st.title('INN HOTEL GROUP')
    
    lt= st.text_input('Enter the Lead time in Days')
    price= st.text_input('Enter the price of the room')
    weekn= st.text_input('Enter the number of week nights in stay')
    wkndn= st.text_input('Enter the number of weekend nights in stay')

    mkt= 1 if st.selectbox('How the booking was made',['Online','Offline']) == 'Online' else 0
    adult= st.selectbox('How many adults',[1,2,3,4])
    arr_m= st.slider('What is the month of arrival?',min_value=1,max_value=12,step=1)
    
    weekd_lambda = lambda x: {'Mon':0,'Tue':1,'Wed':2,'Thur':3,'Fri':4,'Sat':5,'Sun':6}[x]
    arr_w= weekd_lambda(st.selectbox('What is the weekday of arrival',['Mon','Tue','Wed','Thur','Fri','Sat','Sun']))
    dep_w= weekd_lambda(st.selectbox('What is the weekday of departure',['Mon','Tue','Wed','Thur','Fri','Sat','Sun']))
    
    park= 1 if st.selectbox('Does customer need parking?',['Yes','No']) == 'Yes' else 0
    spcl= st.selectbox('Special request required?',[0,1,2,3,4,5])

    lt_val = safe_float(lt)
    price_val = safe_float(price)
    weekn_val = safe_int(weekn)
    wkndn_val = safe_int(wkndn)

    if st.button('Predict'):
        if None in [lt_val, price_val, weekn_val, wkndn_val]:
            st.warning("Please enter valid numeric values for lead time, price, and nights.")
            return

        totan = weekn_val + wkndn_val
        
        try:
            lt_t, price_t = transformer.transform([[lt_val, price_val]])[0]
            inp_list = [lt_t, spcl, price_t, adult, wkndn_val, park, weekn_val, mkt, arr_m, arr_w, totan, dep_w]

            response = prediction(inp_list)
            st.success(response)
        except Exception as e:
            st.error(f"Prediction failed: {e}")

        
if __name__=='__main__':
    main()
