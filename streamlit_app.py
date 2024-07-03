import pandas as pd
from lightgbm import LGBMClassifier
import streamlit as st


# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='Dividends predictor'
)

# -----------------------------------------------------------------------------
# Declare some useful functions.


# -----------------------------------------------------------------------------
# Draw the actual page

# Set the title that appears at the top of the page.
st.text('''
Some explanation here...

Classification report for 5% target:

Accuracy: 0.71
              precision    recall  f1-score   support

       False       0.73      0.85      0.79      2119
        True       0.64      0.46      0.53      1229

    accuracy                           0.71      3348
   macro avg       0.68      0.65      0.66      3348
weighted avg       0.70      0.71      0.69      3348
''')

# Add some spacing
''
''

''
''
''


@st.cache_data
def get_model() -> LGBMClassifier:
    return pd.read_pickle('data/model_v1.pkl')


@st.cache_data
def get_industries() -> list:
    return pd.read_csv('data/industries.csv').to_dict(orient='list')['industry']


# decl: 06/07/2024
# ex: 06/20/2024
# $0.18
# Banks

# Integers
volume = st.number_input('Volume (on Declaration date -1)', format='%i', min_value=1, step=1, value=441195)
market_cap = st.number_input('Market Cap', format='%i', min_value=1, step=1, value=996169767)
# Floats
p_yield = st.number_input('Yield (for now as-is)', format='%0.4f', value=0.0313) # annualized
close_price_d1 = st.number_input('Close Price (on Declaration date -1)', format='%0.1f', value=21.48)
close_price_d30 = st.number_input('Close Price (on Declaration date -30)', format='%0.1f', value=22.51)
close_price_d90 = st.number_input('Close Price (on Declaration date -90)', format='%0.1f', value=22.31)
# Text selector
industry = st.selectbox(label='Industry', options=get_industries())
# Dates
declaration_date = st.date_input(label='Declaration Date')
ex_date = st.date_input(label='Ex Date')

st.header('Prediction results', divider='gray')


if st.button('Make prediction!'):
    model = get_model()

    print('volume: ', volume)
    print('market_cap: ', market_cap)
    print('p_yield: ', p_yield)
    print('close_price_d1: ', close_price_d1)
    close_price_change_d30 = (close_price_d1 / close_price_d30) - 1
    close_price_change_d90 = (close_price_d1 / close_price_d90) - 1
    print('close_price_d30: ', close_price_change_d30)
    print('close_price_d90: ', close_price_change_d90)
    print('industry: ', industry)
    diff_ex_declaration = (ex_date - declaration_date).days
    print('diff_ex_declaration: ', diff_ex_declaration)

    df = pd.DataFrame(
        data=[
            {
                'volume_declaration_prior_1': volume,
                'yield': p_yield,
                'close_price_change_d90': close_price_change_d30,
                'close_price_change_d30': close_price_change_d90,
                'industry': industry,
                'diff_ex_declaration': 15,
                'market_cap': market_cap
            }
        ]
    )

    df['industry'] = df['industry'].astype('category')

    classification = model.predict(df)[0]
    classification_score = model.predict(df, raw_score=True)[0]

    st.text(f'The result is: {classification}')
    st.text(f'With the score: {classification_score}')
