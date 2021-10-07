# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.figure_factory as ff
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# external_stylesheets = ['https://unpkg.com/nes.css@2.3.0/css/nes.min.css']

TP = 0
FN = 0
FP = 0
TN = 0

app = dash.Dash(__name__,
                # assets_external_path='https://unpkg.com/nes.css@2.3.0/css/nes.css'
                )
server = app.server

app.layout = html.Div(

    style={'background-image': 'url("assets/scorbunny.jpg")',
           'background-repeat': 'no-repeat',
           'background-size': '100% 100%'},

    children=[
        html.Link(
            rel='stylesheet',
            href='https://unpkg.com/nes.css@latest/css/nes.min.css'
        ),
    dcc.Interval('graph-update', interval = 2000, n_intervals = 0),
    html.H1(children='Pokémon Laboratory'),

    html.Div(children='''
        Is this Pokémon legendary?
    '''),
    html.Div([
        "defense: ",
        dcc.Input(id='defense-input', value='0', type='text'),

        "Health: ",
        dcc.Input(id='hp-input', value='0', type='text'),

        "special attack: ",
        dcc.Input(id='sp_attack-input', value='0', type='text'),

        "special defense: ",
        dcc.Input(id='sp_defense-input', value='0', type='text'),

        "experience growth: ",
        dcc.Input(id='experience_growth-input', value='0', type='text'),

        html.Button(id='submit-button', n_clicks=0, children='Submit')

    ]),
    html.Br(),
    html.Div(id='my-output'),

    html.Br(),
    html.Div([
        html.Button(id='Correct-button', n_clicks=0, children='Correct!'),
        html.Button(id='Wrong-button', n_clicks=0, children='Wrong!')
    ]),

    html.Br(),
    html.Div(dcc.Graph(id='Confusion-figure')),
])

@app.callback(
    Output('my-output', 'children'),
    Input('submit-button', 'n_clicks'),
    State('defense-input', 'value'),
    State('hp-input', 'value'),
    State('sp_attack-input', 'value'),
    State('sp_defense-input', 'value'),
    State('experience_growth-input', 'value')
)
def make_predictions(n_clicks, defense, hp, sp_attack, sp_defense, experience_growth):
    defense_norm = (int(defense) - 5) / (230 - 5)
    hp_norm = (int(hp) - 1) / (255 - 1)
    sp_attack_norm = (int(sp_attack) - 10) / (194 - 10)
    sp_defense_norm = (int(sp_defense) - 20) / (230 - 20)
    experience_growth_norm = (int(experience_growth) - 600000) / (1640000 - 600000)

    features_input = [defense_norm, hp_norm, sp_attack_norm, sp_defense_norm, experience_growth_norm]
    features_input = np.array(features_input)

    loaded_model = pickle.load(open("legendary_rfclf.sav", 'rb'))
    result = loaded_model.predict(features_input.reshape(1, -1))

    pokemon_type = "Legendary" if result[0] == 1 else "Non-legendary"

    return pokemon_type

@app.callback(
    Output('Confusion-figure', 'figure'),
    Input('Correct-button', 'n_clicks'),
    Input('Wrong-button', 'n_clicks'),
    Input('my-output', 'children'),
    Input('graph-update', 'n_intervals')
)
def correct_predictions(clicks_correct, clicks_wrong, prediction, n):
    global TP
    global TN
    global FP
    global FN

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0]
    if 'Correct-button' in changed_id:
        if prediction == "Legendary":
            TP += 1
        elif prediction == "Non-legendary":
            TN += 1
    elif 'Wrong-button' in changed_id:
        if prediction == "Legendary":
            FP += 1
        elif prediction == "Non-legendary":
            FN += 1

    z = [[FN, TN],
         [TP, FP]]
    z_text = [[str(y) for y in x] for x in z]

    x = ['Positive', 'False']
    y = ['False', 'Positive']

    fig = ff.create_annotated_heatmap(z, x=x, y=y, annotation_text=z_text, colorscale='Viridis')

    fig.update_layout(title_text='<i><b>Confusion matrix</b></i>',
                      # xaxis={'side': 'top'},
                      # yaxis = dict(title='x')
                      )

    # add custom xaxis title
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=0.5,
                            y=-0.15,
                            showarrow=False,
                            text="Real value",
                            xref="paper",
                            yref="paper"))

    # add custom yaxis title
    fig.add_annotation(dict(font=dict(color="black", size=14),
                            x=-0.35,
                            y=0.5,
                            showarrow=False,
                            text="Predicted value",
                            textangle=-90,
                            xref="paper",
                            yref="paper"))

    # adjust margins to make room for yaxis title
    fig.update_layout(margin=dict(t=50, l=200), paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

    # add colorbar
    fig['data'][0]['showscale'] = True


    return fig


if __name__ == '__main__':
    app.run_server(debug=True)