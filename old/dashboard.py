import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import os
from dash import Dash, html, dcc
from dash.dependencies import Input, Output
from PIL import Image
import pandas as pd
import ast
from GLOSA_RL.evaluate import load_dict_from_csv


# Load the  start image
#f = 'rlSAC_03-27-19-33'
img_path = os.path.join('dashboard_source', 'pre_eval', 'screenshots')
images = list()
for filename in os.listdir(img_path):
    if filename.endswith('.png'):
        images.append(int(filename[0:filename.find('.')]))
min_value = min(images)
max_value = max(images)

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)
def create_d_fig():
    tl_info = load_dict_from_csv(os.path.join('dashboard_source', 'pre_eval', 'tlinfo.csv'))
    eval_info = pd.DataFrame.from_dict(load_dict_from_csv(os.path.join('dashboard_source', 'pre_eval', 'eval_metrics.csv')))
    layout = go.Layout(
                       xaxis=dict(title='time(s)'),
                       yaxis=dict(title='distance(m)')
                       )
    d_fig = go.Figure(layout=layout)
    for tl in tl_info:
        tl_info[tl]['phases'] = ast.literal_eval(tl_info[tl]['phases'])
        xs = np.linspace(0, len(tl_info[tl]['phases']), len(tl_info[tl]['phases']))
        ys = float(tl_info[tl]['distance'])
        df = pd.DataFrame({'x': xs, 'y': ys, 'color': tl_info[tl]['phases']})

        d_fig.add_scattergl(x=xs, y=df.y.where(df.color == 'r'), line={'color': 'red', 'width': 3}, showlegend=False)
        d_fig.add_scattergl(x=xs, y=df.y.where(df.color == 'Y'), line={'color': 'yellow', 'width': 3}, showlegend=False)
        d_fig.add_scattergl(x=xs, y=df.y.where(df.color == 'G'), line={'color': 'green', 'width': 3}, showlegend=False)
        d_fig.add_scattergl(x=xs, y=df.y.where(df.color == 'g'), line={'color': 'green', 'width': 3}, showlegend=False)
    start_value = list(eval_info.distance.keys())[0]
    hover_dict = {k - start_value: v * 3.6 for k, v in eval_info.speed.items()}
    d_fig.add_scatter(x=[i - start_value for i in list(eval_info.distance.keys())], y=list(eval_info.distance.values), mode='lines', showlegend=False, line={'color': 'black', 'width': 3})
    return d_fig
d_fig = create_d_fig()
img = Image.open(f'{img_path}/{min_value}.png')
fig = px.imshow(img, binary_format="png", binary_compression_level=0)

app.layout = html.Div([
    dcc.Slider(0, max_value-min_value, 1,
               value=0,
               id='time-slider'
    ),
    html.Div([
    dcc.Graph(figure=fig, id='image_replay')
]),
    dcc.Graph(figure=d_fig, id='distance_replay')
])

@app.callback(
    Output('image_replay', 'figure'),
    Input('time-slider', 'value'))
def update_output(value):
    img = Image.open(f'{img_path}/{min_value+value}.png')
    fig = px.imshow(img, binary_format="png", binary_compression_level=0)
    return fig #'You have selected "{}"'.format(value)

@app.callback(
    Output('distance_replay', 'figure'),
    Input('time-slider', 'value'))
def update_output(value):
    d_fig = create_d_fig()
    d_fig.add_shape(
        type='line',
        x0=value,
        y0=0,
        x1=value,
        y1=2500,
        line=dict(
            color='red',
            width=2,
            dash='dash',
        )
    )
    return d_fig #'You have selected "{}"'.format(value)

if __name__ == '__main__':
    app.run_server(debug=True)