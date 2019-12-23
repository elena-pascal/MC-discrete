# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.offline as offline       # a number of plotly goodies
import plotly.plotly as py
import plotly.graph_objs as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
file_discr = 'BSE_70tilt_ds.out'

BSE_discr = []
with open(file_discr) as f:
    lines = f.read().splitlines()
BSE_discr = [float(line) for line in lines]
trace_discr = go.Histogram(x=BSE_discr, 
                          name = 'discrete', 
                          histnorm='probability')

file_cont = 'BSE_70tilt_cont.out'

BSE_cont = []
with open(file_cont) as f:
    lines = f.read().splitlines()
BSE_cont = [float(line) for line in lines]
trace_cont = go.Histogram(x=BSE_cont, 
                         name = 'continuous', 
                         histnorm='probability')

data = [trace_cont, trace_discr]


app.layout = html.Div(children=[
    html.H1(children='Energy distribution of BSE '),

    html.Div(children='''
       Continuous slowing down approximation versus discrete inelastic scattering
    '''),

    dcc.Graph(
        id='BSE-graph',
        figure={
            'data': data,
            'layout': {
                'title': '30 keV incident beam at 70 tilt on Al', 
                'xaxis': {'title' : 'Energy (eV)'
                        }, 
                'yaxis': {'title' : 'Probability (%)'
                        }
            }
        }
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
