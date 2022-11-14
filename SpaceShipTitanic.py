#SpaceShipTitanic
import dash
from dash import Dash
from dash import dcc
from dash import dash_table
from dash import html
import numpy as np

import plotly.express as px 
import pandas as pd 


import joblib

from dash.dependencies import Input, Output

loaded_model = joblib.load("model.joblib")

df_train = pd.read_csv('train.csv')
df_head = df_train.head(10)
df_eda = pd.read_csv('train.csv')[['HomePlanet','Transported']]
df_eda = df_eda.replace([True,False],[1,0])
serie = df_eda.groupby(['HomePlanet']).sum()


totalEarth = df_eda['HomePlanet'].str.contains('Earth').sum()
totalTrueEarth = serie.at['Earth','Transported']
totalEuropa = df_eda['HomePlanet'].str.contains('Europa').sum()
totalTrueEuropa = serie.at['Europa','Transported']
totalMars = df_eda['HomePlanet'].str.contains('Mars').sum()
totalTrueMars = serie.at['Mars','Transported']
totalFalseEarth = totalEarth-totalTrueEarth
totalFalseEuropa = totalEuropa - totalTrueEuropa
totalFalseMars = totalMars - totalTrueMars

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = Dash(__name__, external_stylesheets=external_stylesheets)

df_eda = pd.DataFrame({
    "HomePlanet": ["Europa", "Earth", "Mars", "Europa", "Earth", "Mars"],
    "Amount": [totalTrueEuropa, totalTrueEarth, totalTrueMars,totalFalseEuropa, totalFalseEarth, totalFalseMars],
    "Transported": ["True", "True", "True", "False", "False", "False"]
})

fig = px.bar(df_eda, x="HomePlanet", y="Amount", color="Transported", barmode="group")


app.layout = html.Div([
		html.H1(children="SpaceShipTitanic", style={'textAlign': 'center'}),
		html.H3(children="Nicolás Penagos A00347293", style={'textAlign': 'center'}),

		dcc.Tabs([

			dcc.Tab(label='Data', children=[
				dash_table.DataTable(df_head.to_dict('records'), [{"name": i, "id": i} for i in df_head.columns])
      

        ]),dcc.Tab(label='EDA', children=[
            dcc.Graph(
        		id='example-graph',
        		figure=fig
    		)
        ]),dcc.Tab(label='Prediction', children=[
        	html.Img(src=r'assets/image.png', alt='image'),
        	html.Div( className = 'row',
        		children = [
        		
	        		html.Div(children = [
		           		'Room Service',
		           		dcc.Input(id='room-service', value=0, type='number')

	           		]),
		           	html.Div([
		           		'Food court',
		           		dcc.Input(id='food-court', value=0, type='number')
		           	]),
		           	html.Div([
		           		'Shopping Mall',
		           		dcc.Input(id='shopping-mall', value=0, type='number')
		           	]),
		           	html.Div([
		           		'Spa',
		           		dcc.Input(id='spa', value=0, type='number')
		           	])

        		]),

        	html.Div(id='output')
         
        ])



		])


])


@app.callback(
	Output(component_id='output', component_property='children'),
	Input(component_id= 'room-service', component_property='value'),
	Input(component_id='food-court', component_property='value'),
	Input(component_id='shopping-mall', component_property='value'),
	Input(component_id='spa', component_property='value'),


)
def predict(roomService, foodCourt, shoppingMall, spa):

	if((type(roomService) == int or type(roomService) == float)and(type(foodCourt) == int or type(foodCourt) == float)and(type(shoppingMall) == int or type(shoppingMall) == float)and(type(spa) == int or type(spa) == float)):
		data = {'RoomService': [roomService], 'FoodCourt': [foodCourt], 'ShoppingMall':[shoppingMall], 'Spa':[spa]}  
		data_df = pd.DataFrame(data)  
		y_prob = loaded_model.predict_proba(data_df)

		y_prob_c1 = y_prob[:, 1]
		th =0.6236
		y_pred = np.zeros(len(y_prob_c1))
		for i in range(len(y_prob_c1)):
		    if (y_prob_c1[i] >= th):
		        y_pred[i] = 1
		        
		if(y_pred[0]==1):
			return 'The person was transported: True'
		else:
			return 'The person was transported: False'

	else:
		return 'Please enter all the params and only numbers'



if __name__ == "__main__":
	app.run_server(debug=True)