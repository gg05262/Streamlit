import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px


st.title('Streamlit for sin and cos fuctioni visualiztion')

x_start = st.slider('x 시작값' ,  0.0, 10.0, 0.0)
x_end = st.slider('x 시작값' ,  10.0, 20.0, 10.0)


x = np.linspace(x_start, x_end)

y_sin = np.sin(x)
y_cos = np.cos(x)


fig , ax = plt.subplots()

ax.plot(x, y_sin)  
ax.plot(x, y_cos)
ax.legend(['sin', 'cos'])
ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')

ax.set_title('sin and cos fuction')

st.pyplot(fig)

@st.cache
def expensive_computataion(x):
    return np.sin(x) + np.cos(x)



result = expensive_computataion(x)

data_canada = px.data.gapminder().query("country == 'Canada'")
data_canada

fig = px.bar(data_canada, x='year', y='pop')
fig.show()

fig1 = px.bar(data_canada, x='year', y='lifeExp')
fig1.show()

fig2 = px.bar(data_canada, x='year', y='gdpPercap')
fig2.show()

fig2_1 = px.line(data_canada, x = 'year', y = 'lifeExp' ,
              title = 'Life expectacy in Canada')
fig2_1.show()

fig2_2 = px.line(data_canada, x = 'year', y = 'pop' ,
              title = 'Life expectacy in Canada')
fig2_2.show()

fig2_3 = px.line(data_canada, x = 'year', y = 'gdpPercap' ,
              title = 'Life expectacy in Canada')

fig2_3.show()

df_Oceania = px.data.gapminder().query("continent == 'Oceania'")
df_Oceania

fig3 = px.bar(df, x = 'year' , y = 'pop' , color = 'country' ,
             labels = {'pop' : 'population of Canada'} , hover_data = ['lifeExp','gdpPercap']
             , barmode = 'group')

fig3.show()


fig4 = px.bar(df, x = 'year' , y = 'pop' , color = 'country' ,
             labels = {'pop' : 'population of Canada'} , hover_data = ['lifeExp','gdpPercap']
             ,pattern_shape_sequence=["."
             ,'+'])

fig4.show()

df = px.data.gapminder().query("continent == 'Oceania'")
df

fig5 = px.bar(df, x='year' , y='pop')
st.plotly_chart(fig5)


df1 = px.data.gapminder().query("year == 2007")
fig6 = px.treemap(df1, path=[px.Constant('World'), 'continent','country'], values = 'pop' , color = 'lifeExp')
st.plotly_chart(fig6)


df2 = px.data.gapminder().query("year == 2007").query("continent == 'Europe'")

df2

labels = ['A','B', 'C' ,'D']
values = [300,200,100,500]

fig7 = go.Figure(data = [go.Pie(labels = labels, values = values, hole=.3)]) # hole 도넛차트 구현
fig7.show()

fig8 = go.Figure(data = [go.Pie(labels = labels, values = values, pull = [0 , 0, 0.2,0]), hole=.3])
fig8.show()

fig9 = px.imshow([[1,23,49],[123,5,4],[45,6,3]]
                )


fig9.show()

df3 = px.data.tips()
df3

fig10 = px.box(df, y = 'total_bill' , x = 'time' , points = 'all' , color = 'smoker')
fig10.show()

fig11 = px.box(df, y = 'total_bill' , x = 'day' , points = 'all' , color = 'smoker')
fig11.show()

fig12 = px.scatter(df.query("year == 2007"), x = 'gdpPercap' , y = 'lifeExp', size = 'pop', color = 'continent')
fig12.show()
