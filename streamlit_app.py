import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import plotly.io as pio

pio.templates.default = "seaborn"
# Define the cost function and gradient
def cost_function(x):
    return 2 * x**2 - 4 * x

def gradient(x):
    return (4 * x) - 4

# Define the gradient descent function
def gradient_descent(iterations, learning_rate, b_start):
    x_path = np.empty(iterations,)
    x_path[0] = b_start
    for i in range(1, iterations):
        derivative = gradient(x_path[i-1])
        x_path[i] = x_path[i-1] - (derivative * learning_rate)
    
    return x_path

# Streamlit user inputs for gradient descent parameters
st.title("Gradient Descent Interactive Visualization")
iterations = st.slider('Number of Iterations', min_value=1, max_value=60, value=30)
learning_rate = st.slider('Learning Rate', min_value=0.001, max_value=0.7, value=0.1, step=0.01)
b_start = st.selectbox('Starting Point (b_start)', options=[4, 5, 3.5, 3, 2.5])

# Generate data for the cost function
x_poly = np.linspace(-3, 5, 181)
y_poly = cost_function(x_poly)

# Run gradient descent and get the path
x_path = gradient_descent(iterations, learning_rate, b_start)
y_path = cost_function(x_path)

# Plot the cost function and gradient descent steps
title = f"Iterations (Epochs) = {iterations}, Learning Rate = {learning_rate}, b_start = {b_start}"

fig = px.line(x=x_poly, y=y_poly, title=title, template='seaborn', labels={"x":"parameter b","y":"Loss function"})
fig.add_trace(go.Scatter(x=x_path, y=y_path, mode='lines+markers', marker=dict(color='#A80000'), name='GD Path'))
fig.update_layout(
    autosize=False,
    width=1200,
    height=500)
# Display the plot in Streamlit
st.plotly_chart(fig, use_container_width=True)

