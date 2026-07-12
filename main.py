import time
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from sympy import Derivative, lambdify, symbols

st.set_page_config(page_title="Gradient Descent Laboratory")
st.title("민수의 Gradient Descent 연구소")

x = symbols('x')
input_func = x ** 2
loss_func = lambdify(x, input_func, 'numpy')
gradient = Derivative(input_func, x).doit()

st.sidebar.header("Loss Function")
st.sidebar.latex(r'''L(x) = x^2''')

st.sidebar.header("Parameters")
theta_init = st.sidebar.slider("Initial Theta", min_value=-100, max_value=100, value=80, step=1)
learning_rate = st.sidebar.slider("Learning Rate", min_value=0.01, max_value=2.0, value=0.3, step=0.01)

if st.sidebar.button("Execute"):
    theta = float(theta_init)
    x_history = [theta]
    y_history = [float(input_func.subs(x, theta))]

    while abs(input_func.subs(x, theta)) > 1:
        d = float(gradient.subs(x, theta))
        theta = theta - learning_rate * d
        x_history.append(theta)
        y_history.append(float(input_func.subs(x, theta)))

    chart_placeholder = st.empty()
    status_placeholder = st.empty()

    min_x, max_x = -100, 100
    x_vals = np.linspace(min_x, max_x, 200)
    y_vals = loss_func(x_vals)

    for step in range(len(x_history)):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(x_vals, y_vals, label=r'$L(x) = x^2$', color='lightgray', linewidth=2)
        ax.plot(x_history[:step + 1], y_history[:step + 1], 'ro--', label='Path', linewidth=2, markersize=6)
        ax.set_xlabel('x')
        ax.set_ylabel('Loss')
        ax.grid(True)
        ax.legend()

        chart_placeholder.pyplot(fig)
        plt.close(fig)
        time.sleep(0.25)

    st.subheader("Log")
    history_data = {
        "Step": list(range(len(x_history))),
        "Theta": [f"{v:.4f}" for v in x_history],
        "Loss": [f"{v:.4f}" for v in y_history],
    }
    st.dataframe(history_data, width='stretch')
else:
    st.info("Press \'Execute\' to start gradient descent.")