import time
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import sympy as sym

st.set_page_config(page_title="Gradient Descent Laboratory")
st.title("민수의 Gradient Descent 연구소")
        
x, y = sym.symbols('x y')
input_func = sym.sin(x) * sym.exp((1 - sym.cos(y))**2) + sym.cos(y) * sym.exp((1 - sym.sin(x))**2) + (x - y)**2
loss_func = sym.lambdify((x, y), input_func, 'numpy')
gradient_x = sym.Derivative(input_func, x).doit()
gradient_y = sym.Derivative(input_func, y).doit()

st.sidebar.header("Loss Function")
st.sidebar.latex(r'''L(x, y) = \sin(x) e^{(1 - \cos(y))^2} + \cos(y) e^{(1 - \sin(x))^2} + (x - y)^2''')

st.sidebar.header("Parameters")
theta_x_init = st.sidebar.slider("Initial x", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
theta_y_init = st.sidebar.slider("Initial y", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
learning_rate = st.sidebar.slider("Learning Rate", min_value=0.0001, max_value=0.01, value=0.0001, step=0.0001)

x_vals = np.linspace(-10, 10, 100)
y_vals = np.linspace(-10, 10, 100)
X, Y = np.meshgrid(x_vals, y_vals)
Z = loss_func(X, Y)

chart_placeholder = st.empty()

fig, ax = plt.subplots(figsize=(8, 6))
contour = ax.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.8)
fig.colorbar(contour, ax=ax, label='Loss')
ax.plot(theta_x_init, theta_y_init, 'ro--', label='Path', markersize=3)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.grid(True)
ax.legend()
chart_placeholder.pyplot(fig)
plt.close(fig)

if st.sidebar.button("Execute"):
    theta_x = float(theta_x_init)
    theta_y = float(theta_y_init)
    x_history = [theta_x]
    y_history = [theta_y]
    loss_history = [float(input_func.subs([(x, theta_x), (y, theta_y)]))]

    for i in range(50):
        dx = float(gradient_x.subs([(x, theta_x), (y, theta_y)]))
        dy = float(gradient_y.subs([(x, theta_x), (y, theta_y)]))
        theta_x = theta_x - learning_rate * dx
        theta_y = theta_y - learning_rate * dy
        loss = float(input_func.subs([(x, theta_x), (y, theta_y)]))
        x_history.append(theta_x)
        y_history.append(theta_y)
        loss_history.append(loss)
        if input_func.subs([(x, theta_x), (y, theta_y)]) < -100.0:
            break

    for step in range(len(x_history)):
        fig, ax = plt.subplots(figsize=(8, 6))
        contour = ax.contourf(X, Y, Z, levels=30, cmap='viridis', alpha=0.8)
        fig.colorbar(contour, ax=ax, label='Loss')
        ax.plot(x_history[:step + 1], y_history[:step + 1], 'ro--', label='Path', linewidth=1, markersize=3)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.grid(True)
        ax.legend()

        chart_placeholder.pyplot(fig)
        plt.close(fig)
        time.sleep(0.01)

    st.subheader("Log")
    history_data = {
        "Step": list(range(len(x_history))),
        "Theta x": [f"{v:.4f}" for v in x_history],
        "Theta y": [f"{v:.4f}" for v in y_history],
        "Loss": [f"{v:.4f}" for v in loss_history],
    }
    st.dataframe(history_data, width='stretch')
else:
    st.info("Press \'Execute\' to start gradient descent.")