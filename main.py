import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sympy import symbols, Derivative, lambdify, E

x = symbols('x')
input_func = x ** 2
loss_func = lambdify(x, input_func, 'numpy')
gradient = Derivative(input_func, x).doit()

# parameter set
theta = 80.0
learning_rate = 0.3

x_history = [theta]
y_history = [float(input_func.subs(x, theta))]

while(abs(input_func.subs(x, theta)) > 1):
    d = float(gradient.subs(x, theta))
    theta = theta - learning_rate * d
    
    x_history.append(theta)
    y_history.append(float(input_func.subs(x, theta)))


# 4. 애니메이션용 그래프 기본 설정 (Figure & Axes)
fig, ax = plt.subplots(figsize=(8, 5))

# 배경이 될 손실함수 곡선 세팅 (x 범위 설정)
min_x, max_x = -100, 100
x_vals = np.linspace(min_x, max_x, 200)
y_vals = loss_func(x_vals)

ax.plot(x_vals, y_vals, label=r'$f(x) = x^3$', color='lightgray', linewidth=2)
ax.set_title('Gradient Descent Animation')
ax.set_xlabel('x')
ax.set_ylabel('Loss')
ax.grid(True)

line, = ax.plot([], [], 'ro--', label='Path')
point, = ax.plot([], [], 'ro', markersize=8)
ax.legend()

# 5. 애니메이션 프레임 업데이트 함수
def update(frame):
    # frame 스텝까지의 이동 경로 업데이트
    line.set_data(x_history[:frame+1], y_history[:frame+1])
    # 현재 위치 점 표시
    point.set_data([x_history[frame]], [y_history[frame]])
    return line, point

# 6. FuncAnimation 객체 생성 (300ms 마다 1프레임 전환)
anim = animation.FuncAnimation(
    fig, update, frames=len(x_history), interval=300, blit=True
)

# 7. GIF 파일로 저장 (컨테이너 환경용)
anim.save('result.gif', writer='pillow')
print("success")