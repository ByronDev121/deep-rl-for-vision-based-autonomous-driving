import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import make_interp_spline, BSpline

# Create the vectors X and Y
theta_1 = np.array(range(-1, 6))
# y = 1/2 * (x ** 2)
# y_prim = x

x_1 = 1
x_2 = 3
x_3 = 4

y_1 = 2
y_2 = 6
y_3 = 8

Z = (0.5/3)*(
    ((theta_1*x_1)-y_1)**2 +
    ((theta_1*x_2)-y_2)**2 +
    ((theta_1*x_3)-y_3)**2
)

Z_prime = (1/3)*(
    ((theta_1*x_1)-y_1)*x_1 +
    ((theta_1*x_2)-y_2)*x_2 +
    ((theta_1*x_3)-y_3)*x_3
)

xnew = np.linspace(theta_1.min(), theta_1.max(), 300)

spl = make_interp_spline(theta_1, Z, k=3)  # type: BSpline
power_smooth = spl(xnew)

# Create the plot
plt.plot(xnew, power_smooth,label='h\'(\u03B8) = 1/2 x^2')
plt.plot(theta_1,Z_prime,label='h\'(\u03B8) = x')

# Add a title
# plt.title('My first Plot with Python')

# Add X and y Label
plt.xlabel('\u03B8')
# plt.ylabel('Cost')

# Add a grid
plt.grid(alpha=.4, linestyle='--')

# Add a Legend
plt.legend()

# Show the plot
plt.show()
