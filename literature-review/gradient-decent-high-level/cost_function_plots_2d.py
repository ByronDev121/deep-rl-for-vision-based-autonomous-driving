import matplotlib.pyplot as plt
import numpy as np

from scipy.interpolate import make_interp_spline, BSpline

# Create the vectors X and Y
theta_1 = np.array(range(-1, 6))

x = np.array(range(-5, 5))
y = x * 2

m = theta_1.shape[0]
Z = np.zeros([theta_1.shape[0]])
Z_prime = np.zeros([theta_1.shape[0]])

for i in range(len(x)):
    Z += ((theta_1*x[i])-y[i])**2
    Z_prime += ((theta_1*x[i])-y[i])*x[i]

Z = (0.5/m)*Z
Z_prime = (1/m)*Z_prime

xnew = np.linspace(theta_1.min(), theta_1.max(), 300)

spl = make_interp_spline(theta_1, Z, k=3)  # type: BSpline
power_smooth = spl(xnew)

# Create the plot
plt.plot(xnew, power_smooth,label='J(\u03B8) = 1/2 x^2')
plt.plot(theta_1, Z_prime, label='J\'(\u03B8) = x')

# Add a title
# plt.title('My first Plot with Python')

# Add X and y Label
plt.xlabel('\u03B8')

# Add a grid
plt.grid(alpha=.4, linestyle='--')

# Add a Legend
plt.legend()

# Show the plot
plt.show()


plt.subplot()

x = np.array(range(50, 57))
y = np.array(range(50, 57))
for i in range(len(x)):
    y[i] = (
        8/100000000 * x[i] -
        6/100000000 * x[i] ** 2 +
        5/100000000 * x[i] ** 3 -
        4/100000000 * x[i] ** 4 +
        3/100000000 * x[i] ** 5 -
        2/100000000 * x[i] ** 6
    )

x_new = np.linspace(x.min(), x.max(), 300)
spl_new = make_interp_spline(x, y, k=3)  # type: BSpline
power_smooth_new = spl_new(x_new)
plt.plot(x_new, power_smooth_new, label='h\'(\u03B8)')
plt.xlabel('\u03B8')
plt.ylabel('J(\u03B8)')
plt.show()


def plot():
    print('test')


def main():
    plot()

if __name__ == '__main__':
    main()

