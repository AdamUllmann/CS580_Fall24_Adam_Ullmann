# Adam Ullmann
# 011215244
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# load the data
data = pd.read_csv('linear_regression_data.csv', header=None)
data.columns = ['x', 'y']
x = data['x']
y = data['y']
# covariance linear regression
y_mean = np.mean(y)
x_mean = np.mean(x)
var_x = np.sum((x - x_mean) ** 2) / len(x)
cov_xy = np.sum((x - x_mean) * (y - y_mean)) / len(x)
slope = cov_xy / var_x
y_intercept = y_mean - slope * x_mean
y_pred = slope * x + y_intercept
# draw the graph
plt.style.use('dark_background')
plt.plot(x, y_pred, color='red', label='Regression Line')
plt.scatter(x, y, color='cyan', label='Data', s=20)
plt.xlabel('Independent x')
plt.ylabel('Dependent y')
plt.title('Linear Regression w/ covariance')
plt.text(x=min(x), y=max(y), s="Slope (m): {:.5f}\nIntercept (b): {:.5f}".format(slope, y_intercept), color='white', verticalalignment='top', horizontalalignment='left', fontsize=10)
plt.legend(loc='lower right')
plt.savefig('linear_regression_plot.png', format='png')
# plt.show()
print("Slope (m): {:.15f}".format(slope))
print("y-Intercept (b): {:.15f}".format(y_intercept))
