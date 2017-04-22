import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy import optimize

m  = 0;

def loadData(filename):
	content = [];
	#Load Data into list	
	with open(filename) as f:
		content = f.readlines()
	content = [x.strip() for x in content] 

	num_columns = len(content[1].split(","));
	num_rows = len(content);

	x_data = np.zeros(shape=(num_rows, num_columns-1));
	y_data = np.zeros(shape=(num_rows, 1));


	#Load Data into numpy array	
	for i in range(0, len(content)):
		line = content[i].split(",");
		flower = line[num_columns-1];
		if flower == "Iris-setosa":
			y_data[i] = 1;
		elif flower == "Iris-versicolor" or flower == "Iris-virginica":
			y_data[i] = 0;
		for j in range(0, num_columns-1):
			x_val = line[j];
			x_data[i,j] = x_val;

	#Normalize The Data
	for i in range(0, num_columns-1): 
		x_min = min(x_data[:,i]);
		x_max = max(x_data[:,i]);
		x_diff = x_max - x_min;
		for j in range(0, num_rows):
			x_data[j,i] = (x_data[j,i]-x_min)/(x_diff);

	return [x_data, y_data];

def plotData(x, y):
	plt.figure(figsize=(10,6));
	
	pos = np.array([x[i] for i in xrange(m) if y[i][0] == 1]);
	neg = np.array([x[i] for i in xrange(m) if y[i][0] == 0]);


	plt.plot(pos[:,0], pos[:,1], 'g+', label="Iris Setosa");
	plt.plot(neg[:,0], neg[:,1], 'mo', label="Iris Versicolor");
	plt.ylabel("Sepal Width (cm)");
	plt.xlabel("Sepal Length (cm)");
	plt.legend();
	plt.grid(True);


if __name__ == "__main__":
	[x_data, y_data] = loadData("iris_data.csv");
	(m, n) = x_data.shape;
	
	plotData(x_data, y_data);
	plt.show();

