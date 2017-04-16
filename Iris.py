import numpy as np
import matplotlib.pyplot as plt
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
		elif flower == "Iris-versicolor":
			y_data[i] = 0;
		for j in range(0, len(line)-1):
			x_val = line[j];
			x_data[i][j-1] = x_val;

	#Normalize The Data
	for i in range(0, num_columns-1): 
		x_min = min(x_data[:,i]);
		x_max = max(x_data[:,i]);
		x_diff = x_max - x_min;
		for j in range(0, num_rows):
			x_data[j,i] = (x_data[j,i]-x_min)/(x_diff);
	return [x_data, y_data];

if __name__ == "__main__":
	[x_data, y_data] = loadData("iris_data.csv");
	print x_data;
	print y_data;

