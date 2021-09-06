# Parallel-Binary-Classification
this project solves A changing Binary Classification problem in parrlel using OpenMp, MPI and Cuda.

•	N - number of points

•	K – number of coordinates of points

•	Coordinates of all points with attached value: 1 for those that belong to set A and -1 for the points that belong to set B.

•	dT – increment value of t, tMAX – maximum value of t

•	alpha - conversion ratio

•	LIMIT – the maximum number of iterations. 

•	QC – Quality of Classifier to be reached 

The first line of the file contains   N    K    dT   tMAX   alpha      LIMIT   QC.  
Next lines are initial coordinates of all points, one per line, its velocity and attached value 1 or -1.

The output file contains following information

•	The minimal value of t with q < QC and a value of q. If for every checked value of t the value of q is bigger than QC – “time was not found” is printed.

•	Values of corresponding weights W
