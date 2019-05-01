import numpy as np
from math import radians, sin, cos

# it samples num_row samples from dist function. Each sample will have num_col values
def get_samples(num_row, num_col, dist):
	training_set=np.zeros((1, num_col))
	for i in range(num_row):
		while True:
			point=np.random.rand(1, num_col)
			if dist(point)==True:
				training_set=np.vstack([training_set,point])
				break
	training_set=training_set[1:,:]
	return training_set

# returns a list of available function to be used for get_samples()
def get_dist_list():
	return ['love', 'love edge', 'square', 'pretzel', 'sin', 'circle']

# returns the appropriate funciton according to the name specified in fun_name
def get_dist_fun(fun_name):
	switch = {
		'sin': dist_sin,
		'circle': dist_circle,
		'love': dist_love,
		'love edge': dist_love_edge,
		'pretzel': dist_pretzel,
		'square': dist_square
	}
	return switch.get(fun_name, lambda: "Error: no valid sample function selected!")

def dist_square(point, side=0.6, center=[0.5, 0.5]):
	if (point[0,0]>=(center[0]-side/2) and point[0,0]<=(center[0]+side/2)) and (point[0,1]>=(center[1]-side/2) and point[0,1]<=(center[1]+side/2)):
		return True
	else:
		return False

def dist_sin(point, thickness=0.4):
	y=(np.sin(10*(point[0,0]))+1)/2
	if point[0,1]>=y and point[0,1]<=(y+thickness):
		return True
	else:
		return False

def dist_circle(point):
	x_c=0.5 # center of the circle
	y_c=0.5
	r=0.1 # radius of the circle
	r_point=(point[0,0]-x_c)**2 + (point[0,1]-y_c)**2
	if r_point<=r:
		return True
	else:
		return False
	
def dist_love(point):
	#ellipse eq: res1=(((point[0,0]-h1)**2)/(a1**2)) + (((point[0,1]-k1)**2)/(b1**2))
	# first ellipse
	a1=0.2
	b1=0.4
	h1=0.4 # x position of the center of the ellipse
	k1=0.5 # y position of the center of the ellipse
	p1=p2=0.75
	teta1=radians(60) # rotation of the ellipse in degrees
	res1=((( (point[0,0]-h1)*sin(teta1) + (point[0,1]-k1)*cos(teta1) )**2)/(a1**2))  +  ((( (point[0,0]-h1)*cos(teta1) - (point[0,1]-k1)*sin(teta1) )**2)/(b1**2))
	
	# second ellipse
	a2=0.2
	b2=0.4
	h2=0.6 # x position of the center of the ellipse
	k2=0.5 # y position of the center of the ellipse
	teta2=radians(-60)  # rotation of the ellipse in degrees
	res2=((( (point[0,0]-h2)*sin(teta2) + (point[0,1]-k2)*cos(teta2) )**2)/(a2**2))  +  ((( (point[0,0]-h2)*cos(teta2) - (point[0,1]-k2)*sin(teta2) )**2)/(b2**2))
	
	# select only points inside the "love" shape
	if res1 <= p1 or res2 <= p2:
		return True
	else:
		return False

def dist_love_edge(point, thickness=0.4):
	a1=0.2
	b1=0.4
	h1=0.4
	k1=0.5
	p1=p2=0.75
	teta1=radians(60)
	res1=((( (point[0,0]-h1)*sin(teta1) + (point[0,1]-k1)*cos(teta1) )**2)/(a1**2))  +  ((( (point[0,0]-h1)*cos(teta1) - (point[0,1]-k1)*sin(teta1) )**2)/(b1**2))
	
	a2=0.2
	b2=0.4
	h2=0.6
	k2=0.5
	teta2=radians(-60)
	res2=((( (point[0,0]-h2)*sin(teta2) + (point[0,1]-k2)*cos(teta2) )**2)/(a2**2))  +  ((( (point[0,0]-h2)*cos(teta2) - (point[0,1]-k2)*sin(teta2) )**2)/(b2**2))
	
	# select only points on the border of the "love" shape
	if (res1>=(p1-thickness/2) and res1<=(p1+thickness/2) and res2>(p2+thickness/8)) or (res2>=(p2-thickness/2) and res2<=(p2+thickness/2) and res1>(p1+thickness/8)):
		return True
	else:
		return False

def dist_pretzel(point, thickness=0.4):
	a1=0.2
	b1=0.4
	h1=0.4
	k1=0.5
	p1=p2=0.75
	teta1=radians(45)
	res1=((( (point[0,0]-h1)*sin(teta1) + (point[0,1]-k1)*cos(teta1) )**2)/(a1**2))  +  ((( (point[0,0]-h1)*cos(teta1) - (point[0,1]-k1)*sin(teta1) )**2)/(b1**2))
	
	a2=0.2
	b2=0.4
	h2=0.6
	k2=0.5
	teta2=radians(-45)
	res2=((( (point[0,0]-h2)*sin(teta2) + (point[0,1]-k2)*cos(teta2) )**2)/(a2**2))  +  ((( (point[0,0]-h2)*cos(teta2) - (point[0,1]-k2)*sin(teta2) )**2)/(b2**2))
	
	# select only points on the border of both ellipses
	if (res1>=(p1-thickness/2) and res1<=(p1+thickness/2)) or (res2>=(p2-thickness/2) and res2<=(p2+thickness/2)):
		return True
	else:
		return False
