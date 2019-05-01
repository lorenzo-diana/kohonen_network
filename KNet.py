import numpy as np
from math import inf
from time import sleep

class K_Net:
	# type of map allowed
	valid_map_type=['linear', 'ring', 'grid']
	# used to state if the object is initialized or not
	num_input_neuron=-1 # object not initialized
	
	"""
	Initialize the object.

	:param in_n: number of input neurons
	:type in_n: integer
	:param out_n: number of output neurons if if out_type!='grid'. If out_type=='grid' the number of output neurons will be out_n**2
	:type out_n: integer
	:param out_type: map type of the output neurons, it must be one of valid_map_type
	:type out_type: string
	:param tr_set: matrix containing the training set. Each row must be a sample. The number of column must be in_n. Value must be floating point between 0 and 1.
	:type tr_set: numpy array
	:param max_ep: maximum epoch to perform during learning phase. It must be greater than zero.
	:type max_ep: integer
	:param e_stop: if the maximum weight update in the last epoch is less than e_stop then the learning phase stop. If e_stop==0 early sto is disabled and learning phase will perform max_ep epochs. It must be greater than zero.
	:type e_stop: float
	:returns: 1 on success, -1 otherwise. If -1 is returned learning_phase cannot be performed.
	:rtype: integer
	"""
	def init(self, in_n, out_n, out_type, tr_set, max_ep, e_stop=0):
		if in_n<1 or out_n<1 or max_ep<1 or e_stop<0 or tr_set.all()<0 or tr_set.all()>1:
			return -1
		
		if out_type in self.valid_map_type:
			self.map_type=out_type
		else:
			return -1
		
		self.num_input_neuron=in_n
		self.num_output_neuron_x=out_n
		self.training_set=tr_set
		self.max_epoch=max_ep
		
		self.early_stop_min_update=e_stop
		
		# set the number of output neurons
		self.num_output_neuron=self.num_output_neuron_x**self.map_dimension(self.map_type)
		
		# set the parameters for the "r" and "a" parameter
		self.R_max=np.power(self.num_output_neuron, (1/self.map_dimension(self.map_type)))/2
		self.R_min=1
		self.A_max=1
		self.A_min=0
		
		return 1
	
	"""
	Perform the learning phase and, if fig and ax are provided, plot the weights of the output neurons after each epoch.

	:param fig: uset to update the plot
	:type fig: matplotlib.figure
	:param ax: used to plot the weights of the output neurons
	:type ax: matplotlib.axes.Axes
	:param connect_weights_points: if True the weights in the plot will be connected by a line, oterwise not.
	:type connect_weights_points: Boolean
	:param sleep_after_epoch: number of seconds to wait before update the plot after each epoch. It must be a positive number.
	:type sleep_after_epoch: integer
	:returns: -1 if the oject is not initialized. Otherwise a numpy array with num_input_neuron rows and num_output_neuron columns representing the trained weights of the output neurons.
	:rtype: integer in case of error, oterwise numpy array
	"""
	def learning_phase(self, fig=None, ax=None, connect_weights_points=True, plot_annotation=False, sleep_after_epoch=0):
		if self.num_input_neuron==-1 or sleep_after_epoch<0:
			return -1
		sleep_after_epoch=int(sleep_after_epoch)
		
		# initialize the parameter of the learning phase
		epoch=0
		a=self.a_t(epoch)
		r=self.r_t(epoch)
		
		max_weights_update=self.early_stop_min_update
		
		# create random weights each time the learning phase is performed
		weights=np.random.rand(self.num_input_neuron, self.num_output_neuron)
		
		# plot the weights and relative annotations
		ann_pos = []
		if (fig!=None and ax!=None):
			line_weights, = ax.plot(weights[0,:], weights[1,:], 'rX-' if (connect_weights_points==True) else 'rX')
			if plot_annotation==True:
				lab=range(len(weights[0,:]))
				for i in range(self.num_output_neuron):
					ann_pos.append( ax.annotate(lab[i]+1, (weights[0,i], weights[1,i])) )
		
		while True:
			# stops if the parmeter of the learning phase reach their minimun or if the early stop condition occurs
			if ((a<=self.A_min) or (epoch>=self.max_epoch) or (max_weights_update<self.early_stop_min_update)):
				break
			# reset at the beginning of each epoch, save the max absolute update value used in the current epoch
			max_weights_update=0
			# perform an epoch
			for sample in self.training_set:
				# select the neuron closest to the current input sample
				winner=self.min_dist(sample, weights)
				# update the winner neuron and the neurons in its neighbourhood
				for i in np.arange(self.num_output_neuron):
					dist=self.distance(winner, i)
					if dist<r: # if the neuron i-th is close to the winner one 
						scale=a*self.phi(dist, r) # more close the i-th neuron is to the winner one greater will be its update
						w_update=(sample-weights[:,i])*scale
						weights[:,i] += w_update # update the weights of the i-th neuron
						
						if np.absolute(w_update).max()>max_weights_update:
							max_weights_update=np.absolute(w_update).max()
			
			# update the weights and relative annotations in the plot
			if (fig!=None and ax!=None):
				if sleep_after_epoch!=0:
					sleep(sleep_after_epoch)
				
				line_weights.set_xdata(weights[0,:])
				line_weights.set_ydata(weights[1,:])
				
				if plot_annotation==True:
					for i in range(self.num_output_neuron):
						ann_pos[i].set_position((weights[0,i], weights[1,i]))
				
				ax.set_title('Epoch '+str(epoch+1))
				
				fig.canvas.draw()
				fig.canvas.flush_events()
			
			# update the parameter after each epoch
			epoch+=1
			a=self.a_t(epoch)
			r=self.r_t(epoch)
		
		# if early stop occurred update the plot title
		if (fig!=None and ax!=None) and (max_weights_update<self.early_stop_min_update):
			ax.set_title('Epoch '+str(epoch)+' - (Early stop)')
			fig.canvas.draw()
			fig.canvas.flush_events()
		
		return weights
	
	# retrun the number of dimensions of the selected map type
	def map_dimension(self, map_t):
		switch = {
			'linear': 1, # one-dimensional map
			'ring': 1, # one-dimensional map
			'grid': 2 # two-dimensional map
		}
		return switch.get(map_t, 0)
	
	# define how the "r" parameter of the learning phase shuold change over epochs
	def r_t(self, epoch):
		return self.R_min + (self.R_max-self.R_min) * np.power(np.e, -1*epoch)
	
	# define how the "a" parameter of the learning phase shuold change over epochs
	def a_t(self, epoch):
		return self.A_min + (self.A_max-self.A_min) * np.power(np.e, -1*epoch)

	"""
	Compute a value that state how much close two point are to each other.

	:param dis: distance between two point considered close to each other, so this value must be <= ra
	:param ra: maximum distance for which two point are considered close to each other
	:returns: 1 if dis==0, that means that the two points are overlapped. 0 if the two points are as much as possible far to each other, that means dis==ra
	:rtype: float
	"""
	def phi(self, dis, ra):
		return 1-((dis*dis)/(ra*ra))

	# define how to measure the distance between two point for the linear map type
	def distance_linear(self, i, j):
		return abs(i-j)

	# define how to measure the distance between two point for the ring map type
	def distance_ring(self, i, j):
		if abs(i-j) > self.num_output_neuron/2:
			return self.num_output_neuron - abs(i-j)
		return abs(i-j)

	# define how to measure the distance between two point for the grid map type
	def distance_grid(self, i, j):
		i_=np.array([0,0])
		j_=np.array([0,0])
		i_[0]=i%self.num_output_neuron_x
		i_[1]=int(i/self.num_output_neuron_x)
		j_[0]=j%self.num_output_neuron_x
		j_[1]=int(j/self.num_output_neuron_x)
		
		return sum(abs(i_-j_)) # distant points seem nearest to each other
		#return sum((i_-j_)**2) # close points seem far to each other

	# select the appropriate function the measure the distance based on the map type
	def distance(self, i, j):
		switch = {
			'linear': self.distance_linear,
			'ring': self.distance_ring,
			'grid': self.distance_grid
		}
		func = switch.get(self.map_type, lambda: "Error: no valid network type!")
		return func(i, j)

	# compute the distances between a sample "s" and aech sample of the weights "w" and return the smallest one
	def min_dist(self, s, w):
		min_d=inf
		min_i=-1
		i=0
		for col in w.T:
			d=sum((s-col)**2)
			if d<min_d:
				min_d=d
				min_i=i
			i+=1
		return min_i
