import numpy as np
from tkinter import *

from matplotlib import pyplot as plt
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from KNet import *
from dist_list import *

# global values
ax=None
btn_test_sample=None
my_net=None
trained_weights=None
canvas=None
training_set=None
connect_weights_points=True
window = Tk()

# select one of the input sample, then highlights that sample and the neuron closest to it (the neuron that shuol recognize the activation that input)
def btn_test_clicked():
	input=training_set[int(np.random.rand(1)*(training_set.shape[0]-1)) , :]
	max_out_index=my_net.min_dist(input, trained_weights)
	if btn_test_sample[0]==None:
		btn_test_sample[0], = ax.plot(input[0], input[1], 's', c='#d94ffb') #'#01ff07')
		btn_test_sample[1], = ax.plot(trained_weights[0,max_out_index], trained_weights[1,max_out_index], 'X', c='#1805db') #'#0165fc')
	else:
		btn_test_sample[0].set_xdata(input[0])
		btn_test_sample[0].set_ydata(input[1])
		btn_test_sample[1].set_xdata(trained_weights[0,max_out_index])
		btn_test_sample[1].set_ydata(trained_weights[1,max_out_index])
	canvas.draw()

# creates the Kohonen network and trains it
def btn_start_clicked():
	global my_net
	global trained_weights
	global canvas
	global training_set
	global ax
	global btn_test_sample
	# disable start and test buttons during execution of the funxtion
	btn_test["state"] = DISABLED
	btn_start["state"] = DISABLED
	btn_start["text"]="Working..."
	btn_test_sample=[None, None]
	# close plot from previously run of this function
	plt.close('all')
	fig=Figure(figsize=(5,5), dpi=100)
	ax=fig.add_subplot(111)
	ax.set_xlim([0, 1])
	ax.set_ylim([0, 1])
	# get the parameter for the Kohonen network from the GUI elements
	num_input_neuron=2
	num_output_neuron=int(spin_num_out_n.get())
	num_training_set=int(spin_num_tr_sample.get())
	max_epoch=int(spin_max_epoch.get())
	net_topology=topology_type.get()
	sample_function=sample_function_type.get()
	plot_label=True if plot_label_enable.get()==1 else False
	delay=delay_enable.get() # delay will be 1 or 0
	e_stop_limit= 1e-5 if (early_stop_enable.get()==1) else 0
	
	'''
	# print debug info
	print('topology_type: ' + net_topology)
	print('spin_num_tr_sample: ' + str(num_training_set))
	print('spin_num_out_n: ' + str(num_output_neuron))
	print('spin_max_epoch: ' + str(max_epoch))
	print('sample function: ' + sample_function)
	print('plot label: ' + str(plot_label))
	print('delay: ' + str(delay))
	print('early stop limit: '+str(e_stop_limit))
	'''
	
	# create the training set
	training_set=get_samples(num_training_set, num_input_neuron, get_dist_fun(sample_function))
	
	# plot the training set
	ax.plot(training_set[:,0], training_set[:,1], 'cs')
	canvas = FigureCanvasTkAgg(fig, master=window)
	canvas.draw()
	canvas.get_tk_widget().grid(column=0, row=9, columnspan=5, sticky=W)
	
	# create the Kohonen network and initialize it
	my_net=K_Net()
	my_net.init(num_input_neuron, num_output_neuron, net_topology, training_set, max_epoch, e_stop_limit)
	# if initialization fails then exit() (some init parameter could be wrong check them!)
	if my_net==-1:
		print('Error during initialization!')
		exit()
	
	# train the network
	trained_weights=my_net.learning_phase(fig, ax, connect_weights_points, plot_label, delay)
	# if learning phase fails then exit() (some learning_phase parameter could be wrong check them!)
	if my_net==-1:
		print('Error during learning phase!')
		exit()
	
	# restore normal behavior for start and test buttons
	btn_start["state"] = NORMAL
	btn_start["text"]="Start"
	btn_test["state"] = NORMAL

def topology_type_change_callback(*args):
	global connect_weights_points
	if topology_type.get()=='grid':
		lb_num_out_n_var.set('Num. output neuron (to be squared):')
		connect_weights_points=False
	else:
		lb_num_out_n_var.set('Num. output neuron:')
		connect_weights_points=True

if __name__ == "__main__":
	window.title("Kohonen network example")
	window.geometry('500x720')
	
	# create variable for the GUI elements
	topology_type=StringVar()
	topology_type.set('linear')
	topology_type.trace("w", topology_type_change_callback)
	
	sample_function_type=StringVar()
	sample_function_type.set('love')
	
	lb_num_out_n_var=StringVar()
	lb_num_out_n_var.set('Num. output neuron:')
	
	plot_label_enable=IntVar()
	plot_label_enable.set(0)
	
	delay_enable=IntVar()
	delay_enable.set(0)
	
	early_stop_enable=IntVar()
	early_stop_enable.set(1)
	
	# create the GUI elements and place them 
	lb_network_topology_menu = Label(window, text="Network topology:")
	lb_network_topology_menu.grid(column=0, row=0, sticky=W)
	network_topology_menu = OptionMenu(window, topology_type, 'linear', 'ring', 'grid')
	network_topology_menu.config(width=5)
	network_topology_menu.grid(column=1, row=0, sticky=W)
	
	options=get_dist_list()
	lb_function_menu = Label(window, text="Sample function:")
	lb_function_menu.grid(column=0, row=1, sticky=W)
	function_menu = OptionMenu(window, sample_function_type, *options)
	function_menu.config(width=7)
	function_menu.grid(column=1, row=1, sticky=W)
	
	lb_num_tr_sample = Label(window, text="Num. training samples:")
	lb_num_tr_sample.grid(column=0, row=2, sticky=W)
	spin_num_tr_sample = Spinbox(window, from_=1, to=1000, width=6)
	spin_num_tr_sample.grid(column=1, row=2, sticky=W)
	
	lb_num_out_n = Label(window, textvariable=lb_num_out_n_var) #text="Num. output neuron:")
	lb_num_out_n.grid(column=0, row=3, sticky=W)
	spin_num_out_n = Spinbox(window, from_=1, to=1000, width=6)
	spin_num_out_n.grid(column=1, row=3, sticky=W)
	
	lb_max_epoch = Label(window, text="Max. epoch:")
	lb_max_epoch.grid(column=0, row=4, sticky=W)
	spin_max_epoch = Spinbox(window, from_=1, to=100, width=6)
	spin_max_epoch.grid(column=1, row=4, sticky=W)
	
	early_stop_check = Checkbutton(window, text="Enable early stop", variable=early_stop_enable)
	early_stop_check.grid(column=0, row=5, sticky=W)
	
	plot_label_check = Checkbutton(window, text="Plot labels on output neuron", variable=plot_label_enable)
	plot_label_check.grid(column=0, row=6, columnspan=2, sticky=W)
	
	delay_check = Checkbutton(window, text="Delay chart update by 1 second after each epoch", variable=delay_enable)
	delay_check.grid(column=0, row=7, columnspan=2, sticky=W)
	
	# button to create the network and start the learning phase
	btn_start = Button(window, text="Start", command=btn_start_clicked)
	btn_start.grid(column=0, row=8)
	
	btn_test = Button(window, text="Test random sample", command=btn_test_clicked, state=DISABLED)
	btn_test.grid(column=1, row=8)
	
	window.mainloop()
