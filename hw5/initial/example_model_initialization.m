% Basic script to create a new network model

addpath layers;

% filter depth originally 3
l = [init_layer('conv',struct('filter_size',2,'filter_depth',1,'num_filters',2))
	init_layer('pool',struct('filter_size',2,'stride',2))
	init_layer('relu',[])
	init_layer('flatten',struct('num_dims',4))
	init_layer('linear',struct('num_in',338,'num_out',10)) % the num_in was also altered
	init_layer('softmax',[])];

model = init_model(l,[28 28 1],10,true);

% Example calls you might make:
% [output,~] = inference(model,input);
% [loss,~] = loss_euclidean(output,ground_truth,[],false);