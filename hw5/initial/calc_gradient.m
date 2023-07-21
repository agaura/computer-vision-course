function [grad] = calc_gradient(model, input, activations, dv_output)
% Calculate the gradient at each layer, to do this you need dv_output
% determined by your loss function and the activations of each layer.
% The loop of this function will look very similar to the code from
% inference, just looping in reverse.

num_layers = numel(model.layers);
grad = cell(num_layers,1);

for i = num_layers:-1:1
    if i == num_layers
        [~,dv_input,ngrad] = model.layers(i).fwd_fn(cell2mat(activations(i-1)),model.layers(i).params,model.layers(i).hyper_params,1,dv_output);
    elseif i == 1
        [~,~,ngrad] = model.layers(i).fwd_fn(input,model.layers(i).params,model.layers(i).hyper_params,1,dv_input);
    else
        [~,dv_input,ngrad] = model.layers(i).fwd_fn(cell2mat(activations(i-1)),model.layers(i).params,model.layers(i).hyper_params,1,dv_input);
    end
    grad(i) = {ngrad};
end
