function [output,activations] = inference(model,input)
% Do forward propagation through the network to get the activation
% at each layer, and the final output

num_layers = numel(model.layers);
activations = cell(num_layers,1);

for i = 1:num_layers
    if i == 1
        [out,~,~] = model.layers(i).fwd_fn(input,model.layers(i).params,model.layers(i).hyper_params,0,0);
    else
        [out,~,~] = model.layers(i).fwd_fn(out,model.layers(i).params,model.layers(i).hyper_params,0,0);
    end
    activations(i) = {out};
end

output = activations{end};
