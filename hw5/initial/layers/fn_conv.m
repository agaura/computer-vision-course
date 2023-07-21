% ----------------------------------------------------------------------
% input: in_height x in_width x num_channels x batch_size
% output: out_height x out_width x num_filters x batch_size
% hyper parameters: (stride, padding for further work)
% params.W: filter_height x filter_width x filter_depth x num_filters
% params.b: num_filters x 1
% dv_output: same as output
% dv_input: same as input
% grad.W: same as params.W
% grad.b: same as params.b
% ----------------------------------------------------------------------

function [output, dv_input, grad] = fn_conv(input, params, hyper_params, backprop, dv_output)

[~,~,num_channels,batch_size] = size(input);
[~,~,filter_depth,num_filters] = size(params.W);
assert(filter_depth == num_channels, 'Filter depth does not match number of input channels');

out_height = size(input,1) - size(params.W,1) + 1;
out_width = size(input,2) - size(params.W,2) + 1;
output = zeros(out_height,out_width,num_filters,batch_size);

for j = 1:batch_size
    for i = 1:num_filters
        conv_layers = zeros(out_height,out_width) + params.b(i);
        for k = 1:num_channels
            conv_layers = conv_layers + conv2(input(:,:,k,j),params.W(:,:,k,i),'valid');
        end
        output(:,:,i,j) = conv_layers;
    end
end

dv_input = [];
grad = struct('W',[],'b',[]);

if backprop
	dv_input = zeros(size(input));
	grad.W = zeros(size(params.W));
	grad.b = zeros(size(params.b));
    for k = 1:batch_size
        for i = 1:num_channels
            conv_layers = zeros(size(input(:,:,1,1)));
            for j = 1:num_filters
                conv_layers = conv_layers + conv2(dv_output(:,:,j,k),rot90(rot90(params.W(:,:,i,j))),'full');
            end
            dv_input(:,:,i,k) = conv_layers;
        end
    end
    for j = 1:num_filters
        for i = 1:batch_size
            grad.b(j,1) = grad.b(j,1) + sum(sum(dv_output(:,:,j,i)));
        end
        grad.b(j,1) = grad.b(j,1)/batch_size;
        for i = 1:num_channels
            conv_layers = zeros(size(params.W(:,:,1,1)));
            for k = 1:batch_size
                conv_layers = conv_layers + conv2(rot90(rot90(input(:,:,i,k))),dv_output(:,:,j,k),'valid');
            end
            grad.W(:,:,i,j) = conv_layers/batch_size;
        end
    end
end