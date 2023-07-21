% ----------------------------------------------------------------------
% input: num_nodes x batch_size
% output: num_nodes x batch_size
% ----------------------------------------------------------------------

function [output, dv_input, grad] = fn_softmax(input, params, hyper_params, backprop, dv_output)

[num_classes,batch_size] = size(input);
output = zeros(num_classes, batch_size);
for j = 1:batch_size
    denom = sum(exp(input(:,j)));
    for i = 1:num_classes
        output(i,j) = exp(input(i,j))/denom;
    end
end

dv_input = [];

% This is included to maintain consistency in the return values of layers,
% but there is no gradient to calculate in the softmax layer since there
% are no weights to update.
grad = struct('W',[],'b',[]); 

if backprop
	dv_input = zeros(size(input));
    for k = 1:batch_size
        denom = sum(exp(input(:,k)));
        mat = zeros(num_classes,num_classes);
        for i = 1:num_classes
            for j = 1:num_classes
                if i == j
                    mat(i,j) = (denom*exp(input(i,k))-exp(2*input(i,k)))/denom^2;
                else
                    mat(i,j) = -1*exp(input(i,k)+input(j,k))/denom^2;
                end
            end
        end
        dv_input(:,k) = mat*dv_output(:,k);
    end
end