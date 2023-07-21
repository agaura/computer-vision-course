% ----------------------------------------------------------------------
% input: num_nodes x batch_size
% labels: batch_size x 1
% ----------------------------------------------------------------------

function [loss, dv_input] = loss_crossentropy(input, labels, hyper_params, backprop)

assert(max(labels) <= size(input,1));

[~,m] = size(input);
tot = 0;
for i = 1:m
    tot = tot - log(input(labels(i,1),i));
end
loss = tot/m;

dv_input = zeros(size(input));
if backprop
    for i = 1:m
        dv_input(labels(i,1),i) = -1/(input(labels(i,1),i));
    end
end