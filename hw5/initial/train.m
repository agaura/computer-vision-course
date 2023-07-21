function [model, loss, all_train_losses, all_test_losses] = train(model,input,label,params,numIters,test,test_labels)

% Initialize training parameters
% This code sets default values in case the parameters are not passed in.

% Learning rate
if isfield(params,'learning_rate') lr = params.learning_rate;
else lr = .1; end % originally .01
% Weight decay
if isfield(params,'weight_decay') wd = params.weight_decay;
else wd = .005; end % originally .0005
% Batch size
if isfield(params,'batch_size') batch_size = params.batch_size;
else batch_size = 1024; end

% There is a good chance you will want to save your network model during/after
% training. It is up to you where you save and how often you choose to back up
% your model. By default the code saves the model in 'model.mat'
% To save the model use: save(save_file,'model');
if isfield(params,'save_file') save_file = params.save_file;
else save_file = 'model.mat'; end

% update_params will be passed to your update_weights function.
% This allows flexibility in case you want to implement extra features like momentum.
update_params = struct('learning_rate',lr,'weight_decay',wd);

[~,~,~,n] = size(input);
perm = randperm(n);
disp(size(input));
disp(size(perm));
input_shuffle = input;
input_shuffle(:,:,:,perm) = input(:,:,:,:);
label_shuffle = label;
label_shuffle(perm,:) = label(:,:);
update_model = model;

all_train_losses = zeros(21,1);
all_test_losses = zeros(21,1);

% uncomment to get a base case for loss

%hyper_params = struct('num_dims',2);
%disp("Beginning Inference");
%[output,~] = inference(update_model,input);
%disp("Loss:");
%[loss_train, ~] = loss_crossentropy(output, label, hyper_params, 1);
%all_train_losses(1) = loss_train;
%disp("Training Loss:");
%disp(loss_train);
%[output,~] = inference(update_model,test);
%[loss_test, ~] = loss_crossentropy(output, test_labels, hyper_params, 1);
%all_test_losses(1) = loss_test;
%disp("Test Loss:");
%disp(loss_test);
%disp("Losses complete");
for k = 1:numIters
    for i = 1:112
        disp([i k]);
        input_batch = input_shuffle(:,:,:,128*(i-1)+1:128*i);
        label_batch = label_shuffle(128*(i-1)+1:128*i,:);
        for j = 1:1
            [output,activations] = inference(update_model,input_batch);
            hyper_params = struct('num_dims',2);
            [loss, dv_output] = loss_crossentropy(output, label_batch, hyper_params, 1);
            disp(loss);
            grad = calc_gradient(update_model,input_batch,activations,dv_output);
            update_model = update_weights(update_model,grad,update_params);
        end
        
        % uncomment to get loss for every quarter of the data
        
        %if mod(i,28) == 0
        %    hyper_params = struct('num_dims',2);
        %    disp("Beginning Inference");
        %    [output,~] = inference(update_model,input);
        %    disp("Loss:");
        %    [loss_train, ~] = loss_crossentropy(output, label, hyper_params, 1);
        %    disp("Training Loss:");
        %    disp(loss_train);
        %    [output,~] = inference(update_model,test);
        %    [loss_test, ~] = loss_crossentropy(output, test_labels, hyper_params, 1);
        %    disp("Test Loss:");
        %    disp(loss_test);
        %    disp("Losses complete");
        %    all_train_losses(4*(k-1)+i/28+1) = loss_train;
        %    all_test_losses(4*(k-1)+i/28+1) = loss_test;
        %end
    end
end

model = update_model;