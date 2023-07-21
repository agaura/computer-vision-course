addpath layers;

l = [init_layer('conv',struct('filter_size',9,'filter_depth',1,'num_filters',10))
    init_layer('relu',[])
	init_layer('pool',struct('filter_size',2,'stride',2))
    init_layer('relu',[])
    init_layer('conv',struct('filter_size',12,'filter_depth',10,'num_filters',10))
    init_layer('flatten',struct('num_dims',4))
    init_layer('softmax',[])];

% conv layer: takes input [32 32 1], returns [24 24 6]
% pool layer: takes input [24 24 6], returns [12 12 6]
% conv layer3: takes input [12 12 16], returns [1 10]

model = init_model(l,[32 32 1],10,true);

% Uncomment this to get the resized inputs
[~,~,~,n1] = size(train_data);
[~,~,~,n2] = size(test_data);
input_train = zeros(32,32,1,n1);
input_test = zeros(32,32,1,n2);
for i = 1:n1
    disp(i);
    input_train(:,:,1,i) = normalize(padarray(train_data(:,:,1,i),[2 2],0,'both'),'range',[-0.1,1.175]);
end
for i = 1:n2
    disp(i);
    input_test(:,:,1,i) = normalize(padarray(test_data(:,:,1,i),[2 2],0,'both'),'range',[-0.1,1.175]);
end

z = clock;
[trainmodel, trainloss, train_losses, test_losses] = train(model,input_train(:,:,:,:),train_label(:,1),struct(),5,input_test,test_label);
disp("Elapsed Time:");
disp(etime(clock,z));
[output,~] = inference(trainmodel,input_test(:,:,:,:));
hyper_params = struct('num_dims',2);
[loss, ~] = loss_crossentropy(output, test_label, hyper_params, 1);
[~, idx] = max(output, [], 1);
accuracy = sum(transpose(idx) == test_label(:,1))/10000;
disp("Loss:");
disp(loss);
disp("Accuracy:");
disp(accuracy);