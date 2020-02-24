function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    printf('x %s \n',mat2str(size(X)));
    printf('t %s \n',mat2str(size(theta)));
    printf('y %s \n',mat2str(size(y)));

    fprintf('\nIter [%d]\n', iter);
    prediction = X * theta;

    printf('p %s \n',mat2str(size(prediction)));

    error = prediction- y;
    printf('e %s \n',mat2str(size(error)));

    k = sum(X' * error);
    printf('k %s \n',mat2str(size(k)));
    theta = theta - alpha/m * k;
    printf('t %s \n',mat2str(size(theta)));


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);


end

end