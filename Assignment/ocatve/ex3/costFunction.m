function [J, grad] = costFunction(theta, X, y)

%%
m = length(y);
J = 0;
grad = zeros(size(theta));

  
%%
prediction = sigmoid(X * theta);
k = (-y .* log(prediction)) - ((1 - y) .* log(1 - prediction));
J = sum(k) / m;



%%
error = prediction - y;
k = X' * error;
grad = k / m;


endfunction
