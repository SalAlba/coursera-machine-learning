function plotDataModel(error_train, error_val)

m = length(error_train);

plot(1:m, error_train, 1:m, error_val);
title('Learning curve for linear regression');
legend('Train', 'Cross Validation');
xlabel('Number of training examples');
ylabel('Error');
axis([0 13 0 150]);

end