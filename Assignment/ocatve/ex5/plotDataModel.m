function plotDataModel(X, y, Xval, yval, Xtest, ytest, theta)

m = length(y);

plot(X, y, 'k+', 'MarkerSize', 10, 'LineWidth', 1.5);
%  Plot fit over the data
xlabel('Change in water level (x)');
ylabel('Water flowing out of the dam (y)');
hold on;
plot(Xval, yval, 'ro', 'MarkerSize', 10, 'LineWidth', 1.5);
plot(Xtest, ytest, 'bx', 'MarkerSize', 10, 'LineWidth', 1.5);
plot(X, [ones(m, 1) X]*theta, '--', 'LineWidth', 2);
legend('train', 'val', 'test', 'linear regression', 'location', 'northwest')
hold off;

end