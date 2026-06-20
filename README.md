# Linear Regression: From Scratch

## Intro

My name is Seif, a Data Science student, and I am trying to make my way into the tech industry as a Machine Learning Engineer, so I started a challenge: Building Machine Learning Algorithms from scratch. Using no built-in libraries, and most importantly no AI. Starting the challenge with Linear Regression.

## How Linear Regression works

Linear Regression is a model that makes predictions by simply calculating the weighted sum of the input features, adding a constant called the bias.

To train a linear regression model we first initialize the parameters randomly, then we measure how well does these parameters fit the data using a loss function, then we use an optimizer to update the parameters, and we keep on repeating these steps until we reach the best fit model with the least loss possible.

## How I did it

So let's walk through the code. The model has two main phases: fitting and predicting.

When you call fit, the model starts by assigning random values to the weights — one for each feature — and a bias term. Then for every epoch, it does three things: makes a prediction with the current parameters, measures how wrong it was using a loss function, and updates the parameters to do better next time.

The prediction itself is simple — multiply each input feature by its weight, sum everything up, and add the bias. That's literally all linear regression is doing under the hood.

For the loss functions, I gave the model three options: MSE, RMSE, and MAE. Getting the gradient formulas right required me to do a quick revision on some linear algebra — nothing too deep, just making sure I was comfortable with the math before writing it out. And to really understand the differences between the loss functions, I used the book Hands-On Machine Learning by Aurélien Géron, which explains it in a very practical way. The basic idea: MSE hits large errors harder, MAE treats all errors more equally and handles outliers better, and RMSE is just MSE but in the same unit as your output, which makes it easier to interpret.

The last piece is gradient descent. After measuring the loss, we calculate how much each parameter contributed to that error, and we shift them slightly in the direction that reduces it. We keep doing this every epoch until the model settles on the best fit it can find.

## Evaluation

To test the model, I ran it on the Kaggle House Prices dataset and compared it against Scikit-learn's LinearRegression. My model got an RMSE of 0.14, while Scikit-learn got 0.20. Which was a nice surprise — it means the gradient descent approach I built actually did better than the standard solution on this dataset.

## Outro

And that's it for this one. Building this from scratch taught me way more than just reading about it ever would — because you can't fake understanding when you have to write every single line yourself.

This is part of a challenge I set for myself: building machine learning algorithms from scratch, no libraries, no AI. Linear regression is just the beginning. Next is logistic regression, and I'll keep going from there.

If you're on a similar path, feel free to connect — links are in the description. See you in the next one.
