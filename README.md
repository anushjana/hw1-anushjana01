# Instructions

⚠️

* **Do not modify the `tests` folder! It may cause errors and hence a decrease in your grading.**

* **Do not modify function names, variable names, or important code logic if you're not instructed to do so in README or
  directly in code comments or docstring. It may cost you a decrease in your grade.**

* **Add, modify or delete only those code parts which are either instructed by README, comment, or docstring,
  or they are intentionally left blank (added placeholders `pass`, `...`, `None`) for you to fill.**

* **Problem specific instructions are done via comments, read them, also pay more attention on TODO comments**

*Functions usually return values, if the instructor wants you to print something, it will directly be instructed in the
code, either by comment, or placeholder, or there will be a `print` function in the code*

### Recommendation

It's recommended to use different virtual environments for different projects (HWs).
You can always find list of [required libraries](requirements.txt) in your HW directory.
You may install requirements before solving your projects, it will increase your chances for having working code:

```shell
pip install -r requirements.txt
```

Happy coding 🧑‍💻.

# Problem statements

[normal_equation](normal_equation.py)  **20 points**


Problem :

You are given a dataset with multiple features,
your task is to implement linear regression using [normal equation](http://mlwiki.org/index.php/Normal_Equation).
You are given feature matrix and target vector,
you have to return the weight vector.

**Note**
Linear regression is in form `Y = w * X`, you don't have bias term.

Requirements:

* You have to use only numpy library to implement this function.
* Using other libraries will be considered wrong solution
* Using pure python solution, `lists`, `for` loop etc, won't give you full grade, though your hard work will be
  appreciated
* You have to use the normal equation to compute the weight vector w.

[linear regression](linear_regression.py)  **80 points**


Read comments in the code carefully, main instructions there.
Finish implementation in `CustomLinearRegression` class.
You should implement linear regression using gradient descent.
You should add L2 regularization your implementation.
You remember that linear regression without regularization is special case of L2 regularization, right?

You have free functions to implement:

* `init_weights` - **20 points**
* `fit` - **40 points**
* `predict` - **15 points**

* add custom plot - **5 points**

You have instructions for each of the function in function docstring.

Requirements:

* You have to use only numpy library to implement functions.
