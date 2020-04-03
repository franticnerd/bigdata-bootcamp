# Python Tools for Data Analysis

## 1. Python Installation: Anaconda

**If you already have a python environment, ignore this session.**

[Anaconda](https://www.anaconda.com/) is a complete, [open source](https://docs.anaconda.com/anaconda/eula) data science package with a community of over 6 million users. It is easy to [download](https://www.anaconda.com/download/) and install, and it is supported on Linux, MacOS, and Windows ([source](https://opensource.com/article/18/4/getting-started-anaconda-python)).

We'll use [Miniconda](https://docs.conda.io/en/latest/miniconda.html) for minimal installation.

### Windows and macOS

1. Download the Python 3.7 distribution packages from [this website](https://docs.conda.io/en/latest/miniconda.html).
2. Install the package according to the instructions.
3. Start to use conda environment with *Anaconda Prompt* or other shells if you enabled this feature during installation.

### Linux with  terminal 

1. Start the terminal.

2. Switch to `~/Download/` with command `cd ~/Download/`. If the path does not exist, create one using `mkdir ~/Download/`.

3. Download the latest Linux Miniconda distribution using `wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh`.

4. Install by `bash Miniconda3-latest-Linux-x86_64.sh`.

5. Follow the prompts on the installer screens.

   If you are unsure about any setting, accept the defaults. You can change them later.

6. To make the changes take effect, close and then re-open your terminal window or use the command `source bashrc`.

7. If you are using *zsh* or other shells, make sure conda is initiated. To do this, switch back to bash and type the command `conda init <shell name>`.

### Check your installation

You can use the command `conda list` to check your conda installation. If the terminal returns a bunch of python packages, then your installation is success.

### Conda environment

With conda, you can create, export, list, remove, and update environments that have different versions of Python and/or packages installed in them. Switching or moving between environments is called activating the environment. You can also share an environment file.

This part is not necessary as you can directly use your *base* environment. However, for those who wants to know more, refer to [this website](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) for details and instructions.

## 2. Package Installation

If you are using anaconda, you can directly use the anaconda package manager. Otherwise, you can use other managers such as *pip*. We'll only demonstrate how to install packages with `conda` instructions.

To look for a specific package, you can visit [this website](https://anaconda.org/) and type the name of that package in the search box. For today's instruction, we need to install `numpy`, `matplotlib`,  `scikit-learn` and `pandas`. 

First, switch to your conda environment, then install those packages by typing these instructions one by one:

```bash
conda install -c conda-forge numpy
conda install -c conda-forge matplotlib
conda install -c conda-forge scikit-learn
conda install -c conda-forge pandas
```

The package manager will automatically install all the dependencies. So if you choose to install scikit-learn first, then you  don't have to install numpy manually as scikit-learn depends on numpy.

If you prefer a fancier and more powerful python shell, you can choose to install `ipython` and even `jupyter notebook`, which allows you to run your commands in your browser.

```bash
conda install -c conda-forge ipython
conda install jupyter
```

## 3. Basic Python Concepts

> A more comprehensive tutorial can be found at [this website](http://cs231n.github.io/python-numpy-tutorial/#python-basic). In this and the following sections we'll just introduce the surface due to time limitation.

First, in your terminal, type `python` or `ipython` or `jupyter notebook` to start a python shell. `ipython` or `jupyter notebook` is recommended.

### Variable definition, input and print

There's no type constraint for a variable, i.e., a variable can be of any type.

```python
a = 123
b = '123'
c = "1234"
print(a, b, c, type(a), type(b), type(c))
```

A variable can be overwritten by different types

```python
a = 123.456
print(type(a))
a = '123'
print(type(a))
```

Input some strings interactively:

```python
x = input('Input something: ')
print(x, type(x))
```

### List, tuple, set and dictionary

- **List** is a collection which is ordered and changeable. Allows duplicate members.
- **Tuple** is a collection which is ordered and unchangeable. Allows duplicate members.
- **Set** is a collection which is unordered and unindexed. No duplicate members.
- **Dictionary** is a collection which is unordered, changeable and indexed. No duplicate members.

```python
_list = [1, 2, 1.2, '1', '2', 1]  # this is a list
_tuple = (1, 2, 1.2, '1', '2', 1)  # this is a tuple
_set = {1, 2, 1.2, '1', '2', 1}  # this is a set
_dict = {  # this is a dict
    1: '111',
    2: '222',
    '1': 567,
    2.2: ['J', 'Q', 'K']
}
print(_list, '\n', _tuple, '\n', _set, '\n', _dict)
```

Access elements

```python
print(_list[0], _list[-2], _list[1: 3])
print(_tuple[1], _tuple[-2])
print(_set[0], _set[-1])  # This will throw an error
print(_dict[1], _dict['1'], _dict[2.2])
```

Shallow copy

```python
a = _list
a[0] = 888
print(a, '\n', _list)
```

### If else

```python
if 888 not in _dict.keys():
    _dict[888] = '???'
elif 999 not in _dict.keys():
    _dict[999] = '!@#$%'
else:
    _dict['qwert'] = 'poiuy'
```

### Loops

`for` loop:

```python
for x in _list:
    print(x)

for i in range(len(_list)):
    print(_list[i])
```

`while` loop:

```python
i = 0
while i != len(_list):
    print(_list[i])
    i += 1
```

### Function

Define a function:

```python
def my_func(x):
	x += 1
    print('in function: ', x)
    return x
```

Call a function

```python
t = 10
tt = my_func(t)
print(f'out of funciton, t: {t}, tt: {tt}')
```

## 4. Basic Numpy Usage

### Array creation

A numpy array is a grid of values, all of the same type, and is indexed by a tuple of nonnegative integers. The number of dimensions is the *rank* of the array; the *shape* of an array is a tuple of integers giving the size of the array along each dimension.

We can initialize numpy arrays from nested Python lists, and access elements using square brackets:

```python
import numpy as np

a = np.array([1, 2, 3])   # Create a rank 1 array
print(type(a), a.dtype)
print(a.shape)
print(a[1])

b = np.array([[1,2,3],[4,5,6]])    # Create a rank 2 array
print(b.shape)
print(b[0, 0], b[0, 1], b[1, 0])
```

Change the type of an array:

```python
print(a.dtype)
a = a.astype(np.float)
print(a.dtype)
```

Other array creation methods:

```python
a = np.zeros((2,2))   # Create an array of all zeros
print(a)
b = np.ones((1,2))    # Create an array of all ones
print(b)
c = np.full((2,2), 7, dtype=np.float32)  # Create a constant array
print(c)
d = np.eye(3)         # Create a 3x3 identity matrix
print(d)
e = np.random.random((3,3))  # Create an array filled with random values
print(e)
```

### Array indexing

Similar to Python lists, numpy arrays can be sliced. Since arrays may be multidimensional, you must specify a slice for each dimension of the array:

```python
a = np.arange(12).reshape(3, 4)		# Create a rank 1 array and reshape it to a 3x4 matrix
b = a[:2, 1:3]
print(a)
print(b)

# Shallow copy
b[0, 0] = 888
print(a)
```

You can mix integer indexing with slice indexing. However, doing so will yield an array of lower rank than the original array:

```python
row_r1 = a[1, :]    # Rank 1 view of the second row of a
row_r2 = a[1:2, :]  # Rank 2 view of the second row of a
print(row_r1, row_r1.shape)
print(row_r2, row_r2.shape)
```

You can also access element in the array via list:

```python
x = [0, 1, 2]
y = [3, 1, 0]
print(a[x, y])
```

Or via boolean array:

```python
b = a > 4
print(b)
print(a[b])
```

### Array math

Basic mathematical functions operate elementwise on arrays, and are available both as operator overloads and as functions in the numpy module:

```python
x = np.arange(1, 5, dtype=np.float).reshape(2, 2)
y = np.arange(5, 9, dtype=np.float).reshape(2, 2)
print(x)
print(y)

# Elementwise sum
print(x + y)
print(np.add(x, y))

# Elementwise difference
print(x - y)
print(np.subtract(x, y))

# Elementwise product
print(x * y)
print(np.multiply(x, y))

# Elementwise division
print(x / y)
print(np.divide(x, y))

# Elementwise square
print(x ** 2)
print(np.power(x, 2))

# Elementwise square root
print(x ** 0.5)
print(np.sqrt(x))
```

Matrix multiplication

```python
x = np.arange(1, 5, dtype=np.float).reshape(2, 2)
y = np.arange(5, 9, dtype=np.float).reshape(2, 2)
print(x)
print(y)

v = np.array([9, 10], dtype=np.float)
w = np.array([11, 12], dtype=np.float)

# Inner product
print(v.dot(w))
print(np.dot(v, w))
print(v @ w)

# Matrix / vector product
print(x.dot(v))
print(np.dot(x, v))
print(x @ v)

# Matrix / matrix product
print(x.dot(y))
print(np.dot(x, y))
print(x @ y)
```

**Attention:** `np.dot()` and `@` behaves differently when the rank of matrix is larger than or equal to 3.

Numpy provides many useful functions for performing computations on arrays such as  `sum`:

```python
print(np.sum(x))  # Compute sum of all elements; prints "10"
print(x.sum())  # same as above
print(np.sum(x, axis=0))  # Compute sum of each column; prints "[4 6]"
print(np.sum(x, axis=1))  # Compute sum of each row; prints "[3 7]"
```

To transpose a matrix, simply use the `T` attribute of an array object:

```python
print(x.T)

# Note that taking the transpose of a rank 1 array does nothing:
print(v)
print(v.T)
```

## 5. Using Matplotlib for visualization

```python
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib qt

# Compute the x and y coordinates for points on a sine curve
x = np.arange(0, 3 * np.pi, 0.1)
y = np.sin(x)

# Plot the points using matplotlib
plt.plot(x, y)
plt.show()  # You must call plt.show() to make graphics appear.
```

**Note: **if you are using jupyter notebook, you can use the command `%matplotlib inline` to make the graphics embedded in the editor or `%matplotlib qt` to make them pop out.

To plot multiple lines at once, and add a title, legend, and axis labels:

```python
x = np.arange(0, 3 * np.pi, 0.1)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Plot the points using matplotlib
plt.plot(x, y_sin)
plt.plot(x, y_cos)
plt.xlabel('x axis label')
plt.ylabel('y axis label')
plt.title('Sine and Cosine')
plt.legend(['Sine', 'Cosine'])
plt.show()
```

You can plot different things in the same figure using the `subplot` function. Here is an example:

```python
# Set up a subplot grid that has height 2 and width 1,
# and set the first such subplot as active.
plt.subplot(2, 1, 1)

# Make the first plot
plt.plot(x, y_sin)
plt.title('Sine')

# Set the second subplot as active, and make the second plot.
plt.subplot(2, 1, 2)
plt.plot(x, y_cos)
plt.title('Cosine')

# Show the figure.
plt.show()
```

## 6. Pandas and Scikit-Learn for Data Science

In this section, we will look at a data science example using pandas as data management tool and scikit-learn (sklearn) as algorithm implementation. This section is modified from [here](https://elitedatascience.com/python-machine-learning-tutorial-scikit-learn).

### Package import

```python
import numpy as np
import pandas as pd

# automatically split the data into training and test set
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# classifiers and regressors
from sklearn.ensemble import RandomForestRegressor
# Construct a Pipeline from the given estimators
from sklearn.pipeline import make_pipeline
# Exhaustive search over specified parameter values for an estimator.
from sklearn.model_selection import GridSearchCV

# Training objective and evaluation metrics
from sklearn.metrics import mean_squared_error, r2_score
# For model persistence
# you can use `from sklearn.externals import joblib` if your sklearn version is earlier than 0.23
import joblib
```

### Load data

You can download the [data](http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv) by clicking the link or using `wget`: `wget http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv` and move the file to your current folder. Then, load the `csv` data through `pandas`:

```python
data = pd.read_csv('winequality-red.csv', sep=';')
```

Or, you can directly load the data through URL.

```python
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')
```

You can also manipulate other data formats with `pandas`. A detailed document is [here](https://pandas.pydata.org/pandas-docs/stable/user_guide/io.html).

### Take a look of the loaded data

The data loaded is stored in the type of `pandas.core.frame.DataFrame`

To give a peak of the data, we can simply use

```python
print(data)
```

This will return a nice-looking preview of the elements in the DataFrame.

To view the name of the features of a DataFrame, one can use

```python
print(data.keys())
```

To access one column, i.e., all instances of a feature, e.g., `pH`, one can use

```python
# These will return the same result
print(data['pH'])
print(data.pH)
```

To access a row, you need the `DataFrame.iloc` attribute:

```python
print(data.iloc[10])
```

We can also easily print some summary statistics:

```python
print(data.describe())
```

### Split data

First, let's separate our target (y) feature from our input (X) features and divide the dataset into training and test sets using the `train_test_split` function:

```python
y = data.quality
X = data.drop('quality', axis=1)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y
)
```

Stratify your sample by the target variable will ensure your training set looks similar to your test set, making your evaluation metrics more reliable.

### Pre-processing

Standardization is the process of subtracting the means from each feature and then dividing by the feature standard deviations. It is a common requirement for machine learning tasks. Many algorithms assume that all features are centered around zero and have approximately the same variance.

```python
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# To prove the trainig and testing sets have (nearly) zero mean and one deviation
print(X_train_scaled.mean(axis=0))
print(X_train_scaled.std(axis=0))
print(X_test_scaled.mean(axis=0))
print(X_test_scaled.std(axis=0))
```

### Fit the model

If we do not need to fine-tune the hyperparameters, we can define a random forest regression model with the default hyperparameters and fit the model using

```python
regr = RandomForestRegressor()
regr.fit(X_train_scaled, y_train)
```

To examine the performance, we use the test set to calculate the scores

```python
pred = regr.predict(X_test_scaled)

print(r2_score(y_test, pred))
print(mean_squared_error(y_test, pred))
```

### Define the cross-validation pipeline

Fine-tuning hyperparameters is an important job in Machine Learning since a set of carefully chosen hyperparameters may greatly improve the performance of the model.

In practice, when we set up the cross-validation pipeline, we won't even need to manually fit the data. Instead, we'll simply declare the class object, like so:

```python
pipeline = make_pipeline(
    preprocessing.StandardScaler(),
    RandomForestRegressor(n_estimators=100)
)
```

To check the hyperparameters, we may use

```python
print pipeline.get_params()
```

or refer to the [official document](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html).

Now, let's declare the hyperparameters we want to tune through cross-validation.

```python
hyperparameters = {
    'randomforestregressor__max_features' : ['auto', 'sqrt', 'log2'],
    'randomforestregressor__max_depth': [None, 5, 3, 1]
}
```

Then, we can set a 10-fold cross validation as simple as

```python
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
```

Finally, we can automatically fine-tune the model using

```python
clf.fit(X_train, y_train)
```

After the model fitting, if we want to check the best hyperparameters, we can use

```python
print(clf.best_params_)
```

Same as before, we evaluate the fitted model on test set

```python
pred = clf.predict(X_test)

print(r2_score(y_test, pred))
print(mean_squared_error(y_test, pred))
```

### Saving and loading model

After training, we may want to save the trained model for future use. For this purpose, we can use

```python
joblib.dump(clf, 'rf_regressor.pkl')
```

When you want to load the model again, simply use this function:

```python
clf2 = joblib.load('rf_regressor.pkl')
 
# Predict data set using loaded model
clf2.predict(X_test)
```

---

Another more comprehensive example can be found [here](https://scikit-learn.org/stable/tutorial/statistical_inference/index.html).



