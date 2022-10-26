# Numpy
Introduction
<h4>✨ Array processing library in python.</h4>
<h4>✨ It provides a high-performance multidimensional array object, and tools for working with these arrays.</h4>
<h4>✨ Fundamental package for scientific computing with Python and used as an efficient multi-dimensional container of generic data. </h4>
 
<h5>features<pre>~Useful linear algebra, Fourier transform, and random number capabilities</pre>
         <pre>~powerful N-dimensional array object</pre>
</h5>
<h4>✨ It is a table of elements (usually numbers), all of the same type, indexed by a tuple of positive integers.</h4>
<h4>✨In NumPy dimensions are called axes. The number of axes is rank.NumPy’s array class is called ndarray. It is also known by the alias array.</h4>
   <pre>   Eg
    [[ 1, 2, 3], 
    [ 4, 2, 5]]     </pre>                                  
   <p> rank = 2 (as it is 2-dimensional or it has 2 axes)</p>
    <p>first dimension(axis) length = 2, second dimension has length = 3</p>
    <p>overall shape can be expressed as: (2, 3)</p>
    
    <h5>NumPy array: Python lists are a substitute for arrays but they fail to deliver the performance required while computing large sets of numerical data. To address this issue we use a python library called NumPy.
 NumPy stands for Numerical Python
 Numpy is not another programming language but a Python extension module. It provides fast and efficient operations on arrays of homogeneous data. 
 NumPy offers an array object called ndarray.</h5>
 
<h4> Why is Numpy so fast?</h4>
<p>Numpy arrays are written mostly in C language. Being written in C, the NumPy arrays are stored in contiguous memory locations which makes them accessible and easier to manipulate. This means that you can get the performance level of a C code with the ease of writing a python program.</p>
 <h3>Array creation Array indexing</h3>
<pre>installation- pip install numpy</pre>
<i><b>import - import numpy as np(or other alias u want to use)</b></i>
<h3>Numpy array from a list</h3>
<p>You can use the np alias to create ndarray of a list using the array() method.</p>

<pre>li = [1,2,3,4]
numpyArr = np.array(li)
or

numpyArr = np.array([1,2,3,4])</pre>
<p>The list is passed to the array() method which then returns a NumPy array with the same elements.</p>


<pre>initialize a NumPy array from a list.
li = [1, 2, 3, 4]
numpyArr = np.array(li)
print(numpyArr)</pre>

<h3>NumPy array from a tuple</h3>
You can make ndarray from a tuple using similar syntax.

<pre>tup = (1,2,3,4)
numpyArr = np.array(tup)
or

numpyArr = np.array((1,2,3,4))</pre>

<b>NumPy stands for Numerical Python. It is a Python library used for working with an array. In Python, we use the list for purpose of the array but it’s slow to process. NumPy array is a powerful N-dimensional array object and its use in linear algebra, Fourier transform, and random number capabilities. It provides an array object much faster than traditional Python lists.</b>

<h3>Types of Array:</h3>
<p><b>One Dimensional Array:</b> A one-dimensional array is a type of linear array.</p>
<b>Multi-Dimensional Array</b>
Data in multidimensional arrays are stored in tabular form.
<i>Anatomy of an array :</i>
 <p>Axis: The Axis of an array describes the order of the indexing into the array.</p>
 <p>Shape: The number of elements along with each axis. It is from a tuple.</p>
 <p>Rank: The rank of an array is simply the number of axes (or dimensions) it has.</p>
 <h2>Some different way of creating Numpy Array :</h2>
 
<pre><p>1. numpy.array(): The Numpy array object in Numpy is called ndarray. We can create ndarray using numpy.array() function.</p>
<b>Syntax: numpy.array(parameter)</b></pre>
<pre><p>2. numpy.fromiter(): The fromiter() function create a new one-dimensional array from an iterable object.</p>
<b>Syntax: numpy.fromiter(iterable, dtype, count=-1)</b></pre>
<pre><p>3. numpy.arange(): This is an inbuilt NumPy function that returns evenly spaced values within a given interval.</p>
<b>Syntax: numpy.arange([start, ]stop, [step, ]dtype=None)</b></pre>
<pre><p>4. numpy.linspace(): This function returns evenly spaced numbers over a specified between two limits. </b>
<b>Syntax: numpy.linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0)</b></pre>
<pre><p>5. numpy.empty(): This function create a new array of given shape and type, without initializing value.
<b>Syntax: numpy.empty(shape, dtype=float, order=’C’)</b></pre>
<pre><p>6. numpy.ones(): This function is used to get a new array of given shape and type, filled with ones(1).</p>
<b>Syntax: numpy.ones(shape, dtype=None, order=’C’)</b></pre>
<pre><p>7. numpy.zeros(): This function is used to get a new array of given shape and type, filled with zeros(0). </p>
<b>Syntax: numpy.ones(shape, dtype=None)</b></pre>

<h2>List vs numpy array</h2>
<p>Some important points about Python Lists:</p>

~The list can be homogeneous or heterogeneous.
~Element-wise operation is not possible on the list.
~Python list is by default 1-dimensional. But we can create an N-Dimensional list. But then too it will be 1 D list storing another 1D list
~Elements of a list need not be contiguous in memory.

Data Type

Every ndarray has an associated data type (dtype) object. This data type object (dtype) informs us about the layout of the array. This means it gives us information about :

Type of the data (integer, float, Python object etc.)
Size of the data (number of bytes)
Byte order of the data (little-endian or big-endian)
If the data type is a sub-array, what is its shape and data type.
The values of a ndarray are stored in a buffer which can be thought of as a contiguous block of memory bytes. So how these bytes will be interpreted is given by the dtype object.
Every Numpy array is a table of elements (usually numbers), all of the same type, indexed by a tuple of positive integers. Every ndarray has an associated data type (dtype) object.
This data type object (dtype) provides information about the layout of the array. The vaues of an ndarray are stored in a buffer which can be thought of as a contiguous block of memory bytes which can be interpreted by the dtype object. Numpy provides a large set of numeric datatypes that can be used to construct arrays.
At the time of Array creation, Numpy tries to guess a datatype, but functions that construct arrays usually also include an optional argument to explicitly specify the datatype.
 

# Python Program to create a data type object
import numpy as np
 
# np.int16 is converted into a data type object.
print(np.dtype(np.int16))
Run on IDE
Output:

int16
# Python Program to create a data type object 
# containing a 32 bit big-endian integer
import numpy as np
 
# i4 represents integer of size 4 byte
# > represents big-endian byte ordering and
# < represents little-endian encoding.
# dt is a dtype object
dt = np.dtype('>i4')
 
print("Byte order is:",dt.byteorder)
 
print("Size is:",dt.itemsize)
 
print("Data type is:",dt.name)

Data type Object (dtype) in NumPy
![datatype(numpy)](https://user-images.githubusercontent.com/90051406/197744880-e5a7ab58-0317-47c6-9752-3db39cb44b88.png)
<h1>Creating Numpy Array</h1>
Array creation using List
Array creation using array functions 
Array creation using numpy methods 

Methods for array creation in Numpy
FUNCTION	DESCRIPTION
empty()	Return a new array of given shape and type, without initializing entries
Syntax: numpy.empty(shape, dtype = float, order = ‘C’) 

empty_like()	Return a new array with the same shape and type as a given array
eye()	Return a 2-D array with ones on the diagonal and zeros elsewhere.
numpy.eye(R, C = None, k = 0, dtype = type <‘float’>) : –The eye tool returns a 2-D array with  1’s as the diagonal and  0’s elsewhere. The diagonal can be main, upper, or lower depending on the optional parameter k. A positive k is for the upper diagonal, a negative k is for the lower, and a  0 k (default) is for the main diagonal.

Parameters : 

R : Number of rows
C : [optional] Number of columns; By default M = N
k : [int, optional, 0 by default]
          Diagonal we require; k>0 means diagonal above main diagonal or vice versa.
dtype : [optional, float(by Default)] Data type of returned array.  
Returns : 

array of shape, R x C, an array where all elements 
are equal to zero, except for the k-th diagonal, 
whose values are equal to one.
identity()	Return the identity array
ones()	Return a new array of given shape and type, filled with ones
parameters, discussed below – 
 

shape : integer or sequence of integers
order  : C_contiguous or F_contiguous
         C-contiguous order in memory(last index varies the fastest)
         C order means that operating row-rise on the array will be slightly quicker
         FORTRAN-contiguous order in memory (first index varies the fastest).
         F order means that column-wise operations will be faster. 
dtype : [optional, float(byDefault)] Data type of returned array.  

ones_like()	Return an array of ones with the same shape and type as a given array
zeros()	Return a new array of given shape and type, filled with zeros
Syntax:

numpy.zeros(shape, dtype = None, order = 'C')
Parameters :
shape : integer or sequence of integers
order  : C_contiguous or F_contiguous
         C-contiguous order in memory(last index varies the fastest)
         C order means that operating row-rise on the array will be slightly quicker
         FORTRAN-contiguous order in memory (first index varies the fastest).
         F order means that column-wise operations will be faster. 
dtype : [optional, float(byDeafult)] Data type of returned array.  

Returns : ndarray of zeros having given shape, order and datatype.
The arange([start,] stop[, step,][, dtype]) : Returns an array with evenly spaced elements as per the interval. The interval mentioned is half-opened i.e. [Start, Stop) 

Parameters : 

start : [optional] start of interval range. By default start = 0
stop  : end of interval range
step  : [optional] step size of interval. By default step size = 1,  
For any output out, this is the distance between two adjacent values, out[i+1] - out[i]. 
dtype : type of output array
Return: 

Array of evenly spaced values.
Length of array being generated  = Ceil((Stop - Start) / Step)
zeros_like()	Return an array of zeros with the same shape and type as a given array
full_like()	Return a full array with the same shape and type as a given array.
array()	Create an array
asarray()	Convert the input to an array
asanyarray()	Convert the input to an ndarray, but pass ndarray subclasses through
ascontiguousarray()	Return a contiguous array in memory (C order)
asmatrix()	Interpret the input as a matrix
copy()	Return an array copy of the given object
frombuffer()	Interpret a buffer as a 1-dimensional array
fromfile()	Construct an array from data in a text or binary file
fromfunction()	Construct an array by executing a function over each coordinate
fromiter()	Create a new 1-dimensional array from an iterable object
fromstring()	A new 1-D array initialized from text data in a string
loadtxt()	Load data from a text file
arange()	Return evenly spaced values within a given interval
linspace()	Return evenly spaced numbers over a specified interval
Syntax : 
 

numpy.linspace(start,
               stop,
               num = 50,
               endpoint = True,
               retstep = False,
               dtype = None)
Parameters : 

-> start  : [optional] start of interval range. By default start = 0
-> stop   : end of interval range
-> restep : If True, return (samples, step). By default restep = False
-> num    : [int, optional] No. of samples to generate
-> dtype  : type of output array
Return : 
 

-> ndarray
-> step : [float, optional], if restep = True
Code 1 : Explaining linspace function 
 
logspace()	Return numbers spaced evenly on a log scale
geomspace()	Return numbers spaced evenly on a log scale (a geometric progression)
meshgrid()	Return coordinate matrices from coordinate vectors
mgrid()	nd_grid instance which returns a dense multi-dimensional “meshgrid
ogrid()	nd_grid instance which returns an open multi-dimensional “meshgrid
diag()	Extract a diagonal or construct a diagonal array
diagflat()	Create a two-dimensional array with the flattened input as a diagonal
tri()	An array with ones at and below the given diagonal and zeros elsewhere
tril()	Lower triangle of an array
triu()	Upper triangle of an array
vander()	Generate a Vandermonde matrix
mat()	Interpret the input as a matrix
bmat()	Build a matrix object from a string, nested sequence, or array


![Group 2](https://user-images.githubusercontent.com/90051406/197808873-d33b02b5-734a-4874-aca2-db0e97a70974.png)

fromiter()is useful for creating non-numeric sequence type array however it can create any type of array. Here we will convert a string into a NumPy array of characters.


How to generate 2-D Gaussian array using NumPy?
numpy.meshgrid() It is used to create a rectangular grid out of two given one-dimensional arrays representing the Cartesian indexing or Matrix indexing. 
Syntax: numpy.meshgrid(*xi, copy=True, sparse=False, indexing=’xy’)
numpy.linspace()– returns number spaces evenly w.r.t interval.
syntax: numpy.linspace(start, stop, num = 50, endpoint = True, retstep = False, dtype = None)
numpy.exp()– this mathematical function helps the user to calculate the exponential of all the elements in the input array.
(Go and see this code)
How to create a vector in Python using NumPy
Vector are built from components, which are ordinary numbers. We can think of a vector as a list of numbers, and vector algebra as operations performed on the numbers in the list. In other words vector is the numpy 1-D array.

In order to create a vector, we use np.array method. 

Syntax : np.array(list)
Argument : It take 1-D list it can be 1 row and n columns or n rows and 1 column
Return : It returns vector which is numpy.ndarray
Creating a Vector 
In this example we will create a horizontal vector and a vertical vector 
Note: We can create vector with other method as well which return 1-D numpy array for example np.arange(10), np.zeros((4, 1)) gives 1-D array, but most appropriate way is using np.array with the 1-D list.
 Creating a Vector 
In this example we will create a horizontal vector and a vertical vector 
Vector-Scalar Multiplication                   Vector Dot Product             Basic Arithmetic operation: 

Numpy fromrecords() method
With the help of numpy.core.fromrecords() method, we can create the record array by using the list of individual records by using numpy.core.fromrecords() method.

Syntax : numpy.core.fromrecords([(tup1), (tup2)], metadata)

Return : Return the record of an array.

<h1>NumPy Array Manipulation</h1>
copy and view
How to Copy NumPy array into another array?
![Group 3](https://user-images.githubusercontent.com/90051406/197821558-8cd1f675-d6ce-490f-b450-b3bb9104c8e5.png)

Numpy array ‘org_array‘ is copied to another array ‘copy_array‘ using np.copy () function
the copy is physically stored at another location and view has the same memory location as the original array.
No Copy: Normal assignments do not make the copy of an array object. Instead, it uses the exact same id of the original array to access it. Further, any changes in either get reflected in the other.
View: This is also known as Shallow Copy. The view is just a view of the original array and view does not own the data. When we make changes to the view it affects the original array, and when changes are made to the original array it affects the view.

Copy: This is also known as Deep Copy. The copy is completely a new array and copy owns the data. When we make changes to the copy it does not affect the original array, and when changes are made to the original array it does not affect the copy.

Example: (making a copy and changing original array)
Array Owning it’s Data:
To check whether array own it’s data in view and copy we can use the fact that every NumPy array has the attribute base that returns None if the array owns the data. Else, the base attribute refers to the original object.

How to swap columns of a given NumPy array?

Insert a new axis within a NumPy array
NumPy provides us with two different built-in functions to increase the dimension of an array i.e.,
 

1D array will become 2D array
2D array will become 3D array
3D array will become 4D array
4D array will become 5D array
1.numpy.newaxis()The first method is to use numpy.newaxis object. This object is equivalent to use None as a parameter while declaring the array. The trick is to use the numpy.newaxis object as a parameter at the index location in which you want to add the new axis.
2.numpy.expand_dims()



numpy.hstack() in Python
numpy.hstack() function is used to stack the sequence of input arrays horizontally (i.e. column wise) to make a single array.

Syntax : numpy.hstack(tup)

Parameters :
tup : [sequence of ndarrays] Tuple containing arrays to be stacked. The arrays must have the same shape along all but the second axis.

Return : [stacked ndarray] The stacked array of the input arrays.
numpy.vstack() in python
numpy.vstack() function is used to stack the sequence of input arrays vertically to make a single array.

Syntax : numpy.vstack(tup)

Parameters :
tup : [sequence of ndarrays] Tuple containing arrays to be stacked. The arrays must have the same shape along all but the first axis.

Return : [stacked ndarray] The stacked array of the input arrays.

Joining NumPy Array
NumPy provides various functions to combine arrays. In this article, we will discuss some of the major ones.

numpy.concatenate
numpy.stack
numpy.block-numpy.block is used to create nd-arrays from nested blocks of lists.

Syntax:

numpy.block(arrays)
Method 1: Using numpy.concatenate()

The concatenate function in NumPy joins two or more arrays along a specified axis. 

Syntax:


numpy.concatenate((array1, array2, ...), axis=0)
The stack() function of NumPy joins two or more arrays along a new axis.

Combining a one and a two-dimensional NumPy Array
Difficulty Level : Medium
Last Updated : 01 Oct, 2020
Read
Discuss

Sometimes we need to combine 1-D and 2-D arrays and display their elements. Numpy has a function named as numpy.nditer(), which provides this facility.

Syntax: numpy.nditer(op, flags=None, op_flags=None, op_dtypes=None, order=’K’, casting=’safe’, op_axes=None, itershape=None, buffersize=0)

Numpy np.ma.concatenate() method
Difficulty Level : Medium
Last Updated : 03 Nov, 2019
Read
Discuss

With the help of np.ma.concatenate() method, we can concatenate two arrays with the help of np.ma.concatenate() method.

Syntax : np.ma.concatenate([list1, list2])
Return : Return the array after concatenation.

Numpy dstack() method
Last Updated : 19 Sep, 2019
Read
Discuss

With the help of numpy.dstack() method, we can get the combined array index by index and store like a stack by using numpy.dstack() method.

Syntax : numpy.dstack((array1, array2))

Return : Return combined array index by index.


How to compare two NumPy arrays?
Difficulty Level : Basic
Last Updated : 03 Jun, 2022
Read
Discuss

Here we will be focusing on the comparison done using NumPy on arrays. Comparing two NumPy arrays determines whether they are equivalent by checking if every element at each corresponding index is the same. 

Method 1: We generally use the == operator to compare two NumPy arrays to generate a new array object. Call ndarray.all() with the new array object as ndarray to return True if the two NumPy arrays are equivalent. 
Method 2: We can also use greater than, less than and equal to operators to compare. To understand, have a look at the code below.

Syntax : numpy.greater(x1, x2[, out])
Syntax : numpy.greater_equal(x1, x2[, out])
Syntax : numpy.less(x1, x2[, out])
Syntax : numpy.less_equal(x1, x2[, out])
Method 3: Using array_equal() 
This array_equal() function checks if two arrays have the same elements and same shape.

Syntax:

numpy.array_equal(arr1, arr2) 
Parameters:

arr1    : [array_like]Input array or object whose elements, we need to test.
arr2    : [array_like]Input array or object whose elements, we need to test.
Return Type: True, two arrays have the same elements and same shape.; otherwise False

Find unique rows in a NumPy array
Difficulty Level : Easy
Last Updated : 03 Oct, 2022
Read
Discuss

In this article, we will discuss how to find unique rows in a NumPy array. To find unique rows in a NumPy array we are using numpy.unique() function of NumPy library.

Syntax of np.unique() in Python
Syntax: numpy.unique()

Parameter:

ar: array
return_index: Bool, if True return the indices of the input array
return_inverse: Bool, if True return the indices of the input array
return_counts: Bool, if True return the number of times each unique item appeared in the input array
axis: int or none, defines the axis to operate on

Python | Numpy np.unique() method
Difficulty Level : Easy
Last Updated : 21 Nov, 2019
Read
Discuss

With the help of np.unique() method, we can get the unique values from an array given as parameter in np.unique() method.

Syntax : np.unique(Array)
Return : Return the unique of an array.

Operations on NumPy Array
Numpy – Binary Operations

Numpy – Mathematical Function
Numpy – String Operations
![Group 4](https://user-images.githubusercontent.com/90051406/197936373-99e26a96-760d-4669-9213-14fd086e63fa.png)
Indexing NumPy Array

Why do we need NumPy ?

A question arises that why do we need NumPy when python lists are already there. The answer to it is we cannot perform operations on all the elements of two list directly. For example we cannot multiply two lists directly we will have to do it element wise. This is where the role of NumPy comes into play.

Types of Indexing

There are two types of indexing :

1. Basic Slicing and indexing : Consider the syntax x[obj] where x is the array and obj is the index. Slice object is the index in case of basic slicing. Basic slicing occurs when obj is :

a slice object that is of the form start : stop : step
an integer
or a tuple of slice objects and integers
All arrays generated by basic slicing are always view of the original array.
 
2. Advanced indexing : Advanced indexing is triggered when obj is : 

an ndarray of type integer or Boolean
or a tuple with at least one sequence object
is a non tuple sequence object
Advanced indexing returns a copy of data rather than a view of it. Advanced indexing is of two types integer and Boolean.

Purely integer indexing : When integers are used for indexing. Each element of first dimension is paired with the element of the second dimension. So the index of the elements in this case are (0,0),(1,0),(2,1) and the corresponding elements are selected.
Boolean Indexing 
This indexing has some boolean expression as the index. Those elements are returned which satisfy that Boolean expression. It is used for filtering the desired element values.

The numpy.compress() function returns selected slices of an array along mentioned axis, that satisfies an axis.

Syntax: numpy.compress(condition, array, axis = None, out = None)
Parameters :

condition : [array_like]Condition on the basis of which user extract elements. 
      Applying condition on input_array, if we print condition, it will return an arra
      filled with either True or False. Array elements are extracted from the Indices having 
      True value.
array     : Input array. User apply conditions on input_array elements
axis      : [optional, int]Indicating which slice to select. 
         By Default, work on flattened array[1-D]
out       : [optional, ndarray]Output_array with elements of input_array, 
               that satisfies condition
Return :

Copy of array with elements of input_array,
that satisfies condition and along given axis

numpy.tril_indices() function return the indices for the lower-triangle of an (n, m) array.

Syntax : numpy.tril_indices(n, k = 0, m = None)
Parameters :
n : [int] The row dimension of the arrays for which the returned indices will be valid.
k : [int, optional] Diagonal offset.
m : [int, optional] The column dimension of the arrays for which the returned arrays will be valid. By default m is taken equal to n.
Return : [tuple of arrays] The indices for the triangle. The returned tuple contains two arrays, each with the indices along one dimension of the array.

EigenVectors. In the below examples, we have used numpy.linalg.eig() to find eigenvalues and eigenvectors for the given square array. 

Syntax: numpy.linalg.eig()


Parameter: An square array.

Return: It will return two values first is eigenvalues and second is eigenvectors.

Numpy | Sorting, Searching and Counting

numpy.sort() : This function returns a sorted copy of an array.
numpy.argsort() : This function returns the indices that would sort an array.

numpy.lexsort() : This function returns an indirect stable sort using a sequence of keys.
 

FUNCTION	DESCRIPTION
numpy.ndarray.sort()	Sort an array, in-place.
numpy.msort()	Return a copy of an array sorted along the first axis.
numpy.sort_complex()	Sort a complex array using the real part first, then the imaginary part.
numpy.partition()	Return a partitioned copy of an array.
numpy.argpartition()	Perform an indirect partition along the given axis using the algorithm specified by the kind keyword.
Searching
Searching is an operation or a technique that helps finds the place of a given element or value in the list. Any search is said to be successful or unsuccessful depending upon whether the element that is being searched is found or not. In Numpy, we can perform various searching operations using the various functions that are provided in the library like argmax, argmin, nanaargmax etc.

numpy.argmax() : This function returns indices of the max element of the array in a particular axis.
numpy.nanargmax() : This function returns indices of the max element of the array in a particular axis ignoring NaNs.The results cannot be trusted if a slice contains only NaNs and Infs.
numpy.argmin() : This function returns the indices of the minimum values along an axis.

FUNCTION	DESCRIPTION
numpy.nanargmin()	Return the indices of the minimum values in the specified axis ignoring NaNs.
numpy.argwhere()	Find the indices of array elements that are non-zero, grouped by element.
numpy.nonzero()	Return the indices of the elements that are non-zero.
numpy.flatnonzero()	Return indices that are non-zero in the flattened version of a.
numpy.where()	Return elements chosen from x or y depending on condition.
numpy.searchsorted()	Find indices where elements should be inserted to maintain order.
numpy.extract()	Return the elements of an array that satisfy some condition.
Counting
numpy.count_nonzero() : Counts the number of non-zero values in the array .

numpy.sort_complex() in Python
Last Updated : 24 Dec, 2018
Read
Discuss

numpy.sort_complex() function is used to sort a complex array.It sorts the array by using the real part first, then the imaginary part.

Syntax : numpy.sort_complex(arr)

Parameters :
arr : [array_like] Input array.

Return : [complex ndarray] A sorted complex array.
