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

<p>~The list can be homogeneous or heterogeneous.</p>
<p>~Element-wise operation is not possible on the list.</p>
<p>~Python list is by default 1-dimensional. But we can create an N-Dimensional list. But then too it will be 1 D list storing another 1D list</p>
<p>~Elements of a list need not be contiguous in memory.</p>

<h3>Data Type</h3>

<p>Every ndarray has an associated data type (dtype) object. This data type object (dtype) informs us about the layout of the array. This means it gives us information about :</p>

<p>~ Type of the data (integer, float, Python object etc.)</p>
<p>~ Size of the data (number of bytes)</p>
<p>~ Byte order of the data (little-endian or big-endian)</p>
<p>If the data type is a sub-array, what is its shape and data type.
The values of a ndarray are stored in a buffer which can be thought of as a contiguous block of memory bytes. So how these bytes will be interpreted is given by the dtype object.
Every Numpy array is a table of elements (usually numbers), all of the same type, indexed by a tuple of positive integers. Every ndarray has an associated data type (dtype) object.
This data type object (dtype) provides information about the layout of the array. The vaues of an ndarray are stored in a buffer which can be thought of as a contiguous block of memory bytes which can be interpreted by the dtype object. Numpy provides a large set of numeric datatypes that can be used to construct arrays.
At the time of Array creation, Numpy tries to guess a datatype, but functions that construct arrays usually also include an optional argument to explicitly specify the datatype.</p>
 

# Python Program to create a data type object
<pre>import numpy as np
 
np.int16 is converted into a data type object.
print(np.dtype(np.int16))
Run on IDE
Output:int16</pre>
<pre>#Python Program to create a data type object 
#containing a 32 bit big-endian integer
import numpy as np
 
# i4 represents integer of size 4 byte
# > represents big-endian byte ordering and
# < represents little-endian encoding.
# dt is a dtype object
dt = np.dtype('>i4')
 
print("Byte order is:",dt.byteorder)
 
print("Size is:",dt.itemsize)
 
print("Data type is:",dt.name)</pre>

<h2>Data type Object (dtype) in NumPy</h2>

<img align="left" alt="datatype(numpy)" width="1000px" src="https://user-images.githubusercontent.com/90051406/198038223-556418b7-6edd-4cc8-ba86-0fe1895bb283.png" />
<br><br>
<h1>Creating Numpy Array</h1>
Array creation using List
<br>
Array creation using array functions 
<br>
<h2>Array creation using numpy methods </h2>

Methods for array creation in Numpy
<br>
<table><tr><th>FUNCTION</th><th>	DESCRIPTION</th></tr>
<tr><td>empty()</td><td>	Return a new array of given shape and type, without initializing entries</td></tr>
<tr><td>empty_like()</td><td>	Return a new array with the same shape and type as a given array</td></tr>
<tr><td>eye()</td><td>	Return a 2-D array with ones on the diagonal and zeros elsewhere.</td></tr>
<tr><td>identity()	</td><td>Return the identity array</td></tr>
<tr><td>ones()</td><td>	Return a new array of given shape and type, filled with ones</td></tr>
<tr><td>ones_like()</td><td>	Return an array of ones with the same shape and type as a given array</td></tr>
<tr><td>zeros()</td><td>	Return a new array of given shape and type, filled with zeros</td></tr>
<tr><td>zeros_like()</td><td>	Return an array of zeros with the same shape and type as a given array</td></tr>
<tr><td>full_like()	</td><td>Return a full array with the same shape and type as a given array.</td></tr>
<tr><td>array()</td><td>	Create an array</td></tr>
<tr><td>asarray()</td><td>	Convert the input to an array</td></tr>
<tr><td>asanyarray()</td><td>	Convert the input to an ndarray, but pass ndarray subclasses through</td></tr>
<tr><td>ascontiguousarray()	</td><td>Return a contiguous array in memory (C order)</td></tr>
<tr><td>asmatrix()</td><td>	Interpret the input as a matrix</td></tr>
<tr><td>copy()</td><td>	Return an array copy of the given object</td></tr>
<tr><td>frombuffer()</td><td>	Interpret a buffer as a 1-dimensional array</td></tr>
<tr><td>fromfile()</td><td>	Construct an array from data in a text or binary file</td></tr>
<tr><td>fromfunction()</td><td>	Construct an array by executing a function over each coordinate</td></tr>
<tr><td>fromiter()	</td><td>Create a new 1-dimensional array from an iterable object</td></tr>
<tr><td>fromstring()</td><td>	A new 1-D array initialized from text data in a string</td></tr>
<tr><td>loadtxt()</td><td>	Load data from a text file</td></tr>
<tr><td>arange()</td><td>	Return evenly spaced values within a given interval</td></tr>
<tr><td>linspace()</td><td>	Return evenly spaced numbers over a specified interval</td></tr>
<tr><td>logspace()</td><td>	Return numbers spaced evenly on a log scale</td></tr>
<tr><td>geomspace()</td><td>	Return numbers spaced evenly on a log scale (a geometric progression)</td></tr>
<tr><td>meshgrid()</td><td>	Return coordinate matrices from coordinate vectors</td></tr>
<tr><td>mgrid()</td><td>	nd_grid instance which returns a dense multi-dimensional “meshgrid</td></tr>
<tr><td>ogrid()</td><td>	nd_grid instance which returns an open multi-dimensional “meshgrid</td></tr>
<tr><td>diag()</td><td>	Extract a diagonal or construct a diagonal array</td></tr>
<tr><td>diagflat()	</td><td>Create a two-dimensional array with the flattened input as a diagonal</td></tr>
<tr><td>tri()</td><td>	An array with ones at and below the given diagonal and zeros elsewhere</td></tr>
<tr><td>tril()</td><td>	Lower triangle of an array</td></tr>
<tr><td>triu()</td><td>	Upper triangle of an array</td></tr>
<tr><td>vander()	</td><td>Generate a Vandermonde matrix</td></tr>
<tr><td>mat()	</td><td>Interpret the input as a matrix</td></tr>
<tr><td>bmat()</td><td>	Build a matrix object from a string, nested sequence, or array</td></tr></table>
 
fromiter()is useful for creating non-numeric sequence type array however it can create any type of array. Here we will convert a string into a NumPy array of characters.


<h3>How to generate 2-D Gaussian array using NumPy?</h3>
numpy.meshgrid() It is used to create a rectangular grid out of two given one-dimensional arrays representing the Cartesian indexing or Matrix indexing. <br>
<pre>Syntax: numpy.meshgrid(*xi, copy=True, sparse=False, indexing=’xy’)</pre>
<b>numpy.linspace()</b>– returns number spaces evenly w.r.t interval.<br>
<pre>syntax: numpy.linspace(start, stop, num = 50, endpoint = True, retstep = False, dtype = None)</pre>
<b>numpy.exp()</b>– this mathematical function helps the user to calculate the exponential of all the elements in the input array.<br>
<i>(Go and see this code)</i>
<h1>How to create a vector in Python using NumPy</h1>
Vector are built from components, which are ordinary numbers. We can think of a vector as a list of numbers, and vector algebra as operations performed on the numbers in the list. In other words vector is the numpy 1-D array.<br>

In order to create a vector, we use np.array method. 

<pre>Syntax : np.array(list)
Argument : It take 1-D list it can be 1 row and n columns or n rows and 1 column
Return : It returns vector which is numpy.ndarray</pre>
<b>Creating a Vector</b> 
<p>In this example we will create a horizontal vector and a vertical vector </p>
<p>Note: We can create vector with other method as well which return 1-D numpy array for example np.arange(10), np.zeros((4, 1)) gives 1-D array, but most appropriate way is using np.array with the 1-D list.</p>
 Creating a Vector 
In this example we will create a horizontal vector and a vertical vector <br>
<pre>Vector-Scalar Multiplication                   Vector Dot Product             Basic Arithmetic operation: </pre>

</h2>Numpy fromrecords() method<h2>
With the help of numpy.core.fromrecords() method, we can create the record array by using the list of individual records by using numpy.core.fromrecords() method.
<pre>
Syntax : numpy.core.fromrecords([(tup1), (tup2)], metadata)
Return : Return the record of an array.</pre>

<h1>NumPy Array Manipulation</h1>
<h2>copy and view</h2>
<h4>How to Copy NumPy array into another array?</h4>
![Group 3](https://user-images.githubusercontent.com/90051406/197821558-8cd1f675-d6ce-490f-b450-b3bb9104c8e5.png)

<p>Numpy array ‘org_array‘ is copied to another array ‘copy_array‘ using np.copy () function
the copy is physically stored at another location and view has the same memory location as the original array.</p>
<p>No Copy: Normal assignments do not make the copy of an array object. Instead, it uses the exact same id of the original array to access it. Further, any changes in either get reflected in the other.</p>
<p>View: This is also known as Shallow Copy. The view is just a view of the original array and view does not own the data. When we make changes to the view it affects the original array, and when changes are made to the original array it affects the view.</p>

<p>Copy: This is also known as Deep Copy. The copy is completely a new array and copy owns the data. When we make changes to the copy it does not affect the original array, and when changes are made to the original array it does not affect the copy.</p>

<pre>Example: (making a copy and changing original array)</pre>
<h3>Array Owning it’s Data:</h3>
<p>To check whether array own it’s data in view and copy we can use the fact that every NumPy array has the attribute base that returns None if the array owns the data. Else, the base attribute refers to the original object.</p>

<h3>How to swap columns of a given NumPy array?</h3>

Insert a new axis within a NumPy array
NumPy provides us with two different built-in functions to increase the dimension of an array i.e.,
 

1D array will become 2D array<br>
2D array will become 3D array<br>
3D array will become 4D array<br>
4D array will become 5D array<br>
1.numpy.newaxis()The first method is to use numpy.newaxis object. This object is equivalent to use None as a parameter while declaring the array. The trick is to use the numpy.newaxis object as a parameter at the index location in which you want to add the new axis.<br>
2.numpy.expand_dims()



<h1>numpy.hstack() in Python</h1>
<p>numpy.hstack() function is used to stack the sequence of input arrays horizontally (i.e. column wise) to make a single array.</p>
<pre>
Syntax : numpy.hstack(tup)

Parameters :
tup : [sequence of ndarrays] Tuple containing arrays to be stacked. The arrays must have the same shape along all but the second axis.

Return : [stacked ndarray] The stacked array of the input arrays.
</pre>
<h1>numpy.vstack() in python</h1>
<p>numpy.vstack() function is used to stack the sequence of input arrays vertically to make a single array.</p>
<pre>
Syntax : numpy.vstack(tup)

Parameters :
tup : [sequence of ndarrays] Tuple containing arrays to be stacked. The arrays must have the same shape along all but the first axis.

Return : [stacked ndarray] The stacked array of the input arrays.
</pre>
<h1>Joining NumPy Array</h1>
NumPy provides various functions to combine arrays. In this article, we will discuss some of the major ones.
<br>
numpy.concatenate<br>
numpy.stack<br>
numpy.block-numpy.block is used to create nd-arrays from nested blocks of lists.<br>
<pre>
Syntax:

numpy.block(arrays)</pre>
<b>Method 1:</b> Using numpy.concatenate()

<p>The concatenate function in NumPy joins two or more arrays along a specified axis. </p>
<pre>
Syntax:numpy.concatenate((array1, array2, ...), axis=0)</pre>
The stack() function of NumPy joins two or more arrays along a new axis.

<h1>Combining a one and a two-dimensional NumPy Array</h1>

<p>Sometimes we need to combine 1-D and 2-D arrays and display their elements. Numpy has a function named as numpy.nditer(), which provides this facility.</p>

<pre>Syntax: numpy.nditer(op, flags=None, op_flags=None, op_dtypes=None, order=’K’, casting=’safe’, op_axes=None, itershape=None, buffersize=0)</pre>

<h1>Numpy np.ma.concatenate() method</h1>

<p>With the help of np.ma.concatenate() method, we can concatenate two arrays with the help of np.ma.concatenate() method.</p>

<pre>Syntax : np.ma.concatenate([list1, list2])
Return : Return the array after concatenation.</pre>

<h1>Numpy dstack() method</h1>
<p>With the help of numpy.dstack() method, we can get the combined array index by index and store like a stack by using numpy.dstack() method.</p>
<pre>Syntax : numpy.dstack((array1, array2))
Return : Return combined array index by index.</pre>
<h2>How to compare two NumPy arrays?</h2>
<p>here we will be focusing on the comparison done using NumPy on arrays. Comparing two NumPy arrays determines whether they are equivalent by checking if every element at each corresponding index is the same. </p>

<b>Method 1:</b> We generally use the == operator to compare two NumPy arrays to generate a new array object. Call ndarray.all() with the new array object as ndarray to return True if the two NumPy arrays are equivalent. 
<b>Method 2:</b> We can also use greater than, less than and equal to operators to compare. To understand, have a look at the code below.
<pre>
Syntax : numpy.greater(x1, x2[, out])
Syntax : numpy.greater_equal(x1, x2[, out])
Syntax : numpy.less(x1, x2[, out])
Syntax : numpy.less_equal(x1, x2[, out])</pre>
<b>Method 3:</b> Using array_equal() 
This array_equal() function checks if two arrays have the same elements and same shape.
<pre>
Syntax:

numpy.array_equal(arr1, arr2) 
Parameters:

arr1    : [array_like]Input array or object whose elements, we need to test.
arr2    : [array_like]Input array or object whose elements, we need to test.
Return Type: True, two arrays have the same elements and same shape.; otherwise False</pre>
<h1>Find unique rows in a NumPy array</h1><br>
we will discuss how to find unique rows in a NumPy array. To find unique rows in a NumPy array we are using numpy.unique() function of NumPy library.
<pre>
Syntax of np.unique() in Python
Syntax: numpy.unique()
Parameter:
ar: array
return_index: Bool, if True return the indices of the input array
return_inverse: Bool, if True return the indices of the input array
return_counts: Bool, if True return the number of times each unique item appeared in the input array
axis: int or none, defines the axis to operate on</pre>

<h>Python | Numpy np.unique() method</h1>
With the help of np.unique() method, we can get the unique values from an array given as parameter in np.unique() method.<br>
<pre>Syntax : np.unique(Array)
Return : Return the unique of an array.</pre>

<h1>Operations on NumPy Array</h1>
<h3>Numpy – Binary Operations</h3>
<h3>Numpy – Mathematical Function</h3>
<h3>Numpy – String Operations</h3>
<h1>Matrix Manipulation</h1>
<img align="left" alt="datatype(numpy)" width="1000px" src="https://user-images.githubusercontent.com/90051406/197936373-99e26a96-760d-4669-9213-14fd086e63fa.png" />
<br><br>
<h1>Indexing NumPy Array</h1>

<h4>Why do we need NumPy ?</h4>

<p>A question arises that why do we need NumPy when python lists are already there. The answer to it is we cannot perform operations on all the elements of two list directly. For example we cannot multiply two lists directly we will have to do it element wise. This is where the role of NumPy comes into play.</p>

<h2>Types of Indexing</h2>

There are two types of indexing :<br>

<b>1. Basic Slicing and indexing :<b> <p>Consider the syntax x[obj] where x is the array and obj is the index. Slice object is the index in case of basic slicing.</p> Basic slicing occurs when obj is :<br>

a slice object that is of the form start : stop : step<br>
an integer<br>
or a tuple of slice objects and integers<br>
All arrays generated by basic slicing are always view of the original array.<br>
 
<b>2. Advanced indexing :</b> Advanced indexing is triggered when obj is :<br> 

an ndarray of type integer or Boolean<br>
or a tuple with at least one sequence object
is a non tuple sequence object
<p>Advanced indexing returns a copy of data rather than a view of it. Advanced indexing is of two types integer and Boolean.</p>

Purely integer indexing : When integers are used for indexing. Each element of first dimension is paired with the element of the second dimension. So the index of the elements in this case are (0,0),(1,0),(2,1) and the corresponding elements are selected.
Boolean Indexing 
This indexing has some boolean expression as the index. Those elements are returned which satisfy that Boolean expression. It is used for filtering the desired element values.

The <h1>numpy.compress()</h1> function returns selected slices of an array along mentioned axis, that satisfies an axis.<br>
<pre>
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
Return :Copy of array with elements of input_array,
that satisfies condition and along given axis</pre>

<h1>numpy.tril_indices()</h1> function return the indices for the lower-triangle of an (n, m) array.<br>

<pre>Syntax : numpy.tril_indices(n, k = 0, m = None)
Parameters :
n : [int] The row dimension of the arrays for which the returned indices will be valid.
k : [int, optional] Diagonal offset.
m : [int, optional] The column dimension of the arrays for which the returned arrays will be valid. By default m is taken equal to n.
Return : [tuple of arrays] The indices for the triangle. The returned tuple contains two arrays, each with the indices along one dimension of the array.</pre>

EigenVectors. In the below examples, we have used numpy.linalg.eig() to find eigenvalues and eigenvectors for the given square array.<br> 

<p[re>Syntax: numpy.linalg.eig()
Parameter: An square array.
Return: It will return two values first is eigenvalues and second is eigenvectors.</pre>

<h1>Numpy | Sorting, Searching and Counting</h1>

numpy.sort() : This function returns a sorted copy of an array.<br>
numpy.argsort() : This function returns the indices that would sort an array.<br>
numpy.lexsort() : This function returns an indirect stable sort using a sequence of keys.<br>
 
<table><tr><th>FUNCTION</th><th>	DESCRIPTION</th></tr>
<tr><td>numpy.ndarray.sort()</td><td>	Sort an array, in-place.</td></tr>
<tr><td>numpy.msort()</td><td>	Return a copy of an array sorted along the first axis.</td></tr>
<tr><td>numpy.sort_complex()	</td><td>Sort a complex array using the real part first, then the imaginary part.</td></tr>
<tr><td>numpy.partition()</td><td>	Return a partitioned copy of an array.</td></tr>
<tr><td>numpy.argpartition()</td><td>	Perform an indirect partition along the given axis using the algorithm specified by the kind keyword.</td></tr></table>
<h1>Searching</h1>
<i>Searching is an operation or a technique that helps finds the place of a given element or value in the list. Any search is said to be successful or unsuccessful depending upon whether the element that is being searched is found or not. In Numpy, we can perform various searching operations using the various functions that are provided in the library like argmax, argmin, nanaargmax etc.</i>
<table>
<tr><td>numpy.argmax() : This function returns indices of the max element of the array in a particular axis.</td></tr>
<tr><td>numpy.nanargmax() : This function returns indices of the max element of the array in a particular axis ignoring NaNs.The results cannot be trusted if a slice contains only NaNs and Infs.</td></tr>
<tr><td>numpy.argmin() : This function returns the indices of the minimum values along an axis.</td></tr></table>
</table>
<table><tr><th>FUNCTION</th><th>	DESCRIPTION</th></tr>
<tr><td>numpy.nanargmin()	</td><td>Return the indices of the minimum values in the specified axis ignoring NaNs.</td></tr>
<tr><td>numpy.argwhere()</td><td>	Find the indices of array elements that are non-zero, grouped by element.</td></tr>
<tr><td>numpy.nonzero()</td><td>	Return the indices of the elements that are non-zero.</td></tr>
<tr><td>numpy.flatnonzero()</td><td>	Return indices that are non-zero in the flattened version of a.</td></tr>
<tr><td>numpy.where()</td><td>	Return elements chosen from x or y depending on condition.</td></tr>
<tr><td>numpy.searchsorted()</td><td>	Find indices where elements should be inserted to maintain order.</td></tr>
<tr><td>numpy.extract()</td><td>	Return the elements of an array that satisfy some condition.</td></tr></table>
<h1>Counting</h1>
numpy.count_nonzero() : Counts the number of non-zero values in the array .

<h1>numpy.sort_complex() in Python</h1>
<p>numpy.sort_complex() function is used to sort a complex array.It sorts the array by using the real part first, then the imaginary part.</p>

<pre>Syntax : numpy.sort_complex(arr)
Parameters :
arr : [array_like] Input array.
Return : [complex ndarray] A sorted complex array.</pre>

<h1>Create your own universal function in NumPy</h1>
<p>Universal functions in Numpy are simple mathematical functions. It is just a term that we gave to mathematical functions in the Numpy library. Numpy provides various universal functions that cover a wide variety of operations. However, we can create our own universal function in Python. To create your own universal function in NumPy, we have to apply some steps given below:</p>

Define the function as usually using the def keyword.<br>
Add this function to numpy library using frompyfunc() method.
Use this function using numpy.<br>
frompyfunc() method function allows to create an arbitrary Python function as Numpy ufunc (universal function). <br>
This method takes the following arguments :<br>
<pre>
Parameters:
function – the name of the function that you create.
inputs – the number of input arguments (arrays) that function takes.
outputs – the number of output (arrays) that function produces.</pre>
<p>Note: For more information, refer to numpy.frompyfunc() in Python.</p>

<pre>
Example :

Create function with name fxn that takes one value and also return one value.
The inputs are array elements one by one.
The outputs are modified array elements using some logic.</pre>

<h2>Some of the basic universal functions in Numpy are-</h2>
 

<h3>Trigonometric functions:</h3>
<p>These functions work on radians, so angles need to be converted to radians by multiplying by pi/180. Only then we can call trigonometric functions. They take an array as input arguments. It includes functions like-</p>

<table><tr><th>FUNCTION</th><th>	DESCRIPTION</th></tr>
<tr><td>sin, cos, tan</td><td>		compute sine, cosine and tangent of angles</td></tr>
<tr><td>arcsin, arccos, arctan</td><td>		calculate inverse sine, cosine and tangent</td></tr>
<tr><td>hypot</td><td>		calculate hypotenuse of given right triangle</td></tr>
<tr><td>sinh, cosh, tanh</td><td>		compute hyperbolic sine, cosine and tangent</td></tr>
<tr><td>arcsinh, arccosh, arctanh	</td><td>	compute inverse hyperbolic sine, cosine and tangent</td></tr>
<tr><td>deg2rad	</td><td>	convert degree into radians</td></tr>
<tr><td>rad2deg	</td><td>	convert radians into degree</td></tr></table>
<h3>Statistical functions:</h3>
These functions are used to calculate mean, median, variance, minimum of array elements. It includes functions like-<br>
<table><tr><th>FUNCTION</th><th>	DESCRIPTION</th></tr>
<tr><td>amin, amax</td><td>	returns minimum or maximum of an array or along an axis
<tr><td>ptp	</td><td>returns range of values (maximum-minimum) of an array or along an axis
<tr><td>percentile(a, p, axis)</td><td>	calculate pth percentile of array or along specified axis
<tr><td>median	</td><td>compute median of data along specified axis
<tr><td>mean</td><td>	compute mean of data along specified axis
<tr><td>std</td><td>	compute standard deviation of data along specified axis
<tr><td>var</td><td>	compute variance of data along specified axis
<tr><td>average	</td><td>compute average of data along specified axis</td></tr></table>
 
<h1>Bit-twiddling functions:</h1>
<p>These functions accept integer values as input arguments and perform bitwise operations on binary representations of those integers. It include functions like-</p>
 <table><tr><th>FUNCTION</th><th>	DESCRIPTION</th></tr>
<tr><td>bitwise_and</td><td>	performs bitwise and operation on two array elements</td></tr>
<tr><td>bitwies_or</td><td>	performs bitwise or operation on two array elements</td></tr>
<tr><td>bitwise_xor</td><td>	performs bitwise xor operation on two array elements</td></tr>
<tr><td>invert</td><td>	performs bitwise inversion of an array elements</td></tr>
<tr><td>left_shift</td><td>	shift the bits of elements to left</td></tr>
<tr><td>right_shift</td><td>	shift the bits of elements to left</td></tr></table>

<h1>String Operations</h1>
This module is used to perform vectorized string operations for arrays of dtype numpy.string_ or numpy.unicode_. All of them are based on the standard string functions in Python’s built-in library.<br>
<b><i>String Operations –</i></b><br>
<b>numpy.lower() : </b>This function returns the lowercase string from the given string. It converts all uppercase characters to lowercase. If no uppercase characters exist, it returns the original string.<br>
<b>numpy.join() :</b> This function is a string method and returns a string in which the elements of sequence have been joined by str separator.<br>

 

<table><tr><th>FUNCTION</th><th>	DESCRIPTION</th></tr>
<tr><td>numpy.strip()</td><td>	It is used to remove all the leading and trailing spaces from a string.</td></tr>
<tr><td>numpy.capitalize()</td><td>	It converts the first character of a string to capital (uppercase) letter. If the string has its first character as capital, then it returns the original string.</td></tr>
<tr><td>numpy.center()</td><td>	It creates and returns a new string which is padded with the specified character..</td></tr>
<tr><td>numpy.decode()</td><td>	It is used to convert from one encoding scheme, in which argument string is encoded to the desired encoding scheme.</td></tr>
<tr><td>numpy.encode()</td><td>	Returns the string in the encoded form</td></tr>
<tr><td>numpy.ljust()</td><td>	Return an array with the elements of a left-justified in a string of length width.</td></tr>
<tr><td>numpy.rjust()</td><td>	For each element in a, return a copy with the leading characters removed.</td></tr>
<tr><td>numpy.strip()</td><td>	For each element in a, return a copy with the leading and trailing characters removed.</td></tr>
<tr><td>numpy.lstrip()</td><td>	Convert angles from degrees to radians.</td></tr>
<tr><td>numpy.rstrip()</td><td>	For each element in a, return a copy with the trailing characters removed.</td></tr>
<tr><td>numpy.partition()</td><td>	Partition each element in a around sep.</td></tr>
<tr><td>numpy.rpartition</td><td>	Partition (split) each element around the right-most separator.</td></tr>
<tr><td>numpy.rsplit()</td><td>	For each element in a, return a list of the words in the string, using sep as the delimiter string.</td></tr>
<tr><td>numpy.title()</td><td>	It is used to convert the first character in each word to Uppercase and remaining characters to Lowercase in string and returns new string.</td></tr>
<tr><td>numpy.upper()</td><td>	Returns the uppercased string from the given string. It converts all lowercase characters to uppercase.If no lowercase characters exist, it returns the original string.</td></tr></table>

<b><i>String Information –</i></b><br>
numpy.count() : This function returns the number of occurrences of a substring in the given string.<br>
numpy.rfind() : This function returns the highest index of the substring if found in given string. If not found then it returns -1.<br>
numpy.isnumeric() : This function returns “True” if all characters in the string are numeric characters, Otherwise, It returns “False”.<br>

<table><tr><th>FUNCTION</th><th>	DESCRIPTION</th></tr>
<tr><td>numpy.find()</td><td>	It returns the lowest index of the substring if it is found in given string. If its is not found then it returns -1.</td></tr>
<tr><td>numpy.index()</td><td>	It returns the position of the first occurrence of substring in a string</td></tr>
<tr><td>numpy.isalpha()</td><td>	It returns “True” if all characters in the string are alphabets, Otherwise, It returns “False”.</td></tr>
<tr><td>numpy.isdecimal()</td><td>	</td><td>It returns true if all characters in a string are decimal. If all characters are not decimal then it returns false.</td></tr>
<tr><td>numpy.isdigit()</td><td>	It returns “True” if all characters in the string are digits, Otherwise, It returns “False”.</td></tr>
<tr><td>numpy.islower()</td><td>	It returns “True” if all characters in the string are lowercase, Otherwise, It returns “False”.</td></tr>
<tr><td>numpy.isspace()</td><td>	Returns true for each element if there are only whitespace characters in the string and there is at least one character, false otherwise.</td></tr>
<tr><td>numpy.istitle()</td><td>	Returns true for each element if the element is a titlecased string and there is at least one character, false otherwise.</td></tr>
<tr><td>numpy.isupper()</td><td>	Returns true for each element if all cased characters in the string are uppercase and there is at least one character, false otherwise.</td></tr>
<tr><td>numpy.rindex()</td><td>	Returns the highest index of the substring inside the string if substring is found. Otherwise it raises an exception.</td></tr>
<tr><td>numpy.startswith()</td><td>	Returns True if a string starts with the given prefix otherwise returns False.</td></tr></table>

<b><i>String Comparison –</i></b><br>
numpy.equal(): This function checks for string1 == string2 elementwise.<br>
numpy.not_equal(): This function checks whether two string is unequal or not.<br>
numpy.greater(): This function checks whether string1 is greater than string2 or not.<br>
