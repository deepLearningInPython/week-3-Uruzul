import numpy as np
from tasks import * #imported since having difficulty seeing where issues arose, and saw this in grade tasks
# Follow the tasks below to practice basic Python concepts.
# Write your code in between the dashed lines.
# Don't import additional packages. Numpy suffices.


# Task 1: Compute Output Size for 1D Convolution
# Instructions:
# Write a function that takes two one-dimensional numpy arrays (input_array, kernel_array) as arguments.
# The function should return the length of the convolution output (assuming no padding and a stride of one).
# The output length can be computed as follows:
# (input_length - kernel_length + 1)

# Your code here:
def compute_output_size_1d(input_array, kernel_array):
    return len(input_array) - len(kernel_array) + 1


# Task 2: 1D Convolution
# Instructions:
# Write a function that takes a one-dimensional numpy array (input_array) and a one-dimensional kernel array (kernel_array)
# and returns their convolution (no padding, stride 1).

# Your code here:
def convolve_1d(input_array, kernel_array):
    output_length = compute_output_size_1d(input_array, kernel_array)
    output = np.zeros(output_length)
    kernel_reversed = kernel_array[::-1]
    for i in range(output_length):
        output[i] = np.sum(input_array[i:i+len(kernel_array)] * kernel_reversed)
    return output
    
# -----------------------------------------------
# Another tip: write test cases like this, so you can easily test your function.
input_array = np.array([1, 2, 3, 4, 5])
kernel_array = np.array([1, 0, -1])
print(convolve_1d(input_array, kernel_array))

# Task 3: Compute Output Size for 2D Convolution
# Instructions:
# Write a function that takes two two-dimensional numpy matrices (input_matrix, kernel_matrix) as arguments.
# The function should return a tuple with the dimensions of the convolution of both matrices.
# The dimensions of the output (assuming no padding and a stride of one) can be computed as follows:
# (input_height - kernel_height + 1, input_width - kernel_width + 1)

# Your code here:
def compute_output_size_2d(input_matrix, kernel_matrix):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel_matrix.shape
    output_height = input_height - kernel_height + 1
    output_width = input_width - kernel_width + 1
    return output_height, output_width
    
# -----------------------------------------------


# Task 4: 2D Convolution
# Instructions:
# Write a function that computes the convolution (no padding, stride 1) of two matrices (input_matrix, kernel_matrix).
# Your function will likely use lots of looping and you can reuse the functions you made above.

# Your code here:
    def convolute_2d(input_matrix, kernel_matrix):
        output_height, output_width = compute_output_size_2d(input_matrix, kernel_matrix)
        output = np.zeros((output_height, output_width))
        kernel_reversed = np.flip(kernel_matrix)
        for i in range(output_height):
            for j in range(output_width):
            region = input_matrix[i:i+kernel_matrix.shape[0], j:j+kernel_matrix.shape[1]]
            output[i, j] = np.sum(region * kernel_reversed)
    
        return output

# Tip: same tips as above, but you might need a nested loop here in order to
# define which parts of the input matrix need to be multiplied with the kernel matrix.

# -----------------------------------------------
