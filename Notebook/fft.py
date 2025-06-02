import cmath

#inputs 
num = int(input())

int_list = [int(input()) for _ in range(2*num)]

def split_into_chunks(num, base = 10000):
    """
    Split a large integer into chunks using a given base.
    Returns coefficients in little-endian order (least significant digit first).
    
    Example: 123456789 with base 10000 becomes [6789, 2456, 1]
    """
    
    if num == 0:
        return [0]
    
    chunks = []
    
    while num > 0:
        chunks.append(num%base)
        num //= base 
        
    return chunks 

def fft(x):
    """
    Compute the Fast Fourier Transform using an iterative method
    (similar to NumPy's internal implementation)
    
    Args:
        x: Input sequence (list or array of complex numbers)
        
    Returns:
        FFT of x (list of complex numbers)
    """
    n = len(x)
    
    # Check if n is a power of 2
    if n & (n-1) != 0:
        raise ValueError("Size must be a power of 2")
    
    # Copy input to output array
    x = list(x)  # Make a copy to avoid modifying the input
    
    # Bit-reversal permutation
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j >= bit:
            j -= bit
            bit >>= 1
        j += bit
        
        if i < j:
            x[i], x[j] = x[j], x[i]
    
    # Compute FFT
    step = 2
    while step <= n:
        half_step = step // 2
        twiddle_step = n // step
        
        for i in range(0, n, step):
            for j in range(half_step):
                idx1 = i + j
                idx2 = i + j + half_step
                
                # Twiddle factor
                twiddle = cmath.exp(-2j * cmath.pi * j * twiddle_step / n)
                
                # Butterfly operation
                temp = x[idx2] * twiddle
                x[idx2] = x[idx1] - temp
                x[idx1] = x[idx1] + temp
                
        step *= 2
        
    return x

def ifft(x):
    """
    Compute the Inverse Fast Fourier Transform using an iterative method
    
    Args:
        x: Input sequence (list or array of complex numbers)
        
    Returns:
        IFFT of x (list of complex numbers)
    """
    n = len(x)
    
    # Take complex conjugate
    x_conj = [complex(val.real, -val.imag) for val in x]
    
    # Apply FFT
    y = fft(x_conj)
    
    # Take complex conjugate and divide by n
    return [complex(val.real/n, -val.imag/n) for val in y]


def mult_fft(a, b):
    """
    Multiply two polynomials using FFT.
    """
    # Get result size and power of 2 ceiling
    result_size = len(a) + len(b) - 1
    n = 1
    while n < result_size:
        n *= 2
        
    # Pad with zeros
    a_pad = a + [0] * (n - len(a))
    b_pad = b + [0] * (n - len(b))
    
    # Convert inputs to complex numbers to improve precision
    a_complex = [complex(val, 0) for val in a_pad]
    b_complex = [complex(val, 0) for val in b_pad]
    
    # Compute FFT
    a_fft = fft(a_complex)
    b_fft = fft(b_complex)
    
    # Element-wise multiplication
    c_fft = [a_fft[i] * b_fft[i] for i in range(n)]
    
    # Compute inverse FFT
    c = ifft(c_fft)
    
    # More aggressive rounding to handle floating point errors
    return [int(c[i].real + 0.5) for i in range(result_size)]


def process_carry(chunks, base=10000):
    """
    Process the carries for the result of polynomial multiplication.
    
    Args:
        chunks: List of coefficients from FFT multiplication
        base: Base used for chunking
    
    Returns:
        List of properly carried coefficients
    """
    
    carry = 0
    result = []
    
    for value in chunks:
        
        # add the carry to the current value
        value += carry
        
        #calculate the new carry and the number
        result.append(value % base)
        carry = value // base
        
    if carry > 0 :
        result.append(carry)
        
    return result

def reconstruct_int(chunks, base =10000):
    """
    Reconstruct an integer from its chunks.
    
    Args:
        chunks: List of integer chunks in little-endian order
        base: Base used for chunking
    
    Returns:
        The reconstructed integer
    """
    
    result = 0
    
    for i in range(len(chunks)-1,-1,-1):
        result = result*base + chunks[i]
    return result




for i in range(0,len(int_list),2):
    
    #construct the polynomial
    a = split_into_chunks(int_list[i])
    b = split_into_chunks(int_list[i+1])
    
    #multiply the polynomials
    c = mult_fft(a, b)
    
    #process the carry
    c_processed = process_carry(c)
    
    #reconstruct the integer
    result = reconstruct_int(c_processed)
    
    #output the result
    print(result)


