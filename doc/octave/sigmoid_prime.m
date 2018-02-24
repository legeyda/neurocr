function retval=sigmoid_prime(x)
	sigma = sigmoid(x);
    retval = sigma*(1-sigma);
endfunction