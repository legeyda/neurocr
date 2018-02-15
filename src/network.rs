
//extern crate num;
extern crate rulinalg;

extern crate num;




use self::rulinalg::matrix::{Matrix, BaseMatrix};
use self::rulinalg::vector::{Vector};
use self::num::{Float, One, Zero};
use self::num::cast::NumCast;
use std::ops::Fn;
use std::ops::{Add, Mul, AddAssign};
use std::vec::Vec;


pub type float = f64;

//fn f(x: f64) -> Box<Float> {
//	FromPrimitive::from_f64(x).unwrap()
//}


#[inline]
fn exp<T: Float>(x: T) -> T {
	x.exp()
}

fn sigmoid<T: Float>(x: T) -> T {
	T::one() / (T::one() + exp(-x))
}

#[test]
fn test_sigmoid() {
	//assert_eq!(1.0, sigmoid(1.0));
}

fn sigmoid_prime<T: Float>(x: T) -> T {
	let sigma = sigmoid(x);
	sigma * (sigma + One::one())
}


struct Sample<T: Float> {
	input: Vector<T>,
	expected: Vector<T>
}

pub struct Layer<T> where T: Float {
	biases: Vector<T>,
	weights: Matrix<T>
}

impl<T> Layer<T> where T: Float {
	fn new(num_inputs: usize, num_neurons: usize) -> Layer<T> {
		Layer { 
			biases:  Vector::zeros(num_neurons), 
			weights: Matrix::zeros(num_neurons, num_inputs)
		}
	}
}



struct Network<T> where T: Float {
	layers: Vec<Layer<T>>
}

impl<T> Network<T> where T: Float + 'static {
	
	fn new(input_size: usize, layer_sizes: &Vec<usize>) -> Network<T> {
		let mut result = Network {
			layers: Vec::new()
		};
		let mut previous_size = input_size;
		for size in layer_sizes {
			result.layers.push(Layer::new(previous_size, *size));
		}
		result
	}
	
	/// for each neuron in layer, weighted sum of its inputs
	fn layer_weighted_sums(&self, layer: &Layer<T>, input: &Vector<T>) -> Vector<T> {
		&(layer.biases) + Vector::new((&(layer.weights)*input).into_vec())
	}
	
	/// apply activation function (sigmoid) to weighted sums of layer
	fn layer_eval(&self, weighted_sums: &Vector<T>) -> Vector<T> {
		weighted_sums.clone().apply(&sigmoid)
	}

	// evaluate weighted sums and outputs of each layer
	fn eval(&self, input: &Vector<T>) -> Vector<T> {
		let mut iter = self.layers.iter();
		let mut value;
		match iter.next() {
	        Some(layer) => {
	        	value = self.layer_eval(&self.layer_weighted_sums(layer, input))
	        },
	        None => panic!("not a single layer in network"),
	    }
		loop {
		    match iter.next() {
		        Some(layer) => {
			        value = self.layer_eval(&self.layer_weighted_sums(layer, &value))
		        },
		        None => break,
		    }
		}
		value
	}
	
	/// evaluate weighted sums and outputs of each layer
	fn eval_layers(&self, input: &Vector<T>) -> (Vec<Vector<T>>, Vec<Vector<T>>) {
		let n = self.layers.len();
		let mut zs:      Vec<Vector<T>> = Vec::with_capacity(n);
		let mut outputs: Vec<Vector<T>> = Vec::with_capacity(n);
		
		let mut iter = self.layers.iter();
		match iter.next() {
	        Some(layer) => {
	        	zs.push(self.layer_weighted_sums(layer, input));
	        	outputs.push(self.layer_eval(&zs[zs.len()-1]));
	        },
	        None => panic!("not a single layer in network"),
	    }
		loop {
		    match iter.next() {
		        Some(layer) => {
					let len=zs.len();
					let temp =  self.layer_weighted_sums(layer, &zs[len-1]);
		        	zs.push(temp);
		        	outputs.push(self.layer_eval(&zs[zs.len()-1]));
		        },
		        None => break,
		    }
		}
		(zs, outputs)
	}


	/// calculate gradient of layer
	/// @param inputs: current input for layer (either data or previous layer output)
	/// @param df_over_dz: partial gradient of output with respect to weighted input of this layer
	fn layer_grad(&self, inputs: &Vector<T>, df_over_dz: &Vector<T>) -> Layer<T> {
		//let 
		Layer {
			biases:  df_over_dz.clone(),
			weights: Matrix::new(df_over_dz.size(), 1, df_over_dz.data().clone()) * Matrix::new(1, inputs.size(), inputs.data().clone())
		}
	}
	
	
	fn sample_grad(&self, input: &Vector<T>, expected_output: &Vector<T>) -> Network<T> {
		let n = self.layers.len();
		let (zs, outputs) = self.eval_layers(input);
		let mut result: Vec<Layer<T>> = Vec::with_capacity(n);

		// for each neuron, derivative of squared error with respect to error itself
		let mut der = (&outputs[n-1] - expected_output).elemul(&zs[0].clone().apply(&sigmoid_prime));
		for i in (n-1)..1 {
			result.insert(0, self.layer_grad(&outputs[i-1], &der));
			// todo optimize convertions between matrix&vector
			der = Vector::new((self.layers[i].weights.transpose()*Matrix::new(der.size(), 1, der.data().clone())).data().clone());
		}
		result.insert(0, self.layer_grad(input, &der));
		
		return Network { layers: result };
	}

	fn batch_grad(&self, test_data: &Vec<Sample<T>>) -> Network<T> {
		let mut iter = test_data.iter();
		let mut result;
		match iter.next() {
	        Some(sample) => {
	        	result = self.sample_grad(&sample.input, &sample.expected);
	        },
	        None => panic!("not a single data in batch"),
	    }
		loop {
		    match iter.next() {
		        Some(sample) => {
			        result = result + &self.sample_grad(&sample.input, &sample.expected);
		        },
		        None => break,
		    }
		}
		result
	}

	fn batch_mse(&self, data: &Vec<Sample<T>>) -> T {
		let one = One::one();
		let mut iter = data.iter();
		let mut sum: T;
		let mut len: T = Zero::zero();
		match iter.next() {
	        Some(datum) => {
	        	sum = (&datum.expected - self.eval(&datum.input)).apply(&(|x| x*x)).sum();
				len = len + one;

	        },
	        None => panic!("not a single data in batch"),
	    }
		loop {
		    match iter.next() {
		        Some(datum) => {
			        sum = sum + (&datum.expected - self.eval(&datum.input)).apply(&(|x| x*x)).sum();
					len = len + one;
		        },
		        None => break,
		    }
		}
		sum/len
	}



	/// evaluate batch of samples and return rate of correct classification attempts
	fn asses_categorizing(&self, data: &Vec<Sample<T>>) -> f32 {
		let mut guessed = 0;
		for datum in data {
			guessed += if datum.expected.argmax().0 == self.eval(&datum.input).argmax().0 { 1 } else { 0 };
		}
		return (guessed as f32) / (data.len() as f32);
	}

	fn refine(&mut self, data: &Vec<Sample<T>>, step: T) {
		let grad = self.batch_grad(data);
		let new  = (self as &Network<T>) + &(grad*(-step));
		*self = new;
	}


}



impl<'rhs, T> Add<&'rhs Network<T>> for Network<T> where T: Float {
	type Output = Network<T>;
	fn add(mut self, rhs: &Network<T>) -> Self::Output {
		if self.layers.len() != rhs.layers.len() {
			panic!("add: network dimensions do not match");
		}
		for i in 0..self.layers.len() {
			self.layers[i] = &self.layers[i] + &rhs.layers[i];
		}
		self
	}
}

impl<'lhs, 'rhs, T> Add<&'rhs Network<T>> for &'lhs Network<T> where T: Float {
	type Output = Network<T>;
	fn add(self, rhs: &Network<T>) -> Self::Output {
		if self.layers.len() != rhs.layers.len() {
			panic!("add: network dimensions do not match");
		}
		let mut layers = Vec::with_capacity(self.layers.len());
		for i in 0..self.layers.len() {
			layers.push(&self.layers[i] + &rhs.layers[i]);
		}
		Network {
			layers: layers
		}
	}
}

impl <T> Mul<T> for Network<T> where T: Float {
	type Output = Network<T>;
	fn mul(mut self, multiplier: T) -> Self::Output {
		for i in 0..self.layers.len() {
			self.layers[i] = &self.layers[i] * multiplier;
		}
		self
	}
}





impl <T> Add<Layer<T>> for Layer<T> where T: Float {
	type Output = Layer<T>;
	fn add(mut self, rhs: Layer<T>) -> Self::Output {
		if self.biases.size() != rhs.biases.size() 
		|| self.weights.cols() != rhs.weights.cols()
		|| self.weights.rows() != rhs.weights.rows() {
			panic!("add: layer dimensions do not match")
		}
		self.biases  = self.biases  + rhs.biases;
		self.weights = self.weights + rhs.weights;
		self
	}
}

impl <'lhs, 'rhs, T> Add<&'rhs Layer<T>> for &'lhs Layer<T> where T: Float {
	type Output = Layer<T>;
	fn add(self, rhs: &Layer<T>) -> Self::Output {
		if self.biases.size() != rhs.biases.size() 
		|| self.weights.cols() != rhs.weights.cols()
		|| self.weights.rows() != rhs.weights.rows() {
			panic!("add: layer dimensions do not match")
		}
		Layer {
			biases:  &self.biases  + &rhs.biases,
			weights: &self.weights + &rhs.weights
		}
	}
}

impl <'a, T> AddAssign<&'a Layer<T>> for Layer<T> where T: Float {
	fn add_assign(&mut self, rhs: &Layer<T>) {
		if self.biases.size() != rhs.biases.size() 
		|| self.weights.cols() != rhs.weights.cols()
		|| self.weights.rows() != rhs.weights.rows() {
			panic!("add: layer dimensions do not match")
		}
		self.biases  = &self.biases  + &rhs.biases;
		self.weights = &self.weights + &rhs.weights;
	}
}

impl <T> Mul<T> for Layer<T> where T: Float {
	type Output = Layer<T>;
	fn mul(mut self, multiplier: T) -> Self::Output {
		self.biases  = self.biases  * multiplier;
		self.weights = self.weights * multiplier;
		self
	}
}

impl<'lhs, T> Mul<T> for &'lhs Layer<T> where T: Float {
	type Output = Layer<T>;
	fn mul(self, multiplier: T) -> Self::Output {
		Layer {
			biases:  &self.biases  * multiplier, 
			weights: &self.weights * multiplier
		}
	}
}