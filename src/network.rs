
//extern crate num;
extern crate rulinalg;

extern crate num;


use self::rulinalg::matrix::{Matrix};
use self::rulinalg::vector::{Vector};
use std::any::Any;
use self::num::{Float, One, FromPrimitive};
use std::ops::Fn;
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



pub struct Layer<T> where T: Float {
	biases: Vector<T>,
	weights: Matrix<T>
}



impl<T> Layer<T> where T: Float {
	fn new(num_neurons: usize, num_inputs: usize) -> Layer<T> {
		Layer { 
			biases:  Vector::zeros(num_neurons), 
			weights: Matrix::zeros(num_neurons, num_inputs)
		}
	}
}



//#[test]
//fn test_layer_eval() {
//	println!("hello from test!");
//	let layer = Layer {
//		biases: vector!(1.0, 2.0, 3.0),
//		weights: matrix!(
//				1.0, 2.0, 3.0;
//				4.0, 5.0, 6.0;
//				7.0, 8.0, 9.0)
//	};
//	let expected = vector!(69.0, 169.0, 269.0);
//	let input = vector!(10.0, 11.0, 12.0);
//	let actual = layer.eval(&input); 
//	assert_vector_eq!(
//		expected, actual
//		
//	);
//}




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
			result.layers.push(Layer::new(*size, previous_size));
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
	/// @param df_over_dz: partial gradient of output with respect to weighted input
	fn layer_grad(&self, inputs: &Vector<T>, df_over_dz: &Vector<T>) -> Layer<T> {
		Layer {
			biases:  df_over_dz.clone(),
			weights: Matrix::new(df_over_dz.size(), 1, df_over_dz.data().clone()) * Matrix::new(1, inputs.size(), inputs.data().clone())
		}
	}
		
	fn grad(&self, inputs: &Vector<T>, expected_output: &Vector<T>) -> Network<T> {
		let n = self.layers.len();
		let two = T::one() + T::one();
		let (zs, outputs) = self.eval_layers(inputs);
		let mut result: Vec<Layer<T>> = Vec::with_capacity(n);

		// for each neuron, derivative of squared error with respect to error itself
		let mut der = (&outputs[n-1] - expected_output)*two;

		for i in (n-2)..0 {
			result.push(self.layer_grad(&outputs[n-1], &der));
			der = der.elemul(&zs[n].clone().apply(&sigmoid_prime));			
		}
		result.push(self.layer_grad(inputs, &der));

		return Network { layers: result };
	}
	
	
}