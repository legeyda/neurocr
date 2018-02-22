


//extern crate num;
extern crate rulinalg;

extern crate num;


extern crate log;

extern crate rand;


use self::rulinalg::matrix::{Matrix, BaseMatrix, BaseMatrixMut};
use self::rulinalg::vector::{Vector};
use self::num::{Float, One, Zero};
use std::ops::{Add, Mul, AddAssign, Fn};
use std::vec::Vec;
use std::fmt::Debug;
use self::rand::Rand;





#[inline]
fn exp<T: Float>(x: T) -> T {
	x.exp()
}

fn sigmoid<T: Float>(x: T) -> T {
	let one: T = T::one();
	one / (one + exp(-x))
}

#[cfg(test)]
//#[test]
fn test_sigmoid() {
	//assert_eq!(1.0, sigmoid(1.0));
}

fn sigmoid_prime<T: Float>(x: T) -> T {
	let sigma = sigmoid(x);
	sigma * (sigma + One::one())
}

#[derive(Debug)]
pub struct Sample<T: Float> {
	pub input: Vector<T>,
	pub expected: Vector<T>
}

#[derive(Debug)]
pub struct Layer<T> where T: Float {
	biases: Vector<T>,
	weights: Matrix<T>
}

fn random<T>() -> T where T: Float + Rand {
    let one = <T as One>::one();
	<T as Zero>::zero() + rand::random::<T>() * (one + one) - one
}

impl<T> Layer<T> where T: Float {
	fn zeros(num_inputs: usize, num_neurons: usize) -> Layer<T> {
		Layer { 
			biases:  Vector::zeros(num_neurons), 
			weights: Matrix::zeros(num_neurons, num_inputs)
		}
	}

	fn new(num_inputs: usize, num_neurons: usize, generator: fn() -> T) -> Layer<T>  {
		Layer { 
			biases:  Vector::zeros(num_neurons)            .apply(&(|ignored| generator())), 
			weights: Matrix::zeros(num_neurons, num_inputs).apply(&(|ignored| generator()))
		}
	}
	
}



#[derive(Debug)]
pub struct Network<T> where T: Float {
	layers: Vec<Layer<T>>
}

impl<T> Network<T> where T: Float + Debug + 'static {
	
	pub fn new(input_size: usize, layer_sizes: &[usize], generator: fn() -> T) -> Network<T> {
		let mut result = Network {
			layers: Vec::new()
		};
		let mut previous_size = input_size;
		for size in layer_sizes {
			result.layers.push(Layer::new(previous_size, *size, generator));
			previous_size=*size;
		}
		result
	}
	
	/// for each neuron in layer, weighted sum of its inputs
	fn layer_weighted_sums(&self, layer: &Layer<T>, input: &Vector<T>) -> Vector<T> {
		debug!("layer_weighted_sums: matrix cols is {:?}, vector size is {:?}", layer.weights.cols(), input.size());
		&(layer.biases) + Vector::new((&(layer.weights)*input).into_vec())
	}
	
	/// apply activation function (sigmoid) to weighted sums of layer
	fn layer_eval(&self, weighted_sums: &Vector<T>) -> Vector<T> {
		weighted_sums.clone().apply(&sigmoid)
	}

	// evaluate weighted sums and outputs of each layer
	pub fn eval(&self, input: &Vector<T>) -> Vector<T> {
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
					let temp =  self.layer_weighted_sums(layer, &zs[zs.len()-1]);
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
		debug!("sample_grad: # of layers is {:?}", n);
		let (zs, outputs) = self.eval_layers(input);
		let mut result: Vec<Layer<T>> = Vec::with_capacity(n);

		// for each neuron, derivative of squared error with respect to error itself
		let mut der = (&outputs[n-1] - expected_output).elemul(&zs[zs.len()-1].clone().apply(&sigmoid_prime));
		trace!("sample_grad: before first iteraton der is {:?}", der);
		// for all layers except the first, input is output of previous layer
		if n>1 {
			let mut i: usize = n-1;
			while i>0 { // todo rewrite with range(...).rev()
				debug!("sample_grad: processing layer # {:?}", i);
				trace!("sample_grad: layer gradient # {:?} is {:?}", i, self.layer_grad(&outputs[i-1], &der));
				result.insert(0, self.layer_grad(&outputs[i-1], &der));
				// todo optimize convertions between matrix&vector
				der = Vector::new((self.layers[i].weights.transpose()*Matrix::new(der.size(), 1, der.data().clone())).data().clone());
				trace!("sample_grad: after iteraton # {:?}: der is {:?}", i, der);
				i-=1;
			}
		}
		// for the first layer, input is network input
		debug!("sample_grad: processing first layer");
		trace!("sample_grad: first layer gradient is {:?}", self.layer_grad(input, &der));
		result.insert(0, self.layer_grad(input, &der));
		
		return Network { layers: result };
	}

	fn batch_grad(&self, test_data: &[Sample<T>]) -> Network<T> {
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

	pub fn refine(&mut self, data: &[Sample<T>], step: T) {
		trace!("refine: data is {:?}", data);	
		let grad = self.batch_grad(data);
		debug!("refine: self # of layers is {:?}", self.layers.len());
		trace!("refine: self is {:?}", self);
		debug!("refine: grad # of layers is {:?}", grad.layers.len());
		trace!("refine: grad is {:?}", grad);	
		*self = (self as &Network<T>) + &(grad*(-step))
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
			panic!("add: network dimensions do not match left {:?}, right {:?}", self.layers.len(), rhs.layers.len());
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



#[cfg(test)]
extern crate env_logger;


#[cfg(test)]
#[test]
fn test_network() {
	env_logger::init();

	/// let us teach network to differ horizontal from vertial lines on 2-by-2 image
	println!();
	println!("======== test_network ========");
	
	let data: Vec<Sample<f64>> = vec![
		// horizontal lines
		Sample {input: vector![1.0, 1.0, 0.0, 0.0], expected: vector![1.0, 0.0] },
		Sample {input: vector![0.0, 0.0, 1.0, 1.0], expected: vector![1.0, 0.0] },
		// vertical lines
		Sample {input: vector![1.0, 0.0, 1.0, 0.0], expected: vector![0.0, 1.0] },
		Sample {input: vector![0.0, 1.0, 0.0, 1.0], expected: vector![0.0, 1.0] },
	];
	
	let sizes: [usize;2] = [10, 2];
    let mut network: Network<f64> = Network::new(4, &sizes, || random());


	println!();
	println!("initial network is {:?}", network);
	



	println!();
	println!("output of first sample is {:?}", network.eval_layers(&data[0].input));
	
	println!();
	println!("gradient of first sample is {:?}",  network.sample_grad(&data[0].input, &data[0].expected));
	println!();
	println!("gradient of second sample is {:?}", network.sample_grad(&data[1].input, &data[1].expected));
	println!();
	println!("gradient of third sample is {:?}",  network.sample_grad(&data[2].input, &data[2].expected));
	println!();
	println!("gradient of fourth sample is {:?}", network.sample_grad(&data[3].input, &data[3].expected));


	println!();
	println!("gradient of test data batch is {:?}", network.batch_grad(&data));

	network.refine(&data[0..1], 0.5);

	println!();
	println!("network after refine is {:?}", network);

	println!();
	println!("output of first  sample is {:?}", network.eval_layers(&data[0].input));
	
	println!();
	println!("gradient of first sample is {:?}", network.sample_grad(&data[0].input, &data[0].expected));


	for i in 1..100 {
		network.refine(&data, 0.5);
		println!();
		println!("output of first sample is {:?}", network.eval_layers(&data[0].input));
			
		println!();
		println!("output of second sample is {:?}", network.eval_layers(&data[1].input));
		
		println!();
		println!("output of third  sample is {:?}", network.eval_layers(&data[2].input));
		
		println!();
		println!("output of fourth sample is {:?}", network.eval_layers(&data[3].input));
		
	}




	




}