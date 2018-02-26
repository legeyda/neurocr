


//extern crate num;
extern crate rulinalg;

extern crate num;


extern crate log;

extern crate rand;


use self::rulinalg::matrix::{Matrix, BaseMatrix, BaseMatrixMut};
use self::rulinalg::vector::{Vector};
use self::num::{Float, One, Zero, FromPrimitive};
use std::ops::{Add, Mul, Fn};
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
	assert_eq!(1.0, sigmoid(1.0));
}

fn sigmoid_prime<T: Float>(x: T) -> T {
	let one: T = One::one();
	let sigma: T = sigmoid(x);
	sigma * (one - sigma)
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
	<T as Zero>::zero() + rand::random::<T>() * (one + one + one + one + one + one) - one - one - one
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

//type float where float: Float + Debug + From<f64> + 'static;

#[derive(Debug)]
pub struct Network<T> where T: Float {
	layers: Vec<Layer<T>>
}

impl<T> Network<T> where T: Float + Debug + FromPrimitive + 'static {
	
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

	pub fn zeros(input_size: usize, layer_sizes: &[usize]) -> Network<T> {
		let mut result = Network {
			layers: Vec::new()
		};
		let mut previous_size = input_size;
		for size in layer_sizes {
			result.layers.push(Layer::zeros(previous_size, *size));
			previous_size=*size;
		}
		result
	}
	
	/// for each neuron in layer, weighted sum of its inputs
	fn layer_weighted_sums(&self, layer: &Layer<T>, input: &Vector<T>) -> Vector<T> {
		debug!("layer_weighted_sums: weights cols is {:?}, input size is {:?}", layer.weights.cols(), input.size());
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
					let temp =  self.layer_weighted_sums(layer, &outputs[outputs.len()-1]);
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
	/// @param df_over_dz: partial gradient of output with respect to weighted sum of this layer
	fn layer_grad(&self, inputs: &Vector<T>, df_over_dz: &Vector<T>) -> Layer<T> {
		//let 
		Layer {
			biases:  df_over_dz.clone(),
			weights: Matrix::new(df_over_dz.size(), 1, df_over_dz.data().clone()) * Matrix::new(1, inputs.size(), inputs.data().clone())
		}
	}
	
	
	fn sample_grad(&self, input: &Vector<T>, expected_output: &Vector<T>) -> Network<T> {
		let n = self.layers.len();
		trace!("sample_grad: # of layers is {:?}", n);
		let (zs, outputs) = self.eval_layers(input);
		let mut result: Vec<Layer<T>> = Vec::with_capacity(n);

		// for each neuron of latest layer, derivative of error function with respect to output
		let mut der = (&outputs[n-1] - expected_output);
		trace!("sample_grad: before first iteraton der is {:?}", der);

		// for all layers except the first, input is output of previous layer
		if n>1 {
			let mut i: usize = n-1;
			// loop for layers from last to second, excluding first
			while i>0 { // todo rewrite with range(...).rev()
				trace!("sample_grad: processing layer # {:?}", i);
				
				// for each neuron in this layer, derivative of error function with respect to its weighted sum
				der = der.elemul(&zs[i].clone().apply(&sigmoid_prime));
				trace!("sample_grad: applied sigmoid prime at # {:?} is {:?}", i, zs[i].clone().apply(&sigmoid_prime));
				trace!("sample_grad: G{:?} is {:?}", i, der);

				result.insert(0, self.layer_grad(&outputs[i-1], &der));
				trace!("sample_grad: layer #{:?}'s gradient is {:?}", i, result[0]);
				
				// for each neuron of previous layer, derivative of error function with respect to output of neuron
				der = Vector::new((
					self.layers[i].weights.transpose() 
					* Matrix::new(der.size(), 1, der.data().clone())
				).data().clone()); // todo optimize convertions between matrix&vector

				i-=1;
			}
		}
		// for the first layer, input is network input
		trace!("sample_grad: processing first layer");

		// for each neuron in first layer, derivative of error function with respect to its weighted sum
		der = der.elemul(&zs[0].clone().apply(&sigmoid_prime));
		trace!("sample_grad: G0 is {:?}", der);

		trace!("sample_grad: first layer's gradient is {:?}", self.layer_grad(input, &der));
		result.insert(0, self.layer_grad(input, &der));
		
		return Network { layers: result };
	}

	fn batch_grad(&self, test_data: &[Sample<T>]) -> Network<T> {
		let n = test_data.len() as f32;
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
		let one: T = One::one();
		result*(one/T::from_f32(n).unwrap())
	}

	pub fn refine(&mut self, data: &[Sample<T>], step: T) {
		trace!("refine: data is {:?}", data);	
		let grad = self.batch_grad(data);
		debug!("refine: self # of layers is {:?}", self.layers.len());
		trace!("refine: self is {:?}", self);
		debug!("refine: grad # of layers is {:?}", grad.layers.len());
		debug!("refine: grad is {:?}", grad);	
		*self = (self as &Network<T>) + &(grad*(-step))
	}

	pub fn squared_error(&self, sample: &Sample<T>) -> T {
		(self.eval(&sample.input) - &sample.expected).apply(&(|x| x*x)).sum()
	}

	pub fn mean_squared_error(&self, data: &[Sample<T>]) -> T {
		let n = data.len();
		let mut mse = self.squared_error(&data[0]);
		for i in 1..data.len() {
			mse = mse + self.squared_error(&data[i]);
		}
		mse / T::from_f32(data.len() as f32).unwrap()
	}

	pub fn classification_rate(&self, data: &[Sample<T>]) -> f32 {
		let mut guessed: usize = 0;
		for datum in data {
			guessed += if datum.expected.argmax().0 == self.eval(&datum.input).argmax().0 { 1 } else { 0 };
		}
		(guessed as f32) / (data.len() as f32)
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

// #[cfg(test)]
// fn quality<T>(outputs: &Vec<T>, ) -> T {
// 	let sample = Sample {input: vector![1.0, 1.0, 0.0, 0.0], expected: vector![1.0, 0.0] };



// }






//#[cfg(test)]
//extern crate simple_logger;

#[cfg(test)]
#[test]
fn init() {
	//env_logger::init();
	//simple_logger::init().unwrap()
}

#[cfg(test)]
fn get_mock_sample() -> Sample<f64> {
	Sample {input: vector![1.0, 1.0, 0.0, 0.0], expected: vector![1.0, 0.0] }
}

#[cfg(test)]
fn get_mock_data<T>() -> Vec<Sample<f64>> {
	vec![
		// horizontal lines
		get_mock_sample(),
		Sample {input: vector![0.0, 0.0, 1.0, 1.0], expected: vector![1.0, 0.0] },
		// vertical lines
		Sample {input: vector![1.0, 0.0, 1.0, 0.0], expected: vector![0.0, 1.0] },
		Sample {input: vector![0.0, 1.0, 0.0, 1.0], expected: vector![0.0, 1.0] },
	]
}

#[cfg(test)]
fn get_mock_network() -> Network<f64> {
	let mut result: Network<f64> = Network::zeros(4, &[4, 2]);
	result.layers[0].biases  = vector![-0.553, -0.504, 0.813, -0.386];
	result.layers[0].weights = Matrix::new(4, 4, vec![
		-0.881,  0.930, -0.382, -0.652, 
		-0.842,  0.579, -0.610, -0.311, 
		-0.422, -0.711, -0.348,  0.150,
   		-0.069,  0.482,  0.275, -0.242]);

	result.layers[1].biases  = vector![-0.883, 0.103];
	result.layers[1].weights = Matrix::new(2, 4, vec![
		-0.961,  0.450, -0.442, -0.499, 
		-0.284,  0.899,  0.679, -0.614]);

	result
}

#[cfg(test)]
fn assert_eq_vectors<T>(left: &Vector<T>, right: &Vector<T>, tol: T, message: &str) where T: Float + Debug + FromPrimitive {
	assert_eq!(left.size(), right.size(), "({:?}) vector sizes do not match", message);
	if (left-right).apply(&(|x| x*x)).sum()/T::from_f32(left.size() as f32).unwrap() > tol {
		assert!(false, format!("{:?} (left: {:?}, right: {:?})", message, left, right));
	};
}

#[cfg(test)]
fn assert_eq_matrices<T>(left: &Matrix<T>, right: &Matrix<T>, tol: T, message: &str) where T: Float + Debug + FromPrimitive {
	assert_eq!(left.cols(), right.cols(),  "({:?}) matrix widths do not match",  message);
	assert_eq!(left.rows(), right.rows(),  "({:?}) matrix heights do not match", message);
	if (left-right).apply(&(|x| x*x)).sum()/T::from_f32(left.rows() as f32).unwrap()/T::from_f32(left.cols() as f32).unwrap() > tol {
		assert!(false, format!("{:?} (left: {:?}, right: {:?})", message, left, right));
	};
}

#[cfg(test)]
#[test]
fn test_eval() {

	debug!("");
	debug!("================ test eval ================");

	assert_eq_vectors(&vector![0.17639, 0.56356], &get_mock_network().eval(&get_mock_sample().input), 0.0001, "evaluation result");
	
}

#[cfg(test)]
#[test]
fn test_eval_layers() {

	debug!("");
	debug!("================ test eval_layers ================");

	let result = get_mock_network().eval_layers(&get_mock_sample().input);

	assert_eq!(2, result.0.len(), "two weghted sums");
	assert_eq_vectors(&vector![-0.504, -0.767, -0.32, 0.027], &result.0[0], 0.00001, "weighted sum of first layer");
	assert_eq_vectors(&vector![-1.54101, 0.25564],            &result.0[1], 0.00001, "weighted sum of second layer");

	assert_eq!(2, result.1.len(), "two activations");
	assert_eq_vectors(&vector![0.37660, 0.31713, 0.42068, 0.50675], &result.1[0], 0.00001, "activations of first layer");
	assert_eq_vectors(&vector![0.17639, 0.56356],                   &result.1[1], 0.00001, "activations of second layer");
	
}

#[cfg(test)]
#[test]
fn test_sampe_grad() {
	
	/// let us teach network to differ horizontal from vertial lines on 2-by-2 image
	debug!("");
	debug!("======== test_sample_grad ========");

	let sample = get_mock_sample();
	let gradient = get_mock_network().sample_grad(&sample.input, &sample.expected);
	
	assert_eq!(2, gradient.layers.len(), "two layers");

	assert_eq_vectors(&vector![-0.11965, 0.13861], &gradient.layers[1].biases, 0.00001, "biases of second layer");
	assert_eq_matrices(&matrix![-0.045061, -0.037945, -0.050334, -0.060633;
	                             0.052202,  0.043958,  0.058312,  0.070243], &gradient.layers[1].weights, 0.00001, "activations of second layer");

	assert_eq_vectors(&vector![0.017753, 0.0153261, 0.0358261, -0.0063497], &gradient.layers[0].biases, 0.00001, "biases of first layer");
	assert_eq_matrices(&matrix![0.017753,  0.017753, 0.000000, 0.000000;
	                            0.015326,  0.015326, 0.000000, 0.000000; 
	                            0.035826,  0.035826, 0.000000, 0.000000;
	                           -0.006350, -0.006350, 0.000000, 0.000000], &gradient.layers[0].weights, 0.00001, "activations of first layer");

}


#[cfg(test)]
#[test]
fn test_batch_grad() {
	
	/// let us teach network to differ horizontal from vertial lines on 2-by-2 image
	debug!("");
	debug!("======== test_batch_grad ========");

	let data: Vec<Sample<f64>> = get_mock_data::<Vec<Sample<f64>>>();
	let gradient = get_mock_network().batch_grad(&data);
	
	assert_eq!(2, gradient.layers.len(), "two layers");

	assert_eq_vectors(&vector![-0.04731325, 0.02097025], &gradient.layers[1].biases, 0.00001, "biases of second layer");
	assert_eq_matrices(&matrix![-0.012864325, -0.011900175, -0.0254638, -0.0217037;
	                             0.005599925,  0.004623725,  0.0114036,  0.009610225], &gradient.layers[1].weights, 0.00001, "activations of second layer");

	assert_eq_vectors(&vector![0.0074017825, 0.000088975, 0.0077797, 0.002693525], &gradient.layers[0].biases, 0.00001, "biases of first layer");
	assert_eq_matrices(&matrix![ 0.004475,    0.00459075, 0.00281125,  0.002927;
	                             0.0015635,  -0.000522,   0.00061075, -0.00147475;
	                             0.00351125,  0.00451,    0.00326975,  0.0042685;
	                             0.00158575,  0.00112275, 0.00157075,  0.00110775], &gradient.layers[0].weights, 0.00001, "activations of first layer");

}


#[cfg(test)]
fn asses_quality<T: Debug + Float + FromPrimitive + 'static>(network: Network<T>, data: Vec<Sample<T>>) -> f32 {
	let mut guessed: usize = 0;
	for datum in &data {
		guessed += if datum.expected.argmax().0 == network.eval(&datum.input).argmax().0 { 1 } else { 0 };
	}
	(guessed as f32) / (data.len() as f32)
}

#[cfg(test)]
#[test]
fn test_learning() {
	
	/// let us teach network to differ horizontal from vertial lines on 2-by-2 image
	debug!("");
	debug!("======== test_learning ========");

	let data: Vec<Sample<f64>> = get_mock_data::<Vec<Sample<f64>>>();
	let mut network: Network<f64> = get_mock_network();
	println!("initial network: guess rate is {:?}, mse is {:?}", network.classification_rate(&data), network.mean_squared_error(&data));


	let mut previous_result = network.mean_squared_error(&data);
	for i in 0..1000 {
		network.refine(&data, 1.0);
		let result = network.mean_squared_error(&data);
		assert!(result<previous_result, "mean squared error reducing ({:?})", i);
		println!("batch #{:?}: guess rate is {:?}, mse is {:?}", i, network.classification_rate(&data), network.mean_squared_error(&data));
		previous_result = result;
	}

}