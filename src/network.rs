

use std::fmt::Debug;
use std::vec::Vec;
use std::slice::Iter;


extern crate num;
use self::num::{Float, One, Zero, FromPrimitive};

extern crate rand;
use self::rand::Rand;

extern crate rulinalg;
use self::rulinalg::vector::{Vector};

use layer::{Layer, LayerEval, LayerGrad};



#[inline]
fn exp<T: Float>(x: T) -> T {
	x.exp()
}

#[derive(Debug)]
pub struct Sample<T: Float> {
	pub input: Vector<T>,
	pub expected: Vector<T>
}

fn random<T>() -> T where T: Float + Rand {
	let one = <T as One>::one();
	<T as Zero>::zero() + rand::random::<T>() * (one + one + one + one + one + one) - one - one - one
}






fn sum_of_squares_value<T: Float>(expected: &Vector<T>, actual: &Vector<T>) -> T {
	(actual.clone() - expected).apply(&(|x| x*x)).sum()
}

fn sum_of_squares_grad<T: Float>(expected: &Vector<T>, actual: &Vector<T>) -> Vector<T> {
	let two: T=T::one()+T::one();
	(actual.clone() - expected)*two
}

pub struct CostFunc<T: Float> {
	value: fn(&Vector<T>, &Vector<T>) -> T,
	grad:  fn(&Vector<T>, &Vector<T>) -> Vector<T>
}

impl<T: Float> CostFunc<T> {
	fn new(value: fn(&Vector<T>, &Vector<T>) -> T, grad: fn(&Vector<T>, &Vector<T>) -> Vector<T>) -> CostFunc<T> {
		CostFunc { value: value, grad: grad }
	}
}

pub fn create_sum_of_squares_cost_function<T: Float>() -> CostFunc<T> {
	CostFunc::new(sum_of_squares_value, sum_of_squares_grad)
}





//type float where float: Float + Debug + From<f64> + 'static;

pub struct Network<T> where T: Float {
	layers: Vec<Box<Layer<T>>>,
	cost_func: CostFunc<T>
}

impl<T> Network<T> where T: Float + Debug + FromPrimitive + 'static {
	
	pub fn new<'a>(layers: Vec<Box<Layer<T>>>, cost_func: CostFunc<T>) -> Network<T> {
		Network { layers: layers, cost_func: cost_func }
	}
	
	// evaluate weighted sums and outputs of each layer
	pub fn eval<'a>(&'a self, input: &Vector<T>) -> Vector<T> {
		let mut iter: Iter<'a, Box<Layer<T>>> = self.layers.iter();
		let mut value;
		match iter.next() {
			Some(layer) => {
				value = layer.eval(input).output();
			},
			None => panic!("not a single layer in network"),
		}
		loop {
			match iter.next() {
				Some(layer) => {
					value = layer.eval(&value).output()
				},
				None => break,
			}
		}
		value
	}
	
	/// evaluate weighted sums and outputs of each layer
	fn eval_layers<'a>(&'a self, input: &Vector<T>) -> Vec<Box<LayerEval<T> + 'a>> {
		let mut result: Vec<Box<LayerEval<T>>> = Vec::with_capacity(self.layers.len());

		let mut iter: Iter<'a, Box<Layer<T> + 'a>> = self.layers.iter();
		let mut value;
		match iter.next() {
			Some(layer) => {
				result.push(layer.eval(&input));
				value = result[result.len()-1].output();
			},
			None => panic!("not a single layer in network"),
		}
		loop {
			match iter.next() {
				Some(layer) => {
					result.push(layer.eval(&value));
					value = result[result.len()-1].output();
				},
				None => break,
			}
		}
		result
	}
	
	
	/*fn sample_grad<'b, 'c>(&self, input: &'b Vector<T>, expected_output: &'c Vector<T>) -> Network<T> {
		let n = self.layers.len();
		let layer_evals = self.eval_layers(input);

		let result = Vec::with_capacity(n);
		let mut der = (self.cost_func.grad)(&expected_output, &layer_evals[n-1].output());
		if n>1 {
			let mut i: usize = n-1;
			while i>0 {
				let grad = layer_evals[i].grad(&der);
				result.insert(0, grad.param_grad());
				der = grad.input_grad();
				i-=1;
			}
		}
		let grad = layer_evals[0].grad(&der);
		result.insert(0, grad.param_grad());

		Network::new(result)
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
	}*/

	fn sample_grad<'a>(&'a self, input: &Vector<T>, expected: &Vector<T>) -> Vec<Box<LayerGrad<T> + 'a>> {
		let n = self.layers.len();
		let mut result: Vec<Box<LayerGrad<T> + 'a>> = Vec::with_capacity(n);
		let layer_evals = self.eval_layers(input);

		let mut i=n-1;
		result.insert(0, layer_evals[i].grad(&(self.cost_func.grad)(expected, &layer_evals[i].output())));
		
		if i>=0 {
			i=i-1;
			while i>0 {
				result.insert(0, layer_evals[i].grad(&result[0].input_grad()));
				i=i-1;
			}
			result.insert(0, layer_evals[i].grad(&result[0].input_grad()));
		}

		result
	}

	fn batch_grad<'a>(&'a self, test_data: &[Sample<T>]) -> Vec<Vec<Box<LayerGrad<T> + 'a>>> {
		let mut iter = test_data.iter();
		let mut result: Vec<Vec<Box<LayerGrad<T> + 'a>>> = Vec::with_capacity(test_data.len());
		match iter.next() {
			Some(sample) => {
				result.insert(0, self.sample_grad(&sample.input, &sample.expected));
			},
			None => panic!("not a single data in batch"),
		}
		loop {
			match iter.next() {
				Some(sample) => {
					result.insert(0, self.sample_grad(&sample.input, &sample.expected));
				},
				None => break,
			}
		}
		result
	}

	pub fn batch_learn<'a>(&'a mut self, data: &[Sample<T>], step: T) {
		let step_for_each: T = step/T::from_f32(data.len() as f32).unwrap();
		for sample_grads in self.batch_grad(data) {
			for layer_grad in sample_grads {
				layer_grad.apply(step_for_each);
			}
		}
	}


	pub fn cost<'a> (&'a self, input: &Vector<T>,  expected: &Vector<T>) -> T {
		(self.cost_func.value)(expected, &self.eval(input))
	}

	pub fn batch_mean_cost<'a>(&'a self, data: &[Sample<T>]) -> T {
		let mut result: T = Zero::zero();
		for datum in data {
			result = result + self.cost(&datum.input, &datum.expected);
		}
		result/T::from_f32(data.len() as f32).unwrap()
	}

	pub fn sample_class_rate<'a> (&'a self, sample: &Sample<T>) -> f32 {
		if sample.expected.argmax().0 == self.eval(&sample.input).argmax().0 { 1.0 } else { 0.0 }
	}

	pub fn batch_class_rate<'a, 'b> (&'a self, data: &'b [Sample<T>]) -> f32 {
		let mut guessed: f32 = 0.0;
		for datum in data {
			guessed += self.sample_class_rate(datum);
		}
		guessed / (data.len() as f32)
	}


}









#[cfg(test)]
extern crate env_logger;

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