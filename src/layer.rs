extern crate num;
use self::num::Float;

extern crate rulinalg;
use self::rulinalg::matrix::{Matrix, BaseMatrix, BaseMatrixMut};
use self::rulinalg::vector::Vector;





#[inline]
fn exp<T: Float>(x: T) -> T {
	x.exp()
}

pub struct actfunc<T> {
	eval()
}

/// constians function references for evaluating of activation function and its derivative
#[derive(Debug)]
pub struct ActiFunc<T> {
	value:  fn(T)    -> T,
	value2: fn(T, T) -> T,
	diff:   fn(T)    -> T,
	diff2:  fn(T, T) -> T
}

impl<T> ActiFunc<T> {
	fn new(value: fn(T) -> T, value2: fn(T, T) -> T, diff: fn(T) -> T, diff2: fn(T, T) -> T) -> ActiFunc<T> {
		ActiFunc {
			value: value, value2: value2, diff: diff, diff2: diff2
		}
	}
}




fn sigmoid<T: Float>(x: T) -> T {
	let one: T = T::one();
	one / (one + exp(-x))
}

fn sigmoid2<T: Float>(x: T, _diff: T) -> T {
	sigmoid(x)
}

fn sigmoid_diff<T: Float>(x: T) -> T {
	sigmoid_diff2(x, sigmoid(x))
}

fn sigmoid_diff2<T: Float>(_x: T, value_at_x: T) -> T {
	value_at_x * (T::one() - value_at_x)
}

pub fn create_sigmoid_actifunc<T: Float>() -> ActiFunc<T> {
	ActiFunc::new(sigmoid, sigmoid2, sigmoid_diff, sigmoid_diff2)
}













pub trait Layer<T> where T: Float {
	fn input(&'mut self, input: &Vector<T>);
	fn output(&self) -> Vector<T>;
	fn grad(&self, output_grad: &Vector<T>) -> Vector<T>; 
	fn update(&'mut self, delta: &Vector<T>)
}



/// structs for weighted layer
#[derive(Debug)]
pub struct WeightedLayer<T> where T: Float {
	biases:   Vector<T>,
	weights:  Matrix<T>,
	function: ActiFunc<T>
}

impl<T> WeightedLayer<T> where T: Float {
	
	pub fn new(biases: Vector<T>, weights: Matrix<T>, function: ActiFunc<T>) -> WeightedLayer<T>  {
		WeightedLayer {
			biases:   Vector::zeros(num_neurons)            .apply(&(|_ignored| generator())), 
			weights:  Matrix::zeros(num_neurons, num_inputs).apply(&(|_ignored| generator())),
			function: function
		}
	}

}

impl <T> Layer<T> for WeightedLayer<T> where T: Float {
	fn eval<'a>(&'a self, input: &Vector<T>) -> Box<LayerEval<T> + 'a> {
		Box::new(WeightedLayerEval::new(self, input))		
	}
}



struct WeightedLayerEval<'this, T> where T: Float + 'this {
	layer:         &'this WeightedLayer<T>,
	input:         Vector<T>,
	weighted_sums: Vector<T>,
	activations:   Vector<T>
}

impl<'this, T> WeightedLayerEval<'this, T> where T: Float {
	pub fn new<'a>(layer: &'a WeightedLayer<T>, input: &Vector<T>) -> WeightedLayerEval<'a, T> {
		let n = input.size();
		assert_eq!(layer.biases.size(),  n, "size mismatch: self.biases size is {:?}, but input size is {:?}", layer.biases.size(), n);
		assert_eq!(layer.weights.cols(), n, "size mismatch: self.weights width is {:?}, but input size is {:?}", layer.weights.cols(), n);


		let weighted_sums = layer.weights.clone() * input + layer.biases.clone();
		let activations   = weighted_sums.clone().apply(&layer.function.value);

		WeightedLayerEval {
			layer: layer,
			input: input.clone(),
			weighted_sums: weighted_sums,
			activations:   activations
		}
	}
}

impl<'this, T> LayerEval<T> for WeightedLayerEval<'this, T> where T: Float {

	fn output(&self) -> Vector<T> {
		self.activations.clone()
	}

	fn grad<'a>(&'a self, output_grad: &Vector<T>) -> Box<LayerGrad<T> + 'a> {
		Box::new(WeightedLayerGrad::new(self, output_grad))
	}

}



struct WeightedLayerGrad<'a, T> where T: Float + 'a {
	layer_eval:         &'a WeightedLayerEval<'a, T>,
	weighted_sums_grad: Vector<T>
}

impl<'a, T> WeightedLayerGrad<'a, T> where T: Float {
	pub fn new<'b>(layer_eval: &'a WeightedLayerEval<'a, T>, output_grad: &'b Vector<T>) -> WeightedLayerGrad<'a, T> {
		let n = layer_eval.activations.size();
		assert_eq!(n, output_grad.size(), "LayerEval::grad: layer size is {:?}, output size {:?}", n, output_grad.size());
		
		let mut weighted_sums_grad = Vector::zeros(n);
		for i in 0..n {
			weighted_sums_grad[0] = output_grad[i] * (layer_eval.layer.function.diff2)(layer_eval.weighted_sums[i], layer_eval.activations[i]);
		};

		WeightedLayerGrad {
			layer_eval:         layer_eval,
			weighted_sums_grad: weighted_sums_grad
		}
	}
}

impl<'a, T> LayerGrad<T> for WeightedLayerGrad<'a, T> where T: Float + 'a {

	fn input_grad(&self) -> Vector<T> {
		// transpose(weights) * nabla_weighted_sums
		let mut result: Vector<T> = Vector::zeros(self.layer_eval.layer.weights.rows());
		for j in 0..self.layer_eval.layer.weights.cols() {
			for i in 0..self.layer_eval.layer.weights.rows() {
				result[j] = result[j] + self.layer_eval.layer.weights.row(i)[j];
			}
		}
		result	
	}

	fn apply(&mut self, multiplier: T) {
		// biases  += nabla_weignted_sums*multiplier
		// weights += transpose(nabla_weighted_sums)*inputs .* multiplier
		for i in 0..self.layer_eval.layer.biases.size() {
			self.layer_eval.layer.biases[i] = self.layer_eval.layer.biases[i] + self.weighted_sums_grad[i] * multiplier;
			for j in 0..self.layer_eval.layer.weights.cols() {
				self.layer_eval.layer.weights.row(i)[j] = 
				self.layer_eval.layer.weights.row(i)[j] + 
						self.weighted_sums_grad[i]*self.layer_eval.input[j]*multiplier;
			}
		}
	}

}








