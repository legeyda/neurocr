

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

	/// evaluate activations for neurons of the layer given weighted input
	value:  fn(&Vector<T>)    -> Vector<T>,

	/// evaluate vector of primes of activation function for the layer
	diff:   fn(&Vector<T>)    -> Vector<T>,

	/// Vector<T>
	diff2:  fn(&Vector<T>, &Vector<T>) -> Vector<T>
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

pub fn sumOfSquares(): CostFunc<Float> {
	CostFunc {
		value: |expected: &Vector<T>, actual: &Vector<T>| (actual.clone() - expected).apply(&(|x| x*x)).sum(),
		grad:  |expected: &Vector<T>, actual: &Vector<T>| (actual.clone() - expected).apply(&(|x| x*x)).sum()
	}
}










pub trait Functional<T: Float> {
	/// evaluate function at given point
	fn eval(&self, &Vector<T>)    -> T;

	/// evaluate vector of primes of activation function for the layer
	fn grad(&self, &Vector<T>)    -> Vector<T>;
}

pub struct SumOfSquaresFunctional<T: Float> {
	expected: Vector<T>
}

pub impl<T> SumOfSquaresFunctional<T: Float> {
	fn new(expected: Vector<T>) -> SumOfSquaresFunctional<T> {
		SumOfSquaresFunctional {expected: expected}
	}
}

pub impl<T> Functional<T> for SumOfSquaresFunctional<T> {

	fn eval(&self, point &Vector<T>) -> T {
		(point.clone() - self.expected).apply(&(|x| x*x)).sum()
	}

	fn grad(&self, point: &Vector<T>) -> Vector<T> {
		(point.clone() - self.expected) * (T::one()+T::one());
	}

}










pub struct Map<T: Float> {
	/// evaluate activations for neurons of the layer given weighted input
	eval:    fn(&Vector<T>)    -> Vector<T>,

	/// evaluate vector of primes of activation function for the layer
	primes:   fn(&Vector<T>)    -> Vector<T>,

	/// Vector<T>
	primes2:  fn(&Vector<T>, &Vector<T>) -> Vector<T>
}


impl<T> Map<T> {
	fn new(value: fn(T) -> T, diff: fn(T) -> T, diff2: fn(T, T) -> T) -> ActiFunc<T> {
		ActiFunc {
			value: value, diff: diff, diff2: diff2
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



