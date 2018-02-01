
//extern crate num;
extern crate rulinalg;

extern crate num;


use self::rulinalg::matrix::{Matrix};
use self::rulinalg::vector::{Vector};
use std::any::Any;
use self::num::{Float, One, FromPrimitive};
use std::ops::Fn;


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
	
	
	fn weighted_sum(&self, input: &Vector<T>) -> Vector<T> {
		&(self.biases) + Vector::new((&(self.weights)*input).into_vec())
	}
	
	fn eval(&self, input: &Vector<T>) -> Vector<T> {
		self.weighted_sum(input).apply(&sigmoid)
	}
	
	
	//fn grad(&self, input: Vector<T>) -> (Vector<T>, Matrix<T>) {
		// for each neuron, partial derivative of cost function
		// with respect to weighted_sum
		//let dc_dz:   
		
		
		//
	//	(self.biases.clone(), self.weights.clone())
	//}
	
}

#[test]
fn test_layer_eval() {
	println!("hello from test!");
	let layer = Layer {
		biases: vector!(1.0, 2.0, 3.0),
		weights: matrix!(
				1.0, 2.0, 3.0;
				4.0, 5.0, 6.0;
				7.0, 8.0, 9.0)
	};
	let expected = vector!(69.0, 169.0, 269.0);
	let input = vector!(10.0, 11.0, 12.0);
	let actual = layer.eval(&input); 
	assert_vector_eq!(
		expected, actual
		
	);
}


/*

pub trait Network {
	type Type = Float;
}

pub struct SimpleNetwork<T> where T: Float {
	layers: Vec<Layer>
}

impl<T> Network<T> where T: Float {

	fn new(sized: Vec<usize>) -> Network<T> {
		let result = Network<T>{Vec
	}
		
	fn eval(input: Vector<T>) -> Vector<T> {
		
	}
		
}
*/


