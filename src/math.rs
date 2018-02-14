
use std::ops::{Add, Mul};

extern crate num;
use self::num::{Float, One};



/// the idea of functional on a vector space
trait Functional<VectorType, ScalarType> 
		where VectorType: Sized + Add<VectorType, Output=VectorType> + Mul<ScalarType, Output=VectorType>, 
		      ScalarType: Sized + Float {

	/// evaluate functional value at given point
	fn eval(&self, point: &VectorType) -> ScalarType;

	/// evaluate function gradient at given point
	fn grad(&self, point: &VectorType) -> VectorType;
	
}



fn minimize<VectorType, ScalarType>(subject:        &Functional<VectorType, ScalarType>,
		                            mut estimation:  VectorType, 
		                            mut step:        ScalarType, 
		                            on_new:         &Fn(&VectorType) -> bool,
		                            max_iterations: u16) -> VectorType
		where VectorType: Sized + Copy + Add<VectorType, Output=VectorType> + Mul<ScalarType, Output=VectorType>, 
		      ScalarType: Sized + Copy + Float {
    
	// type annotations required: cannot resolve `<_ as std::ops::Mul>::Output == _`
	// note: required by `network::num::One::one`
	//let one_and_half: ScalarType = One::one() * 1.5;
    
    let one: ScalarType = One::one();
    let one_and_half = (one+one) / (one+one+one);

	let mut i=0;
	while i<max_iterations {
		let mut value: ScalarType = subject.eval(&estimation);
		let grad = subject.grad(&estimation);

		let mut new_estimation: VectorType = estimation + grad*(-step);
        if subject.eval(&new_estimation) < value {
            step = step * one_and_half;
        } else {
            loop {
                step = step * one_and_half;
                new_estimation = estimation + grad*(-step);
                if i>max_iterations || subject.eval(&new_estimation) < value {
                    break;
                }
            }
        };
        estimation = new_estimation;
		if !on_new(&estimation) {
			break;
		}
        i=i+1;
	};

    estimation
}