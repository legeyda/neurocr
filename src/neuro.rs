





/// function from scalar to scalar
pub trait Function<T> where T: Sized {

    /// function value at given point
    fn value(&self, input: T) -> T;

    /// if function derivative at this point is known
    /// 
    fn value2(&self, input: T, deriv: T) -> T;
   
};

/// 
pub trait Differentiable<T>: Function<T> where T: Sized {

    /// derivative at given point
    fn diff(&self, input: T) -> T;

    /// derivative at given point
    /// can (or not ) make use of value for optimization
    fn diff2(&self, input: T, value T) -> T;

};






/// sigmoid function 1/(1+exp(-x))
struct Sigmoid {};

impl Sigmoid {
    fn new() -> Sigmoid {
        Sigmoid{}
    }
};


impl<T> Function<T> for Sigmoid where T: Sized + Float {

    fn value(&self, input: T) -> T {
        let one: T = T::one();
        one / (one + exp(-x))
    }

    fn value2(&self, input: T, diff: T) -> T {
        self.value(input); // todo maybe make use of diff somehow?
    }

};

impl<T> Differentiable<T> for Sigmoid where T: Sized + Float {


    fn diff(input: T) -> T {
        let one: T = One::one();
        let sigma: T = self::value(x);
        sigma * (one - sigma)
    }


    /// derivative at given point
    /// can (or not ) make use of value for optimization
    fn diff2(input: T, value T) -> T {
        value * (T::one() - value)
    }


};







/// sigmoid function -ln(x)
struct LnFunction {};

impl LnFunction {
    fn new() -> LnFunction {
        LnFunction{}
    }
};