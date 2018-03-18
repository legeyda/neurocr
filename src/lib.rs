use std::vec::Vec;


extern crate log;

#[cfg(test)]
extern crate env_logger;



extern crate num;
use self::num::{Float, One, Zero};

extern crate rand;
use self::rand::Rand;

extern crate rulinalg;
use rulinalg::vector::Vector;


mod layer;
use layer::{WeightedLayer, create_sigmoid_actifunc};


mod mnistfiles;
use mnistfiles::IMAGE_SIZE;

mod network;
use network::{Sample, Network};
use network::create_sum_of_squares_cost_function;



const BATCH_SIZE: usize = 10;


/// convert from mnist data to vector of network samples
fn mnist_to_samples(images: &[u8], labels: &[u8]) -> Vec<Sample<f64>> {
    let mut result: Vec<network::Sample<f64>> = Vec::with_capacity(labels.len());

    for i in 0..labels.len() {
        let mut image_data: Vector<f64> = Vector::zeros(IMAGE_SIZE);
        for pixel in images[IMAGE_SIZE*i..IMAGE_SIZE*(i+1)].iter().enumerate() {
            //// println!("pixel # {:?}, value is {:?}", pixel.0, pixel.1);
            image_data[pixel.0] = ((*pixel.1) as f64) / 256.0;
            ////println!("pixel result is {:?}", image_data[pixel.0]);
        }

        let mut classification: Vector<f64> = Vector::zeros(10);
        classification[labels[i] as usize] = 1.0;
        ////println!("classification at {:?} is {:?}", i, classification);


        result.push(Sample {
            input:    image_data, 
            expected: classification
        });
    }
    result
}

fn random<T>() -> T where T: Float + Rand {
    let one = <T as One>::one();
	<T as Zero>::zero() + rand::random::<T>() * (one + one + one + one + one + one) - one - one - one
}

pub fn go() {
    let mnist_data = mnistfiles::load_data();
    
// MnistData {
	// 	train_images: Matrix::new(mnist.trn_img.len()/IMAGE_SIZE, IMAGE_SIZE, vec_to_f64(mnist.trn_img)),
	// 	train_labels: Vector::new(vec_to_f64(mnist.trn_lbl)),
	// 	test_images:  Matrix::new(mnist.tst_img.len()/IMAGE_SIZE, IMAGE_SIZE, vec_to_f64(mnist.test_images.data()),
	// 	test_labels:  Vector::new(vec_to_f64(mnist.tst_lbl)) 
	// }

    let trn_data: Vec<network::Sample<f64>> = mnist_to_samples(&mnist_data.trn_img, &mnist_data.trn_lbl);
    let tst_data: Vec<network::Sample<f64>> = mnist_to_samples(&mnist_data.tst_img, &mnist_data.tst_lbl);



    let mut network: Network<f64> = Network::new(vec![
        Box::new(WeightedLayer::new(IMAGE_SIZE, 100, create_sigmoid_actifunc(), random)), 
        Box::new(WeightedLayer::new(100, 10, create_sigmoid_actifunc(), random))
    ], create_sum_of_squares_cost_function());

    //debug!("go: initial network is {:?}", network);
    let _gr = network.batch_class_rate(&tst_data);
    println!("initial network: guess rate is {:?}, mse is {:?}", network.batch_class_rate(&tst_data), network.batch_mean_cost(&tst_data));

    for k in 1..101 {
        for i in 0..trn_data.len()/BATCH_SIZE-1 {
            network.batch_learn(&trn_data[BATCH_SIZE*i..BATCH_SIZE*(i+1)], 50.0/(BATCH_SIZE as f64));
            //debug!("go: batch #{:?}, network is {:?}", i, network);
        }
        println!("epoch #{:?}: guess rate is {:?}, mse is {:?}", k, network.batch_class_rate(&tst_data), network.batch_mean_cost(&tst_data));
    }


}