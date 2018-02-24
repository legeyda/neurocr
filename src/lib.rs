
#[macro_use]
extern crate log;

#[macro_use]
extern crate rulinalg;

#[cfg(test)]
extern crate env_logger;

extern crate num;
extern crate rand;

mod mnistfiles;

mod network;
mod math;
use rulinalg::vector::Vector;
use std::vec::Vec;
use mnistfiles::IMAGE_SIZE;
use network::{Sample, Network};
use self::num::{Float, One, Zero};
use self::rand::Rand;
const BATCH_SIZE: usize = 10;


/// convert from mnist data to vector of network samples
fn mnist_to_samples(images: &[u8], labels: &[u8]) -> Vec<Sample<f64>> {
    let mut result: Vec<network::Sample<f64>> = Vec::with_capacity(labels.len());

    for i in 0..labels.len() {
        let mut image_data = Vector::zeros(IMAGE_SIZE);
        for pixel in images[IMAGE_SIZE*i..IMAGE_SIZE*(i+1)].iter().enumerate() {
            //// println!("pixel # {:?}, value is {:?}", pixel.0, pixel.1);
            image_data[pixel.0] = *pixel.1 as f64;
            ////println!("pixel result is {:?}", image_data[pixel.0]);
        }

        let mut classification = Vector::zeros(10);
        classification[labels[i] as usize] = 100.0;
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
	<T as Zero>::zero() + rand::random::<T>() * (one + one) - one
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



    let sizes: [usize;2] = [30, 10];
    let mut network: Network<f64> = Network::new(IMAGE_SIZE, &sizes, random);

    debug!("go: initial network is {:?}", network);


    for k in 1..100 {
        for i in 0..trn_data.len()/BATCH_SIZE {
            network.refine(&trn_data[BATCH_SIZE*i..BATCH_SIZE*(i+1)], 100.0);
            //debug!("go: batch #{:?}, gradient is {:?}", i, network.batch_grad(&trn_data[BATCH_SIZE*i..BATCH_SIZE*(i+1)]));


            let mut guessed = 0;
            for datum in &tst_data {
                guessed += if datum.expected.argmax().0 == network.eval(&datum.input).argmax().0 { 1 } else { 0 };
            }
            debug!("go: batch #{:?}, network is {:?}", i, network);
            println!("batch #{:?}: guess rate is {:?}", i, (guessed as f32) / (tst_data.len() as f32));
        }
    }


}