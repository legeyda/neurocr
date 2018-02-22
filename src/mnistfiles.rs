extern crate mnist;
extern crate rulinalg;



use std::env::home_dir;
use std::fs::create_dir_all;
use std::path::PathBuf;
use self::mnist::{Mnist, MnistBuilder};
use std::vec::Vec;

fn get_share_dir() -> PathBuf {
	match home_dir() {
		Some(mut path) => {
			path.push(".local");
			path.push("share");
			path.push("mnist");
			return path;
		}
		None => panic!("unable to get user home dir")
	}
}

fn get_cache_dir() -> PathBuf {
	match home_dir() {
		Some(mut path) => {
			path.push(".cache");
			path.push("mnist");
			return path;
		}
		None => panic!("unable to get user home dir")
	}
}

pub fn ensure_base_dir() -> PathBuf {
	// todo search files in .share/local/mnist,
	// then in .cache/mnist,
	// then download 
	let cache_dir = get_cache_dir();
	create_dir_all(cache_dir.as_path());
	get_share_dir()
}


//

// pub struct MnistData {
// 	train_images: Matrix<f64>,
// 	train_labels: Vector<u8>,
// 	test_images: Matrix<f64>,
// 	test_labels: Vector<u8>,
// }


pub const IMAGE_WIDTH:  usize = 28;
pub const IMAGE_HEIGHT: usize = 28;
pub const IMAGE_SIZE:   usize = IMAGE_HEIGHT*IMAGE_WIDTH;


pub fn load_data() -> Mnist {
	MnistBuilder::new()
		.label_format_digit()
		//.label_format_one_hot()
		.base_path(ensure_base_dir().as_os_str().to_str().unwrap())
		.finalize()

	// MnistData {
	// 	train_images: Matrix::new(mnist.trn_img.len()/IMAGE_SIZE, IMAGE_SIZE, vec_to_f64(mnist.trn_img)),
	// 	train_labels: Vector::new(vec_to_f64(mnist.trn_lbl)),
	// 	test_images:  Matrix::new(mnist.tst_img.len()/IMAGE_SIZE, IMAGE_SIZE, vec_to_f64(mnist.test_images.data()),
	// 	test_labels:  Vector::new(vec_to_f64(mnist.tst_lbl)) 
	// }
}

