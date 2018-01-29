extern crate mnist;
extern crate rulinalg;



use std::env::home_dir;
use std::fs::create_dir_all;
use std::ffi::OsStr;
use std::path::PathBuf;
use self::mnist::{Mnist, MnistBuilder};
use self::rulinalg::matrix::{BaseMatrix, BaseMatrixMut, Matrix};
use self::rulinalg::vector::{Vector};

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

pub fn ensure_files() -> PathBuf {
	// todo search files in .share/local/mnist,
	// then in .cache/mnist,
	// then download 
	let cache_dir = get_cache_dir();
	create_dir_all(cache_dir.as_path());
	cache_dir
}

pub struct MnistData {
	train_images: Matrix<u8>,
	train_labels: Vector<u8>,
	test_images: Matrix<u8>,
	test_labels: Vector<u8>,
}

pub fn get_data() -> MnistData {
	const  IMAGE_SIZE: usize = 28*28;
		
	let mnist = MnistBuilder::new()
		.base_path(get_share_dir().as_os_str().to_str().unwrap())
        .finalize();
	
	MnistData {
		train_images: Matrix::new(mnist.trn_img.len()/IMAGE_SIZE, IMAGE_SIZE, mnist.trn_img),
		train_labels: Vector::new(mnist.trn_lbl),
		test_images:  Matrix::new(mnist.tst_img.len()/IMAGE_SIZE, IMAGE_SIZE, mnist.tst_img),
		test_labels:  Vector::new(mnist.tst_lbl) 
	}
}

