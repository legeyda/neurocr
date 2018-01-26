
use std::env;
use std::env::home_dir;
use std::fs::create_dir_all;
use std::ffi::OsString;
use std::path::PathBuf;

fn get_base_dir() -> PathBuf {
	match home_dir() {
		Some(mut path) => {
			path.push(".cache");
			path.push(env!("CARGO_PKG_NAME"));
			return path;
		}
		None => panic!("unable to get user home dir")
	}
}

fn ensure_base_dir() -> PathBuf {
	let result = get_base_dir();
	create_dir_all(result.as_path());
	return result;
}


pub fn ensure() -> PathBuf {
	return ensure_base_dir();
}