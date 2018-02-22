
extern crate env_logger;

extern crate neurocr;

fn main() {
	env_logger::init();
	neurocr::go();
}
