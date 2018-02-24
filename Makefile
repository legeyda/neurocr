

RUST_LOG?=neurocr=debug

all:
	cargo build

run: 
	RUST_BACKTRACE=full RUST_LOG=${RUST_LOG} cargo run

debug: 
	RUST_BACKTRACE=full RUST_LOG=${RUST_LOG} cargo run


log: 
	mkdir -p target
	RUST_BACKTRACE=full RUST_LOG=${RUST_LOG} cargo run > target/run-output.log 2>&1

test:
	RUST_BACKTRACE=full RUST_LOG=${RUST_LOG} cargo test -- --nocapture

test-log:
	mkdir -p target
	RUST_BACKTRACE=full RUST_LOG=${RUST_LOG} cargo test -- --nocapture > target/test-output.log 2>&1