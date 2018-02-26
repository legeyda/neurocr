

CARGO?=cargo

RUST_BACKTRACE=full


RUST_LOG?=neurocr=error
ERROR_RUST_LOG?=neurocr=error
DEBUG_RUST_LOG?=neurocr=debug





RUST_ENV?=RUST_BACKTRACE=${RUST_BACKTRACT}

TARGET?=target




all:
	cargo build

run: 
	${RUST_ENV} RUST_LOG=${ERROR_RUST_LOG} ${CARGO} run --release

debug: 
	${RUST_ENV} RUST_LOG=${DEBUG_RUST_LOG} ${CARGO} run

test:
	${RUST_ENV} RUST_LOG=${DEBUG_RUST_LOG} ${CARGO} test -- --nocapture
