RUSTFLAGS="-Ctarget-cpu=native -lblas" cargo build --release

cd target/release/
mv libblocks_extension.so blocks_extension.so
mv blocks_extension.so ../../.env/lib/python3.8/site-packages/

maturin develop -r -- "-Ctarget-cpu=native" "-lblas"