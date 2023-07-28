pub mod blas;
pub mod block;
pub mod types;
pub mod linear_blocks;
pub mod layer;
pub mod utils;
pub mod activations;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
