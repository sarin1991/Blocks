use blocks::linear_blocks::dyad::dyad_block::parameters::DyadBlockParameterConfig;
use blocks::linear_blocks::dyad::dyad_block::dyad_block::DyadBlock;
use blocks::layer::layer_allocations::LayerAllocations;
use blocks::layer::layer::Layer;
use blocks::types::types::ViewRepr;
use ndarray::Array2;
use std::time::Instant;

fn main() {
    let batch_size = 1024;
    let dyad_dim = 32;
    let dim_in = 32;
    let dim_out = 32;
    let has_bias = true;
    let chunk_size = 32;
    let num_blocks = 1;
    let num_threads_per_block = 1;
    let parameter_config = DyadBlockParameterConfig::new(dyad_dim, dim_in, dim_out, has_bias);
    let dyad_block = DyadBlock::new(&parameter_config);
    let dyad_layer = Layer::<DyadBlock>::new(chunk_size, num_blocks, num_threads_per_block, dyad_block, parameter_config);
    let input = Array2::<f32>::zeros((dyad_dim*dim_in,batch_size));
    let mut output = Array2::<f32>::zeros((dyad_dim*dim_out,batch_size));
    let dyad_params = DyadBlock::allocate_parameters(&parameter_config);
    loop {
        let now = Instant::now();
        dyad_layer.forward(&dyad_params.view_repr(), &input.view(), &mut output.view_mut());
        let elapsed = now.elapsed();
        println!("Forward time taken is - {:#?}",elapsed);
    }
}