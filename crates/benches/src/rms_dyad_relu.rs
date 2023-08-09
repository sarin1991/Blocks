use blocks::non_linear_blocks::rms_dyad_relu::rms_dyad_relu::RMSDyadReLUBlock;
use blocks::non_linear_blocks::rms_dyad_relu::parameters::RMSDyadReLUBlockParameterConfig;
use blocks::layer::layer_allocations::LayerAllocations;
use blocks::layer::layer::Layer;
use blocks::types::types::{ViewRepr, ViewMutRepr};
use ndarray::Array2;
use std::time::Instant;

fn main() {
    let batch_size = 1024*128;
    let dyad_dim = 32;
    let dim_in = 32;
    let dim_out = 32;
    let has_bias = true;
    let chunk_size = 128;
    let num_blocks = 4;
    let num_threads_per_block = 2;
    let rms_norm_chunk_size = 16;
    let parameter_config = RMSDyadReLUBlockParameterConfig::new(dyad_dim, dim_in, dim_out, has_bias, rms_norm_chunk_size);
    let rms_dyad_relu_block = RMSDyadReLUBlock::new(&parameter_config);
    let rms_dyad_relu_layer = Layer::<RMSDyadReLUBlock>::new(chunk_size, num_blocks, num_threads_per_block, rms_dyad_relu_block, parameter_config);
    let input = Array2::<f32>::zeros((dyad_dim*dim_in,batch_size));
    let mut output = Array2::<f32>::zeros((dyad_dim*dim_out,batch_size));
    let mut input_grad = Array2::<f32>::zeros((dyad_dim*dim_in,batch_size));
    let output_grad = Array2::<f32>::zeros((dyad_dim*dim_out,batch_size));
    let rms_dyad_relu_params = RMSDyadReLUBlock::allocate_parameters(&parameter_config);
    let mut rms_dyad_relu_param_grad = RMSDyadReLUBlock::allocate_parameters(&parameter_config);
    loop {
        let now = Instant::now();
        rms_dyad_relu_layer.forward(&rms_dyad_relu_params.view_repr(), &input.view(), &mut output.view_mut());
        let elapsed = now.elapsed();
        println!("Forward time taken is - {:#?}",elapsed);
        let now = Instant::now();
        rms_dyad_relu_layer.backward(&mut rms_dyad_relu_param_grad.view_mut_repr(), 
            &mut input_grad.view_mut(), &output_grad.view(), 
            &input.view(), &rms_dyad_relu_params.view_repr());
        let elapsed = now.elapsed();
        println!("Backward time taken is - {:#?}",elapsed);
    }
}