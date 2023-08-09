use ndarray::{Array1,ArrayView1,ArrayViewMut1};
use ndarray::{Array2,ArrayViewMut2};
use crate::linear_blocks::dyad::dyad_block::dyad_block::DyadBlock;
use crate::activations::relu::ReLUBlock;
use crate::norm::rms_norm::RMSNormBlock;
use crate::types::types::ViewRepr;
use crate::block::{OwnedBlock, BaseBlock, FusedBlock, Block};
use crate::layer::layer::LayerChunk;
use crate::layer::layer_allocations::LayerAllocations;
use crate::types::types::BlockTypeAllocate;
use super::parameters::RMSDyadReLUBlockParameterConfig;
use super::parameters::{RMSDyadReLUBlockParameter,RMSDyadReLUBlockParameterView,RMSDyadReLUBlockParameterMutView};

pub struct RMSDyadReLUBlock {
    rms_block: RMSNormBlock,
    dyad_block: DyadBlock,
    relu_block: ReLUBlock,
    pub parameter_config: RMSDyadReLUBlockParameterConfig,
}

impl RMSDyadReLUBlock {
    pub fn new(rms_dyad_relu_block_config:&RMSDyadReLUBlockParameterConfig) -> Self {
        let dyad_config = &rms_dyad_relu_block_config.dyad_block_parameter_config;
        let dyad_block = DyadBlock::new(&dyad_config);
        let dim = dyad_config.dyad_dim * dyad_config.dim_in;
        let rms_block = RMSNormBlock::new(dim, rms_dyad_relu_block_config.rms_norm_chunk_size);
        let relu_block = ReLUBlock::new();
        Self {
            rms_block,
            dyad_block,
            relu_block,
            parameter_config:*rms_dyad_relu_block_config,
        }
    }
}

impl BaseBlock for RMSDyadReLUBlock {
    type P = RMSDyadReLUBlockParameter;
    type A = f32;
    type I = Array2<f32>;
    type F = Array1<f32>;
    type O = Array2<f32>;
}

impl OwnedBlock for RMSDyadReLUBlock {
    fn forward(&self, 
            parameters:&RMSDyadReLUBlockParameterView,
            input:&mut ArrayViewMut2<f32>,
            output:&mut ArrayViewMut2<f32>,
            _allocations:&mut &mut f32,
            forward_context:&mut ArrayViewMut1<f32>) {
        //rms norm
        self.rms_block.forward(&parameters.rms_params, input, &mut &mut 0.0f32, forward_context);
        //dyad block
        self.dyad_block.forward(&parameters.dyad_block_params, &input.view_repr(), output, &mut &mut 0.0f32, &mut &mut 0.0f32);
        //relu block
        self.relu_block.forward(&&0.0f32, output, &mut &mut 0.0f32, &mut &mut 0.0f32);
    }
    fn backward(&self,
            parameter_gradients:&mut RMSDyadReLUBlockParameterMutView,
            input_gradients:&mut ArrayViewMut2<f32>,
            output_gradients:&mut ArrayViewMut2<f32>,
            output:&mut ArrayViewMut2<f32>,
            input:&mut ArrayViewMut2<f32>,
            parameters:&RMSDyadReLUBlockParameterView,
            _allocations:&mut &mut f32,
            forward_context:&ArrayView1<f32>,
        ) {
        // relu block
        self.relu_block.backward(&mut &mut 0.0f32, output_gradients, 
            output, &&0.0f32, &mut &mut 0.0f32, &&0.0f32);
        // dyad backward
        self.dyad_block.backward(&mut parameter_gradients.dyad_block_params, input_gradients, 
            &output_gradients.view(), &output.view(), &input.view(), 
            &parameters.dyad_block_params, &mut &mut 0.0f32, &&0.0f32);
        // rms norm backward
        self.rms_block.backward(&mut parameter_gradients.rms_params, 
            input_gradients, input, 
            &parameters.rms_params, &mut &mut 0.0f32, &forward_context.view());
    }
}

impl LayerChunk for RMSDyadReLUBlock {
    fn layer_chunk_forward<'p>(&self, 
            parameters:&RMSDyadReLUBlockParameterView,
            mut input:ArrayViewMut2<f32>,
            mut output:ArrayViewMut2<f32>,
            allocations:&mut &mut f32,
            forward_context:&mut ArrayViewMut1<f32>) {
        self.forward(parameters, &mut input, &mut output, allocations, forward_context);
    }
    fn layer_chunk_backward<'gp,'p,'a,'f>(&self,
            parameter_gradients:&mut RMSDyadReLUBlockParameterMutView,
            mut input_gradients:ArrayViewMut2<f32>,
            mut output_gradients:ArrayViewMut2<f32>,
            mut output:ArrayViewMut2<f32>,
            mut input:ArrayViewMut2<f32>,
            parameters:&RMSDyadReLUBlockParameterView,
            allocations:&mut &mut f32,
            forward_context:&ArrayView1<f32>,
        ) {
        self.backward(parameter_gradients, &mut input_gradients, &mut output_gradients, 
            &mut output, &mut input, parameters, allocations, forward_context);
    }
}

impl LayerAllocations for RMSDyadReLUBlock{
    type AllocationConfig = RMSDyadReLUBlockParameterConfig;
    fn allocate_parameters(config:&Self::AllocationConfig) -> Self::P {
        RMSDyadReLUBlockParameter::allocate(config)
    }
    fn create_allocations(_chunk_size:usize,_config:&Self::AllocationConfig) -> Self::A {
        0.0f32
    }
    fn allocate_forward_context(chunk_size:usize,_config:&Self::AllocationConfig) -> Self::F {
        let rms_array = Array1::<f32>::zeros(chunk_size);
        rms_array
    }
}