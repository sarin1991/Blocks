use ndarray::{Array1,ArrayView1,ArrayViewMut1};
use ndarray::{Array2,ArrayViewMut2};
use crate::linear_blocks::dyad::dyad_block::dyad_block::DyadBlock;
use crate::linear_blocks::dyad::dyad_block::parameters::{DyadBlockParameterConfig, DyadBlockParameter, DyadBlockParameterView, DyadBlockParameterMutView};
use crate::activations::relu::ReLUBlock;
use crate::norm::rms_norm::RMSNormBlock;
use crate::types::types::{BlockConfig, ViewRepr};
use crate::block::{OwnedBlock, BaseBlock, FusedBlock, Block};
use crate::layer::layer::LayerChunk;
use crate::layer::layer_allocations::LayerAllocations;
use crate::types::types::BlockTypeAllocate;

#[derive(Clone,Copy,Debug)]
pub struct RMSDyadReLUBlockParameterConfig {
    pub dyad_block_parameter_config: DyadBlockParameterConfig,
    pub rms_norm_chunk_size: usize,
}
impl BlockConfig for RMSDyadReLUBlockParameterConfig {}

impl RMSDyadReLUBlockParameterConfig {
    pub fn new(dyad_dim: usize, dim_in: usize, 
        dim_out: usize, has_bias: bool, rms_norm_chunk_size: usize) -> Self {
        let dyad_block_parameter_config = 
            DyadBlockParameterConfig::new(dyad_dim, dim_in, dim_out, has_bias);
        Self { 
            dyad_block_parameter_config, 
            rms_norm_chunk_size,
        }
    }
}

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
    type P = (Array1<f32>,DyadBlockParameter);
    type A = Array1<f32>;
    type I = Array2<f32>;
    type F = f32;
    type O = Array2<f32>;
}

impl OwnedBlock for RMSDyadReLUBlock {
    fn forward(&self, 
            parameters:&(ArrayView1<f32>,DyadBlockParameterView),
            input:&mut ArrayViewMut2<f32>,
            output:&mut ArrayViewMut2<f32>,
            allocations:&mut ArrayViewMut1<f32>,
            _forward_context:&mut &mut f32) {
        //rms norm
        self.rms_block.forward(&parameters.0, input, &mut &mut 0.0f32, allocations);
        //dyad block
        self.dyad_block.forward(&parameters.1, &input.view_repr(), output, &mut &mut 0.0f32, &mut &mut 0.0f32);
        //relu block
        self.relu_block.forward(&&0.0f32, output, &mut &mut 0.0f32, &mut &mut 0.0f32);
    }
    fn backward(&self,
            parameter_gradients:&mut (ArrayViewMut1<f32>,DyadBlockParameterMutView),
            input_gradients:&mut ArrayViewMut2<f32>,
            output_gradients:&mut ArrayViewMut2<f32>,
            output:&mut ArrayViewMut2<f32>,
            input:&mut ArrayViewMut2<f32>,
            parameters:&(ArrayView1<f32>,DyadBlockParameterView),
            allocations:&mut ArrayViewMut1<f32>,
            _forward_context:&&f32,
        ) {
        // relu block
        self.relu_block.backward(&mut &mut 0.0f32, output_gradients, 
            output, &&0.0f32, &mut &mut 0.0f32, &&0.0f32);
        // dyad backward
        self.dyad_block.backward(&mut parameter_gradients.1, input_gradients, 
            &output_gradients.view(), &output.view(), &input.view(), 
            &parameters.1, &mut &mut 0.0f32, &&0.0f32);
        // rms norm backward
        self.rms_block.backward(&mut parameter_gradients.0, 
            input_gradients, input, 
            &parameters.0, &mut &mut 0.0f32, &allocations.view());
    }
}

impl LayerChunk for RMSDyadReLUBlock {
    fn layer_chunk_forward<'p>(&self, 
            parameters:&(ArrayView1<'p,f32>,DyadBlockParameterView<'p>),
            mut input:ArrayViewMut2<f32>,
            mut output:ArrayViewMut2<f32>,
            allocations:&mut ArrayViewMut1<f32>,
            forward_context:&mut &mut f32) {
        self.forward(parameters, &mut input, &mut output, allocations, forward_context);
    }
    fn layer_chunk_backward<'gp,'p,'a,'f>(&self,
            parameter_gradients:&mut (ArrayViewMut1<'gp,f32>,DyadBlockParameterMutView<'gp>),
            mut input_gradients:ArrayViewMut2<f32>,
            mut output_gradients:ArrayViewMut2<f32>,
            mut output:ArrayViewMut2<f32>,
            mut input:ArrayViewMut2<f32>,
            parameters:&(ArrayView1<'p,f32>,DyadBlockParameterView<'p>),
            allocations:&mut ArrayViewMut1<f32>,
            forward_context:&&f32,
        ) {
        self.backward(parameter_gradients, &mut input_gradients, &mut output_gradients, 
            &mut output, &mut input, parameters, allocations, forward_context);
    }
}

impl LayerAllocations for RMSDyadReLUBlock{
    type AllocationConfig = RMSDyadReLUBlockParameterConfig;
    fn allocate_parameters(config:&Self::AllocationConfig) -> Self::P {
        let dyad_param = DyadBlockParameter::allocate(&config.dyad_block_parameter_config);
        let dim = config.dyad_block_parameter_config.dyad_dim*config.dyad_block_parameter_config.dim_in;
        let rms_norm_param = Array1::<f32>::zeros(dim);
        (rms_norm_param,dyad_param)
    }
    fn create_allocations(chunk_size:usize,_config:&Self::AllocationConfig) -> Self::A {
        let rms_array = Array1::<f32>::zeros(chunk_size);
        rms_array
    }
    fn allocate_forward_context(_chunk_size:usize,_config:&Self::AllocationConfig) -> Self::F {
        0.0f32
    }
}