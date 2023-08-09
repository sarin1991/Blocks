use ndarray::{Array1,ArrayView1,ArrayViewMut1};
use crate::types::types::{BlockType,BlockTypeAllocate, ValidateRepr};
use crate::types::types::{ViewRepr,ViewMutRepr};
use crate::types::types::BlockConfig;
use crate::linear_blocks::dyad::dyad_block::parameters::DyadBlockParameterConfig;
use crate::linear_blocks::dyad::dyad_block::parameters::{DyadBlockParameter,
    DyadBlockParameterView,DyadBlockParameterMutView};
use std::ops::AddAssign;

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

pub struct RMSDyadReLUBlockParameterView<'a> {
    pub rms_params: ArrayView1<'a,f32>,
    pub dyad_block_params: DyadBlockParameterView<'a>,
}
impl<'a> BlockType for RMSDyadReLUBlockParameterView<'a> {
    type Config = RMSDyadReLUBlockParameterConfig;
}

pub struct RMSDyadReLUBlockParameterMutView<'a> {
    pub rms_params: ArrayViewMut1<'a,f32>,
    pub dyad_block_params: DyadBlockParameterMutView<'a>,
}
impl<'a> AddAssign<RMSDyadReLUBlockParameter> for RMSDyadReLUBlockParameterMutView<'a> {
    fn add_assign(&mut self, rhs: RMSDyadReLUBlockParameter) {
        self.rms_params += &rhs.rms_params;
        self.dyad_block_params += rhs.dyad_block_params;    }
}
impl<'a> BlockType for RMSDyadReLUBlockParameterMutView<'a> {
    type Config = RMSDyadReLUBlockParameterConfig;
}
impl<'a> ViewRepr for RMSDyadReLUBlockParameterMutView<'a> {
    type View<'b> = RMSDyadReLUBlockParameterView<'b>
    where Self:'b;
    fn view_repr<'b>(&'b self) -> Self::View<'b> {
        RMSDyadReLUBlockParameterView{
            rms_params:self.rms_params.view(),
            dyad_block_params:self.dyad_block_params.view_repr(),
        }
    }
}

pub struct RMSDyadReLUBlockParameter {
    pub rms_params: Array1<f32>,
    pub dyad_block_params: DyadBlockParameter,
}
impl<'a> AddAssign for RMSDyadReLUBlockParameter {
    fn add_assign(&mut self, rhs: Self) {
        self.rms_params += &rhs.rms_params;
        self.dyad_block_params += rhs.dyad_block_params;
    }
}
impl BlockType for RMSDyadReLUBlockParameter {
    type Config = RMSDyadReLUBlockParameterConfig;
}

impl BlockTypeAllocate for RMSDyadReLUBlockParameter {
    fn allocate(config:&Self::Config) -> Self {        
        let dyad_dim = config.dyad_block_parameter_config.dyad_dim;
        let dim_in = config.dyad_block_parameter_config.dim_in;
        let rms_params = Array1::<f32>::ones(dyad_dim*dim_in);
        let dyad_block_params = DyadBlockParameter::allocate(&config.dyad_block_parameter_config);
        RMSDyadReLUBlockParameter{
            rms_params,
            dyad_block_params,
        }
    }
}

impl ViewRepr for RMSDyadReLUBlockParameter {
    type View<'a> = RMSDyadReLUBlockParameterView<'a>
    where Self:'a;
    fn view_repr<'a>(&'a self) -> Self::View<'a> {
        RMSDyadReLUBlockParameterView{
            rms_params:self.rms_params.view(),
            dyad_block_params:self.dyad_block_params.view_repr(),
        }
    }
}

impl ValidateRepr for RMSDyadReLUBlockParameter {
    fn validate<'b>(config: &Self::Config,view: &RMSDyadReLUBlockParameterView<'b>) -> bool {
        let mut flag = true;
        let dyad_dim = config.dyad_block_parameter_config.dyad_dim;
        let dim_in = config.dyad_block_parameter_config.dim_in;
        if view.rms_params.len()!= dyad_dim*dim_in {
            flag = false;
        }
        if !DyadBlockParameter::validate(&config.dyad_block_parameter_config, &view.dyad_block_params){
            flag = false;
        }
        flag
    }
}

impl ViewMutRepr for RMSDyadReLUBlockParameter {
    type ViewMut<'a> = RMSDyadReLUBlockParameterMutView<'a>
    where Self:'a;
    fn view_mut_repr<'a>(&'a mut self) -> Self::ViewMut<'a> {
        RMSDyadReLUBlockParameterMutView{
            rms_params:self.rms_params.view_mut(),
            dyad_block_params:self.dyad_block_params.view_mut_repr(),
        }
    }
}