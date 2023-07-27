use ndarray::{Array3,ArrayView3,ArrayViewMut3};
use ndarray::{Array1,ArrayView1,ArrayViewMut1};
use crate::types::types::{BlockType,BlockTypeAllocate, ValidateRepr};
use crate::types::types::{ViewRepr,ViewMutRepr};
use crate::types::types::BlockConfig;
use std::ops::AddAssign;

#[derive(Clone,Copy,Debug)]
pub struct DyadBlockParameterConfig {
    pub dyad_dim: usize,
    pub dim_in: usize,
    pub dim_out: usize,
    pub has_bias: bool,
}
impl BlockConfig for DyadBlockParameterConfig {}

impl DyadBlockParameterConfig {
    pub fn new(dyad_dim: usize, dim_in: usize, 
        dim_out: usize, has_bias: bool) -> Self {
            DyadBlockParameterConfig { 
            dyad_dim, 
            dim_in, 
            dim_out,
            has_bias,
        }
    }
}

pub struct DyadBlockParameterView<'a> {
    pub w_upper: ArrayView3<'a,f32>,
    pub w_lower: ArrayView3<'a,f32>,
    pub bias: Option<ArrayView1<'a,f32>>,
}
impl<'a> BlockType for DyadBlockParameterView<'a> {
    type Config = DyadBlockParameterConfig;
}

pub struct DyadBlockParameterMutView<'a> {
    pub w_upper: ArrayViewMut3<'a,f32>,
    pub w_lower: ArrayViewMut3<'a,f32>,
    pub bias: Option<ArrayViewMut1<'a,f32>>,
}
impl<'a> AddAssign<DyadBlockParameter> for DyadBlockParameterMutView<'a> {
    fn add_assign(&mut self, rhs: DyadBlockParameter) {
        self.w_upper += &rhs.w_upper;
        self.w_lower += &rhs.w_lower;
        match (&mut self.bias,&rhs.bias) {
            (Some(b),Some(rhs_b)) => {
                *b += rhs_b;
            },
            _ => {},
        }
    }
}
impl<'a> BlockType for DyadBlockParameterMutView<'a> {
    type Config = DyadBlockParameterConfig;
}
impl<'a> ViewRepr for DyadBlockParameterMutView<'a> {
    type View<'b> = DyadBlockParameterView<'b>
    where Self:'b;
    fn view_repr<'b>(&'b self) -> Self::View<'b> {
        DyadBlockParameterView{
            w_upper:self.w_upper.view(),
            w_lower:self.w_lower.view(),
            bias: self.bias.view_repr(),
        }
    }
}

pub struct DyadBlockParameter {
    pub w_upper: Array3<f32>,
    pub w_lower: Array3<f32>,
    pub bias: Option<Array1<f32>>,
}
impl<'a> AddAssign for DyadBlockParameter {
    fn add_assign(&mut self, rhs: Self) {
        self.w_upper += &rhs.w_upper;
        self.w_lower += &rhs.w_lower;
        match (&mut self.bias,&rhs.bias) {
            (Some(b),Some(rhs_b)) => {
                *b += rhs_b;
            },
            _ => {},
        }
    }
}
impl BlockType for DyadBlockParameter {
    type Config = DyadBlockParameterConfig;
}

impl BlockTypeAllocate for DyadBlockParameter {
    fn allocate(config:&Self::Config) -> Self {        
        let dyad_dim = config.dyad_dim;
        let dim_in = config.dim_in;
        let dim_out = config.dim_out;
        let shape = (dyad_dim,dim_out,dim_in);
        let w_upper = Array3::<f32>::zeros(shape);
        let w_lower = Array3::<f32>::zeros(shape);
        let bias = match config.has_bias {
            true => Some(Array1::<f32>::zeros(dyad_dim*dim_out)),
            false => None,
        };
        DyadBlockParameter { 
            w_upper, 
            w_lower,
            bias
        }
    }
}

impl ViewRepr for DyadBlockParameter {
    type View<'a> = DyadBlockParameterView<'a>
    where Self:'a;
    fn view_repr<'a>(&'a self) -> Self::View<'a> {
        DyadBlockParameterView{
            w_upper:self.w_upper.view(),
            w_lower:self.w_lower.view(),
            bias:self.bias.view_repr(),
        }
    }
}

impl ValidateRepr for DyadBlockParameter {
    fn validate<'b>(config: &Self::Config,view: &DyadBlockParameterView<'b>) -> bool {
        let mut flag = true;
        let shape = [config.dyad_dim,config.dim_in,
            config.dim_out];
        if view.w_lower.shape()!=&shape {
            flag = false;
        }
        if view.w_upper.shape()!=&shape {
            flag = false;
        }
        match &view.bias {
            Some(b) => {
                if !config.has_bias {
                    flag = false;
                }
                else if b.shape()[0]!= config.dyad_dim*config.dim_in{
                    flag = false;
                }
            },
            None => {
                if config.has_bias {
                    flag = false;
                }
            },
        }
        flag
    }
}

impl ViewMutRepr for DyadBlockParameter {
    type ViewMut<'a> = DyadBlockParameterMutView<'a>
    where Self:'a;
    fn view_mut_repr<'a>(&'a mut self) -> Self::ViewMut<'a> {
        DyadBlockParameterMutView{
            w_upper:self.w_upper.view_mut(),
            w_lower:self.w_lower.view_mut(),
            bias:self.bias.view_mut_repr(),
        }
    }
}