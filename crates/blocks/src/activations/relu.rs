use ndarray::{Array2, ArrayViewMut2, ArrayView2};

use crate::block::FusedBlock;

pub struct ReLuBlock {
}

impl ReLuBlock {
    pub fn new() -> Self {
    }
}

impl FusedBlock for ReLuBlock{
    type P = f32;
    type I = Array2<f32>;
    type F = f32;
    fn forward<'p,'io,'a,'f>(&self, 
        parameters:&&f32,
        input_output:&mut ArrayViewMut2<f32>,
        forward_context:&mut &mut f32) {
        let l = input_output.len();
        let input_output_slice = &mut input_output.as_slice_mut().unwrap()[0..l];
        for i in 0..l {
            input_output_slice[i] = input_output_slice[i].max(0.0f32);
        }
    }
    fn backward<'gp,'gio,'io,'p,'f>(&self,
        parameter_gradients:&mut &mut f32,
        input_output_gradients:&mut ArrayViewMut2<f32>,
        input_output:&ArrayView2<f32>,
        parameters:&&f32,
        forward_context:&&f32){
        let l = input_output.len();
        let input_output_slice = &input_output.as_slice().unwrap()[0..l];
        let input_output_grad_slice = &mut input_output_gradients.as_slice_mut().unwrap()[0..l];
        for i in 0..l {
            let val = input_output_slice[i];
            if val==0.0f32 {
                input_output_gradients[i] = 0.0f32;
            }
        }
    }
}