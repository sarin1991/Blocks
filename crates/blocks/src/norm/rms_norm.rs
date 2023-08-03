use ndarray::{Array2, ArrayViewMut2};
use ndarray::{Array1, ArrayViewMut1, ArrayView1};
use ndarray::{Zip,Axis};
use crate::block::FusedBlock;

const EPS:f32 = 0.000001;

pub struct RMSNormBlock {
    pub dim:usize,
    pub chunk_size:usize,
}

impl RMSNormBlock {
    pub fn new(dim:usize,chunk_size:usize) -> Self {
        Self { dim, chunk_size }
    }
}

impl FusedBlock for RMSNormBlock{
    type P = Array1<f32>;
    type I = Array2<f32>;
    type A = f32;
    type F = Array1<f32>;
    fn forward<'p,'io,'a,'f>(&self, 
        parameters:&ArrayView1<f32>,
        input_output:&mut ArrayViewMut2<f32>,
        _allocations:&mut &mut f32,
        forward_context:&mut ArrayViewMut1<f32>) {
        let in_out_iter = input_output.axis_chunks_iter_mut(Axis(1),self.chunk_size);
        let fc_iter = forward_context.axis_chunks_iter_mut(Axis(0),self.chunk_size);
        in_out_iter.zip(fc_iter)
        .for_each(|(mut in_out,
            mut fc)|{
            let chunk_size = in_out.shape()[1];
            let rms_slice = &mut fc.as_slice_mut().unwrap()[0..chunk_size];
            for i in 0..chunk_size {
                rms_slice[i] = 0.0f32;
            }
            in_out.axis_iter(Axis(0))
            .for_each(|in_out| {
                let in_out_slice = &in_out.as_slice().unwrap()[0..chunk_size];
                for i in 0..chunk_size {
                    rms_slice[i] += in_out_slice[i]*in_out_slice[i];
                }
            });
            // calc rms
            for i in 0..chunk_size {
                let ms = rms_slice[i]/(self.dim as f32);
                rms_slice[i] = (1f32/(ms.max(EPS))).sqrt();
            }
            // calc out
            Zip::from(in_out.axis_iter_mut(Axis(0)))
            .and(parameters)
            .for_each(|mut in_out, param|{
                let in_out_slice = &mut in_out.as_slice_mut().unwrap()[0..chunk_size];
                for i in 0..chunk_size {
                    in_out_slice[i] = param*rms_slice[i]*in_out_slice[i];
                }
            });
        });
    }
    fn backward<'gp,'gio,'io,'p,'f>(&self,
        parameter_gradients:&mut ArrayViewMut1<f32>,
        input_output_gradients:&mut ArrayViewMut2<f32>,
        input_output:&mut ArrayViewMut2<f32>,
        parameters:&ArrayView1<f32>,
        _allocations:&mut &mut f32,
        forward_context:&ArrayView1<f32>){
        // calc param grad
        Zip::from(parameter_gradients)
        .and(input_output_gradients.axis_iter(Axis(0)))
        .and(input_output.axis_iter(Axis(0)))
        .and(parameters)
        .for_each(|p_grad, in_out_grad, 
            in_out, p|{
            let l = in_out_grad.len();
            let in_out_grad_slice = &in_out_grad.as_slice().unwrap()[0..l];
            let in_out_slice = &in_out.as_slice().unwrap()[0..l];
            for i in 0..l {
                *p_grad += in_out_grad_slice[i] * in_out_slice[i]/(*p);
            }
        });
        // calc input grad
        let mut out_grad_output_inner = vec![0f32;self.chunk_size];
        let in_out_iter = input_output.axis_chunks_iter_mut(Axis(1), self.chunk_size);
        let in_out_grad_iter = input_output_gradients.axis_chunks_iter_mut(Axis(1), self.chunk_size);
        let rms_iter = forward_context.axis_chunks_iter(Axis(0), self.chunk_size);
        in_out_iter.zip(in_out_grad_iter).zip(rms_iter)
        .for_each(|((mut in_out,mut in_out_grad),
            rms)|{
            let dim = in_out.shape()[0] as f32;
            let chunk_size = in_out.shape()[1];
            let out_grad_output_inner = &mut out_grad_output_inner.as_mut_slice()[0..chunk_size];
            out_grad_output_inner.fill(0.0f32);
            // calc out_grad_output_inner
            Zip::from(in_out_grad.axis_iter(Axis(0)))
            .and(in_out.axis_iter(Axis(0)))
            .for_each(|in_out_grad,in_out|{
                let in_out_grad_slice = &in_out_grad.as_slice().unwrap()[0..chunk_size];
                let in_out_slice = &in_out.as_slice().unwrap()[0..chunk_size];
                for i in 0..chunk_size {
                    out_grad_output_inner[i] += in_out_grad_slice[i]*in_out_slice[i];
                }
            });
            // calc input
            let rms_slice = &rms.as_slice().unwrap()[0..chunk_size];
            Zip::from(in_out.axis_iter_mut(Axis(0)))
            .and(parameters)
            .for_each(|mut in_out,param|{
                let in_out_slice = &mut in_out.as_slice_mut().unwrap()[0..chunk_size];
                for i in 0..chunk_size {
                    in_out_slice[i] = in_out_slice[i]/(param*rms_slice[i]);
                }
            });
            // calc input grad
            Zip::from(in_out_grad.axis_iter_mut(Axis(0)))
            .and(in_out.axis_iter(Axis(0)))
            .and(parameters)
            .for_each(|mut in_out_grad, in_out,param|{
                in_out_grad.iter_mut().zip(in_out).zip(rms_slice.iter()).zip(out_grad_output_inner.iter())
                .for_each(|(((in_out_grad, in_out),rms),out_grad_output_inner)|{
                    *in_out_grad = -(*out_grad_output_inner)*(*in_out)*(*rms)*(*rms)/dim + (*in_out_grad)*(*param)*(*rms);
                });
            });
        });
    }
}