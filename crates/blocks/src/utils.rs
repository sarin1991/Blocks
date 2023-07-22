use ndarray::{ArrayViewMut2, ArrayView2};
use ndarray::{Zip,Axis};

pub fn copy_array(output:&mut ArrayViewMut2<f32>, input:&ArrayView2<f32>) {
    Zip::from(output.axis_iter_mut(Axis(0)))
    .and(input.axis_iter(Axis(0)))
    .for_each(|mut output,input|{
        output.as_slice_mut().unwrap().copy_from_slice(input.as_slice().unwrap());
    });
}