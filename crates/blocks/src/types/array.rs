use ndarray::{Array,ArrayView,ArrayViewMut,Dim,Dimension, IntoDimension};
use super::types::{BlockConfig,BlockType,BlockTypeAllocate,ViewRepr,ViewMutRepr,ValidateRepr};

impl<const L: usize> BlockConfig for [usize;L] {}

impl<const L: usize> BlockType for Array<f32,Dim<[usize;L]>> {
    type Config = [usize;L];
}

impl<'a,const L: usize> BlockType for ArrayViewMut<'a,f32,Dim<[usize;L]>> {
    type Config = [usize;L];
}

impl<const L: usize> BlockTypeAllocate for Array<f32,Dim<[usize;L]>> 
where 
    Dim<[usize;L]>:Dimension,
    [usize;L]:IntoDimension<Dim = Dim<[usize;L]>>,
{
    fn allocate(config:&[usize;L]) -> Self {
        Array::<f32,Dim<[usize;L]>>::zeros(*config)
    }
}

impl<const L: usize> ViewRepr for Array<f32,Dim<[usize;L]>> 
where Dim<[usize;L]>:Dimension,
{
    type View<'a> = ArrayView<'a,f32,Dim<[usize;L]>>
    where Self:'a;
    fn view_repr<'a>(&'a self) -> Self::View<'a> {
        self.view()
    }
}

impl<'a,const L: usize> ViewRepr for ArrayViewMut<'a,f32,Dim<[usize;L]>> 
where Dim<[usize;L]>:Dimension,
{
    type View<'b> = ArrayView<'b,f32,Dim<[usize;L]>>
    where Self:'b;
    fn view_repr<'b>(&'b self) -> Self::View<'b> {
        self.view()
    }
}

impl<const L: usize> ViewMutRepr for Array<f32,Dim<[usize;L]>> 
where Dim<[usize;L]>:Dimension,
{
    type ViewMut<'a> = ArrayViewMut<'a,f32,Dim<[usize;L]>>
    where Self:'a;
    fn view_mut_repr<'a>(&'a mut self) -> Self::ViewMut<'a> {
        self.view_mut()
    }
}

impl<const L: usize> ValidateRepr for Array<f32,Dim<[usize;L]>> 
where Dim<[usize;L]>:Dimension,
{
    fn validate<'a>(config: &Self::Config,view: &Self::View<'a>) -> bool {
        let shape = view.shape();
        let flag = if shape==config {
            true
        }
        else {
            false
        };
        flag
    }
}