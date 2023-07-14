use crate::types::{BlockType,ViewRepr,ViewMutRepr};

pub trait Block : BlockType {
    type P:BlockType<Config = Self::Config> + ViewMutRepr;
    type A:BlockType<Config = Self::Config> + ViewMutRepr;
    type I:BlockType<Config = Self::Config> + ViewMutRepr;
    type F:BlockType<Config = Self::Config> + ViewMutRepr;
    type O:BlockType<Config = Self::Config> + ViewMutRepr;
    fn forward<'p,'i,'o,'a,'f>(&self, 
        parameters:&<Self::P as ViewRepr>::View<'p>,
        input:&<Self::I as ViewRepr>::View<'i>,
        output:&mut <Self::O as ViewMutRepr>::ViewMut<'o>,
        allocations:&mut <Self::A as ViewMutRepr>::ViewMut<'a>,
        forward_context:&mut <Self::F as ViewMutRepr>::ViewMut<'f>);
    fn backward<'gp,'gi,'go,'o,'i,'p,'a,'f>(&self,
        parameter_gradients:&mut <Self::P as ViewMutRepr>::ViewMut<'gp>,
        input_gradients:&mut <Self::I as ViewMutRepr>::ViewMut<'gi>,
        output_gradients:&<Self::O as ViewRepr>::View<'go>,
        output:&<Self::O as ViewRepr>::View<'o>,
        input:&<Self::I as ViewRepr>::View<'i>,
        parameters:&<Self::P as ViewRepr>::View<'p>,
        allocations:&mut <Self::A as ViewMutRepr>::ViewMut<'a>,
        forward_context:&<Self::F as ViewRepr>::View<'f>
    );
}
