use crate::types::types::{ViewRepr,ViewMutRepr};

pub trait Block {
    type P:ViewMutRepr;
    type A:ViewMutRepr;
    type I:ViewMutRepr;
    type F:ViewMutRepr;
    type O:ViewMutRepr;
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

pub trait EqualBlock {
    type P:ViewMutRepr;
    type A:ViewMutRepr;
    type I:ViewMutRepr;
    type F:ViewMutRepr;
    fn forward<'p,'i,'o,'a,'f>(&self, 
        parameters:&<Self::P as ViewRepr>::View<'p>,
        input:&<Self::I as ViewRepr>::View<'i>,
        output:&mut <Self::I as ViewMutRepr>::ViewMut<'o>,
        allocations:&mut <Self::A as ViewMutRepr>::ViewMut<'a>,
        forward_context:&mut <Self::F as ViewMutRepr>::ViewMut<'f>);
    fn backward<'gp,'gi,'go,'o,'i,'p,'a,'f>(&self,
        parameter_gradients:&mut <Self::P as ViewMutRepr>::ViewMut<'gp>,
        input_gradients:&mut <Self::I as ViewMutRepr>::ViewMut<'gi>,
        output_gradients:&<Self::I as ViewRepr>::View<'go>,
        output:&<Self::I as ViewRepr>::View<'o>,
        input:&<Self::I as ViewRepr>::View<'i>,
        parameters:&<Self::P as ViewRepr>::View<'p>,
        allocations:&mut <Self::A as ViewMutRepr>::ViewMut<'a>,
        forward_context:&<Self::F as ViewRepr>::View<'f>
    );
}

impl<T> Block for T
where T: EqualBlock
{
    type P = <T as EqualBlock>::P;
    type A = <T as EqualBlock>::A;
    type I = <T as EqualBlock>::I;
    type F = <T as EqualBlock>::F;
    type O = <T as EqualBlock>::I;
    fn forward<'p,'i,'o,'a,'f>(&self, 
            parameters:&<Self::P as ViewRepr>::View<'p>,
            input:&<Self::I as ViewRepr>::View<'i>,
            output:&mut <Self::O as ViewMutRepr>::ViewMut<'o>,
            allocations:&mut <Self::A as ViewMutRepr>::ViewMut<'a>,
            forward_context:&mut <Self::F as ViewMutRepr>::ViewMut<'f>) {
        self.forward(parameters, input, output, allocations, forward_context);
    }
    fn backward<'gp,'gi,'go,'o,'i,'p,'a,'f>(&self,
            parameter_gradients:&mut <Self::P as ViewMutRepr>::ViewMut<'gp>,
            input_gradients:&mut <Self::I as ViewMutRepr>::ViewMut<'gi>,
            output_gradients:&<Self::O as ViewRepr>::View<'go>,
            output:&<Self::O as ViewRepr>::View<'o>,
            input:&<Self::I as ViewRepr>::View<'i>,
            parameters:&<Self::P as ViewRepr>::View<'p>,
            allocations:&mut <Self::A as ViewMutRepr>::ViewMut<'a>,
            forward_context:&<Self::F as ViewRepr>::View<'f>
        ) {
        self.backward(parameter_gradients, input_gradients, output_gradients, 
            output, input, parameters, allocations, forward_context)
    }
}

pub trait FusedBlock {
    type P:ViewMutRepr;
    type I:ViewMutRepr;
    type F:ViewMutRepr;
    fn forward<'p,'io,'a,'f>(&self, 
        parameters:&<Self::P as ViewRepr>::View<'p>,
        input_output:&mut <Self::I as ViewMutRepr>::ViewMut<'io>,
        forward_context:&mut <Self::F as ViewMutRepr>::ViewMut<'f>);
    fn backward<'gp,'gio,'io,'p,'f>(&self,
        parameter_gradients:&mut <Self::P as ViewMutRepr>::ViewMut<'gp>,
        input_output_gradients:&mut <Self::I as ViewMutRepr>::ViewMut<'gio>,
        input_output:&<Self::I as ViewRepr>::View<'io>,
        parameters:&<Self::P as ViewRepr>::View<'p>,
        forward_context:&<Self::F as ViewRepr>::View<'f>
    );
}