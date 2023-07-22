use super::types::{ViewRepr,ViewMutRepr};

impl ViewRepr for f32
{
    type View<'a> = &'a f32
    where Self:'a;
    fn view_repr<'a>(&'a self) -> Self::View<'a> {
        self
    }
}

impl ViewRepr for &mut f32
{
    type View<'b> = &'b f32
    where Self:'b;
    fn view_repr<'b>(&'b self) -> Self::View<'b> {
        self
    }
}

impl ViewMutRepr for f32
{
    type ViewMut<'a> = &'a mut f32
    where Self:'a;
    fn view_mut_repr<'a>(&'a mut self) -> Self::ViewMut<'a> {
        self
    }
}