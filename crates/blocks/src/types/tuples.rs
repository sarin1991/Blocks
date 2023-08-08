use super::types::{ViewRepr,ViewMutRepr,BlockTypeAllocate,BlockType,BlockConfig};

impl<T1,T2> ViewRepr for (T1,T2) 
where 
    T1 : ViewRepr,
    T2 : ViewRepr,
{
    type View<'a> = (T1::View<'a>,T2::View<'a>)
    where Self:'a;
    fn view_repr<'a>(&'a self) -> Self::View<'a> {
        (self.0.view_repr(),self.1.view_repr())
    }
}

impl<T1,T2,T3> ViewRepr for (T1,T2,T3) 
where 
    T1 : ViewRepr,
    T2 : ViewRepr,
    T3 : ViewRepr,
{
    type View<'a> = (T1::View<'a>,T2::View<'a>,T3::View<'a>)
    where Self:'a;
    fn view_repr<'a>(&'a self) -> Self::View<'a> {
        (self.0.view_repr(),self.1.view_repr(),self.2.view_repr())
    }
}

impl<T1,T2,T3,T4> ViewRepr for (T1,T2,T3,T4) 
where 
    T1 : ViewRepr,
    T2 : ViewRepr,
    T3 : ViewRepr,
    T4 : ViewRepr,
{
    type View<'a> = (T1::View<'a>,T2::View<'a>,T3::View<'a>,T4::View<'a>)
    where Self:'a;
    fn view_repr<'a>(&'a self) -> Self::View<'a> {
        (self.0.view_repr(),self.1.view_repr(),self.2.view_repr(),self.3.view_repr())
    }
}

impl<T1,T2> ViewMutRepr for (T1,T2) 
where 
    T1 : ViewMutRepr,
    T2 : ViewMutRepr,
{
    type ViewMut<'a> = (T1::ViewMut<'a>,T2::ViewMut<'a>)
    where Self:'a;
    fn view_mut_repr<'a>(&'a mut self) -> Self::ViewMut<'a> {
        (self.0.view_mut_repr(),self.1.view_mut_repr())
    }
}

impl<T1,T2,T3> ViewMutRepr for (T1,T2,T3) 
where 
    T1 : ViewMutRepr,
    T2 : ViewMutRepr,
    T3 : ViewMutRepr,
{
    type ViewMut<'a> = (T1::ViewMut<'a>,T2::ViewMut<'a>,T3::ViewMut<'a>)
    where Self:'a;
    fn view_mut_repr<'a>(&'a mut self) -> Self::ViewMut<'a> {
        (self.0.view_mut_repr(),self.1.view_mut_repr(),self.2.view_mut_repr())
    }
}

impl<T1,T2,T3,T4> ViewMutRepr for (T1,T2,T3,T4) 
where 
    T1 : ViewMutRepr,
    T2 : ViewMutRepr,
    T3 : ViewMutRepr,
    T4 : ViewMutRepr,
{
    type ViewMut<'a> = (T1::ViewMut<'a>,T2::ViewMut<'a>,T3::ViewMut<'a>,T4::ViewMut<'a>)
    where Self:'a;
    fn view_mut_repr<'a>(&'a mut self) -> Self::ViewMut<'a> {
        (self.0.view_mut_repr(),self.1.view_mut_repr(),self.2.view_mut_repr(),self.3.view_mut_repr())
    }
}

impl<T1,T2> BlockConfig for (T1,T2) 
where 
    T1 : BlockConfig,
    T2 : BlockConfig,
{}

impl<T1,T2> BlockType for (T1,T2) 
where 
    T1 : BlockType,
    T2 : BlockType,
{
    type Config = (T1::Config,T2::Config);
}

impl<T1,T2> BlockTypeAllocate for (T1,T2) 
where 
    T1 : BlockTypeAllocate,
    T2 : BlockTypeAllocate,
{
    fn allocate(config:&Self::Config) -> Self {
        (T1::allocate(&config.0),T2::allocate(&config.1))
    }
}