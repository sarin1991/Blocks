use super::types::{BlockType,ViewRepr,ViewMutRepr,ValidateRepr};

impl<T> BlockType for Option<T>
where T:BlockType {
    type Config = T::Config;
}

impl<T> ViewRepr for Option<T> 
where T:ViewRepr,
{
    type View<'a> = Option<T::View<'a>>
    where Self:'a;
    fn view_repr<'a>(&'a self) -> Self::View<'a> {
        match self {
            Some(s) => Some(s.view_repr()),
            None => None,
        }
    }
}

impl<T: ViewMutRepr> ViewMutRepr for Option<T> {
    type ViewMut<'a> = Option<T::ViewMut<'a>>
    where Self:'a;
    fn view_mut_repr<'a>(&'a mut self) -> Self::ViewMut<'a> {
        match self {
            Some(s) => Some(s.view_mut_repr()),
            None => None,
        }
    }
}

impl<T:ValidateRepr> ValidateRepr for Option<T> 
where T:ValidateRepr,
{
    fn validate<'a>(config: &Self::Config,view: &Self::View<'a>) -> bool {
        match view {
            Some(v) => <T as ValidateRepr>::validate(config,v),
            None => true, 
        }
    }
}